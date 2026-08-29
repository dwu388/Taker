from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

from .axiom_ocr_hybrid import (
    HybridOCR, OCRDecision, parse_dev_pair, parse_duration, parse_fee, parse_integer,
    parse_money, parse_paid, parse_pct,
)
from .common import load_json, normalize_token_key, parse_duration_minutes


def _scale(v: float, scale: float) -> int:
    return int(round(v * scale))


def detect_thumbnail_anchors(image: np.ndarray, cfg: dict[str, Any]) -> list[tuple[int,int,int,int]]:
    """Find independent thumbnail anchors; card count is whatever the image actually contains."""
    h, w = image.shape[:2]
    ref_w = float(cfg.get("reference_image_width", 844))
    scale = w / ref_w
    x0f, x1f = cfg.get("anchor_region_x", [0.01, .16])
    x0, x1 = int(w*x0f), int(w*x1f)
    roi = image[:, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Thumbnail images have local visual complexity in a narrow left column.
    sq = cv2.boxFilter(gray.astype(np.float32)**2, -1, (_scale(15,scale), _scale(15,scale)))
    mean = cv2.boxFilter(gray.astype(np.float32), -1, (_scale(15,scale), _scale(15,scale)))
    var = np.maximum(0, sq - mean**2)
    mask = (var >= float(cfg.get("thumbnail_min_variance", 350.0))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_scale(9,scale), _scale(9,scale)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minw,minh = cfg.get("thumbnail_min_size_px", [42,42])
    maxw,maxh = cfg.get("thumbnail_max_size_px", [92,92])
    candidates = []
    for c in contours:
        x,y,cw,ch = cv2.boundingRect(c)
        if _scale(minw,scale) <= cw <= _scale(maxw,scale)*2 and _scale(minh,scale) <= ch <= _scale(maxh,scale)*2:
            candidates.append((x+x0,y,cw,ch))
    # Cluster by y and keep strongest/largest anchor in each card band.
    candidates.sort(key=lambda r:r[1])
    dedupe = _scale(cfg.get("dedupe_anchor_y_px",55), scale)
    anchors: list[tuple[int,int,int,int]] = []
    for r in candidates:
        if anchors and abs(r[1]-anchors[-1][1]) < dedupe:
            if r[2]*r[3] > anchors[-1][2]*anchors[-1][3]:
                anchors[-1] = r
        else:
            anchors.append(r)
    # Fallback: row-edge projection if thumbnail complexity detection failed entirely.
    if not anchors:
        edge = cv2.Canny(gray, 50, 120)
        proj = edge.sum(axis=1)
        threshold = np.percentile(proj, 78)
        ys = np.where(proj >= threshold)[0]
        groups=[]
        for y in ys:
            if not groups or y-groups[-1][-1] > _scale(20,scale): groups.append([int(y)])
            else: groups[-1].append(int(y))
        card_h = _scale(cfg.get("card_height_px",142), scale)
        for g in groups:
            y = int(np.median(g))
            if not anchors or y-anchors[-1][1] > int(card_h*.65):
                anchors.append((x0,y,_scale(60,scale),_scale(60,scale)))
    return anchors


def crop_cards(image: np.ndarray, anchors: list[tuple[int,int,int,int]], cfg: dict[str, Any]) -> list[tuple[np.ndarray, dict[str,int]]]:
    h,w=image.shape[:2]
    scale=w/float(cfg.get("reference_image_width",844))
    card_h=_scale(cfg.get("card_height_px",142),scale)
    offset=_scale(cfg.get("anchor_to_card_top_px",-6),scale)
    cards=[]
    for x,y,cw,ch in anchors:
        top=max(0,y+offset)
        bottom=min(h,top+card_h)
        if bottom-top < int(card_h*.8):
            continue
        cards.append((image[top:bottom,0:w].copy(), {"left":0,"top":top,"right":w,"bottom":bottom,"anchor_x":x,"anchor_y":y}))
    return cards


def crop_cell(card: np.ndarray, box: list[float]) -> np.ndarray:
    h,w=card.shape[:2]
    x0,y0,x1,y1=box
    a=max(0,int(round(x0*w))); b=max(0,int(round(y0*h)))
    c=min(w,int(round(x1*w))); d=min(h,int(round(y1*h)))
    return card[b:d,a:c].copy()


def simple_text(cell: np.ndarray, whitelist: str | None = None, psm: int = 7) -> tuple[str,float]:
    scale=max(2.0, 48/max(1,cell.shape[0]))
    up=cv2.resize(cell,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(up,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(2.0,(4,4)).apply(gray)
    conf=f"--psm {psm}"
    if whitelist: conf += f" -c tessedit_char_whitelist={whitelist}"
    d=pytesseract.image_to_data(clahe,config=conf,output_type=pytesseract.Output.DICT)
    parts=[]; cs=[]
    for t,c in zip(d.get("text",[]),d.get("conf",[])):
        t=str(t).strip()
        if not t: continue
        parts.append(t)
        try:
            cc=float(c)
            if cc>=0: cs.append(cc/100)
        except: pass
    return " ".join(parts).strip(), max(cs) if cs else 0.0


def _decision_value(dec: OCRDecision, row: dict[str,Any], field: str):
    row[field]=dec.value
    row.setdefault("field_confidence",{})[field]=dec.confidence
    row.setdefault("ocr_diagnostics",{})[field]=dec.to_dict()
    row.setdefault("field_source",{})[field]="hybrid_ocr" if dec.value is not None else "missing"


def extract_card(card: np.ndarray, cfg: dict[str,Any], ocr: HybridOCR, debug_dir: Path | None=None, index: int=0) -> dict[str,Any]:
    cells_cfg=cfg["cells"]
    cells={k:crop_cell(card,v) for k,v in cells_cfg.items()}
    if debug_dir:
        rd=debug_dir/f"row_{index:02d}"; rd.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(rd/"card.png"),card)
        for k,img in cells.items(): cv2.imwrite(str(rd/f"{k}.png"),img)
    row:dict[str,Any]={"field_confidence":{},"ocr_diagnostics":{},"field_source":{}}
    name,nc=simple_text(cells["name"],psm=7); row["name"]=name or None; row["field_confidence"]["name"]=nc
    addr,ac=simple_text(cells["short_address_hint"],whitelist="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz....",psm=7)
    am=re.search(r"([1-9A-HJ-NP-Za-km-z]{2,10}\.{2,3}[1-9A-HJ-NP-Za-km-z]{2,10})",addr)
    row["short_address_hint"]=am.group(1) if am else (addr or None); row["field_confidence"]["short_address_hint"]=ac
    age_low=tuple(cfg.get("ocr",{}).get("age_hsv_low",[35,55,90])); age_high=tuple(cfg.get("ocr",{}).get("age_hsv_high",[105,255,255]))
    age_dec=ocr.recognize(cells["age"],parse_duration,whitelist="0123456789smhdwoy",require_agreement=False,hsv_mask=(age_low,age_high)); _decision_value(age_dec,row,"age_minutes")
    reuse_dec=ocr.recognize(cells["image_reuse_count"],parse_integer,whitelist="0123456789",require_agreement=True); _decision_value(reuse_dec,row,"image_reuse_count")

    for field in ("market_cap_usd","volume_usd"):
        dec=ocr.recognize(cells[field],parse_money,whitelist="$0123456789.,KMBkmb",require_agreement=False); _decision_value(dec,row,field)
    fee=ocr.recognize(cells["fees_sol"],parse_fee,whitelist="0123456789.",require_agreement=True,decimal_guard=True); _decision_value(fee,row,"fees_sol")
    tx=ocr.recognize(cells["txns"],parse_integer,whitelist="TXtx0123456789,",require_agreement=True); _decision_value(tx,row,"txns")
    for field in ("holders","pro_traders","kols","recent_visitors"):
        dec=ocr.recognize(cells[field],parse_integer,whitelist="0123456789,",require_agreement=True); _decision_value(dec,row,field)
    dev=ocr.recognize(cells["dev_pair"],parse_dev_pair,whitelist="0123456789/",require_agreement=True)
    row["ocr_diagnostics"]["dev_pair"]=dev.to_dict(); row["field_confidence"]["dev_counts"]=dev.confidence
    if dev.value is not None: row["dev_migrations"],row["dev_creations"]=dev.value
    else: row["dev_migrations"]=row["dev_creations"]=None

    for field in ("top10_holders_pct","sniper_pct","insider_pct","bundler_pct"):
        dec=ocr.recognize(cells[field],parse_pct,whitelist="0123456789.%",require_agreement=True); _decision_value(dec,row,field)
    fund=ocr.recognize(cells["funding_time"],parse_duration,whitelist="0123456789smhdwoy",require_agreement=True); _decision_value(fund,row,"funding_time_minutes")
    ft,fc=simple_text(cells["funding_time"],whitelist="0123456789smhdwoy",psm=7)
    row["funding_time_raw"]=ft if parse_duration_minutes(ft.lower().replace(" ","")) is not None else None
    ds,dsc=simple_text(cells["tracked_dev_status_raw"],whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",psm=7)
    row["tracked_dev_status_raw"]=ds.upper() if re.fullmatch(r"[A-Z]{1,6}",ds.strip()) else None; row["field_confidence"]["tracked_dev_status_raw"]=dsc
    paid=ocr.recognize(cells["dex_paid"],parse_paid,whitelist="PaidPAIDpaid",require_agreement=False); _decision_value(paid,row,"dex_paid")

    # Hard semantic integrity checks, not trading opinions.
    h=row.get("holders")
    if h is not None and row.get("pro_traders") is not None and row["pro_traders"] > h:
        row["pro_traders"]=None; row["ocr_diagnostics"]["pro_traders"]["reason"]="semantic_contradiction_pro_gt_holders"
    if h is not None and row.get("kols") is not None and row["kols"] > h:
        row["kols"]=None; row["ocr_diagnostics"]["kols"]["reason"]="semantic_contradiction_kol_gt_holders"
    if row.get("dev_migrations") is not None and row.get("dev_creations") is not None and row["dev_migrations"]>row["dev_creations"]:
        row["dev_migrations"]=row["dev_creations"]=None
    row["token_key"]=normalize_token_key(row.get("name"),row.get("short_address_hint"))
    return row


def extract_screenshot(path: str|Path, config_path: str|Path="axiom_migrated_config.json", debug: bool=False, allow_tesseract_only: bool=False) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    cfg=load_json(config_path,{})
    image=cv2.imread(str(path))
    if image is None: raise FileNotFoundError(f"Could not read image: {path}")
    ocr=HybridOCR(require_rapid=bool(cfg.get("ocr",{}).get("require_rapidocr",True)) and not allow_tesseract_only)
    anchors=detect_thumbnail_anchors(image,cfg); cards=crop_cards(image,anchors,cfg)
    debug_dir=Path(path).with_suffix("").with_name(Path(path).stem+"_debug") if debug else None
    rows=[]
    for i,(card,geom) in enumerate(cards):
        row=extract_card(card,cfg,ocr,debug_dir=debug_dir,index=i); row["geometry"]=geom
        if row.get("token_key") and sum(row.get(k) is not None for k in ("market_cap_usd","volume_usd","txns","holders"))>=1:
            rows.append(row)
    report={"image":str(path),"anchors_detected":len(anchors),"cards_detected":len(cards),"rows_extracted":len(rows),"rapidocr_available":ocr.rapid is not None,"rapidocr_error":ocr.rapid_error}
    return rows,report


def main():
    ap=argparse.ArgumentParser(description="Extract cleaned Axiom Migrated semantic cards")
    ap.add_argument("image"); ap.add_argument("--config",default="axiom_migrated_config.json"); ap.add_argument("--debug",action="store_true"); ap.add_argument("--allow-tesseract-only",action="store_true")
    args=ap.parse_args(); rows,report=extract_screenshot(args.image,args.config,args.debug,args.allow_tesseract_only)
    print(json.dumps({"report":report,"rows":rows},indent=2,default=str))
if __name__=="__main__": main()
