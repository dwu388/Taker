from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .db import connect, migrate


def _matrix_row(feature:dict[str,Any], cols:list[str], medians:dict[str,float]):
    vals=[]
    for c in cols:
        v=feature.get(c)
        try: x=float(v) if v is not None and math.isfinite(float(v)) else float(medians.get(c,0.0))
        except Exception: x=float(medians.get(c,0.0))
        vals.append(x)
    return np.asarray([vals],dtype=np.float32)


def _predict_head(rec,X):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if rec["kind"]=="classification":
            return float((rec["lgb"].predict_proba(X)[:,1][0]+rec["xgb"].predict_proba(X)[:,1][0])/2)
        return float((rec["lgb"].predict(X)[0]+rec["xgb"].predict(X)[0])/2)


def load_current_rows(db_path:str, all_latest:bool=False):
    migrate(db_path); con=connect(db_path)
    try:
        if all_latest:
            rows=con.execute("""
                SELECT o.*, f.feature_json, v.visibility_bonus,v.capture_count,v.consecutive_capture_count,v.reappearance_count
                FROM axiom_observations o JOIN axiom_features_v18 f USING(observation_id)
                LEFT JOIN axiom_visibility v USING(token_key)
                JOIN (SELECT token_key,MAX(snapshot_at) mx FROM axiom_observations GROUP BY token_key) z ON z.token_key=o.token_key AND z.mx=o.snapshot_at
            """).fetchall()
        else:
            mx=con.execute("SELECT MAX(snapshot_at) FROM axiom_observations").fetchone()[0]
            rows=con.execute("""
                SELECT o.*, f.feature_json, v.visibility_bonus,v.capture_count,v.consecutive_capture_count,v.reappearance_count
                FROM axiom_observations o JOIN axiom_features_v18 f USING(observation_id)
                LEFT JOIN axiom_visibility v USING(token_key) WHERE o.snapshot_at=?
            """,(mx,)).fetchall()
        return [dict(r) for r in rows]
    finally: con.close()


def predict_rows(db_path:str, model_path:str, all_latest:bool=False)->list[dict[str,Any]]:
    bundle=joblib.load(model_path); cols=bundle["feature_columns"]
    rows=load_current_rows(db_path,all_latest); out=[]
    for r in rows:
        feat=json.loads(r["feature_json"]); pred={"token_key":r["token_key"],"token_address":r.get("token_address"),"name":r.get("name"),"short_address_hint":r.get("short_address_hint"),"snapshot_at":r["snapshot_at"],"market_cap_usd":r.get("market_cap_usd"),"visibility_bonus":float(r.get("visibility_bonus") or 0.0),"capture_count":r.get("capture_count")}
        for target,rec in bundle["heads"].items():
            X=_matrix_row(feat,cols,rec["medians"]); val=_predict_head(rec,X)
            if rec["kind"]=="classification": pred["p_"+target]=max(0.0,min(1.0,val))
            elif target=="log_peak_multiple_24h": pred["pred_peak_multiple_expected"]=math.exp(val)
            elif target=="log1p_time_to_peak_minutes_24h": pred["pred_time_to_peak_minutes_24h"]=max(0.0,math.expm1(val))
            elif target=="log1p_time_to_death_minutes_if_death_24h": pred["pred_time_to_death_minutes_if_death_24h"]=max(0.0,math.expm1(val))
            elif target=="drawdown_before_peak_pct_24h": pred["pred_drawdown_before_peak_pct_24h"]=max(0.0,val)
            elif target=="peak_to_trough_before_peak_pct_24h": pred["pred_peak_to_trough_before_peak_pct_24h"]=max(0.0,val)
            else: pred["pred_"+target]=val
        qvals=[]
        for q,rec in sorted(bundle.get("quantiles",{}).items()):
            X=_matrix_row(feat,cols,rec["medians"]); qvals.append((q,math.exp(_predict_head(rec,X))))
        if qvals:
            ordered=np.maximum.accumulate([v for _,v in qvals])
            for (q,_),v in zip(qvals,ordered):
                pred[f"pred_peak_multiple_{q}"]=float(v)
                if pred.get("market_cap_usd") is not None:
                    pred[f"pred_peak_market_cap_usd_{q}"]=float(v)*float(pred["market_cap_usd"])
        for h in ("1h","4h","12h"):
            p=pred.get(f"p_dead_within_{h}_24h")
            if p is not None: pred[f"p_survive_{h}"]=1.0-p
        out.append(pred)
    return out


def plan_handoff(db_path:str, predictions:list[dict[str,Any]], max_candidates:int=5, min_probability:float=.50)->dict[str,int]:
    primary="p_success_2x_before_50dd_or_dead_24h"
    fallback="p_hit_plus100_by_12h"
    ranked=[]
    for p in predictions:
        prob=p.get(primary)
        if prob is None: prob=p.get(fallback)
        if prob is None or float(prob)<min_probability: continue
        priority=min(1.0,float(prob)+float(p.get("visibility_bonus") or 0.0))
        ranked.append((priority,float(prob),p))
    ranked.sort(key=lambda x:x[0],reverse=True); selected=ranked[:max_candidates]
    con=connect(db_path)
    try:
        for priority,prob,p in selected:
            state="ready_future_api" if p.get("token_address") else "awaiting_mint"
            con.execute("""INSERT INTO axiom_api_handoff_queue(token_key,token_address,snapshot_at,model_probability,visibility_bonus,handoff_priority,state,prediction_json,updated_at)
                VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(token_key) DO UPDATE SET token_address=excluded.token_address,snapshot_at=excluded.snapshot_at,model_probability=excluded.model_probability,visibility_bonus=excluded.visibility_bonus,handoff_priority=excluded.handoff_priority,state=excluded.state,prediction_json=excluded.prediction_json,updated_at=CURRENT_TIMESTAMP""",
                (p["token_key"],p.get("token_address"),p["snapshot_at"],prob,p.get("visibility_bonus",0.0),priority,state,json.dumps(p,default=str)))
        con.commit()
    finally: con.close()
    return {"selected":len(selected),"ready_future_api":sum(1 for _,_,p in selected if p.get("token_address")),"awaiting_mint":sum(1 for _,_,p in selected if not p.get("token_address"))}


def write_csv(rows:list[dict[str,Any]], out_path:str, rank_by:str|None=None):
    if rank_by:
        rows.sort(key=lambda r:(r.get(rank_by) is not None,float(r.get(rank_by) or -1e99)),reverse=True)
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def export_handoff(db_path:str, out_dir:str="data"):
    con=connect(db_path)
    try: rows=[dict(r) for r in con.execute("SELECT * FROM axiom_api_handoff_queue ORDER BY handoff_priority DESC")]
    finally: con.close()
    p=Path(out_dir); p.mkdir(parents=True,exist_ok=True)
    for name,subset in (("api_handoff_queue.csv",rows),("api_handoff_candidates.csv",[r for r in rows if r["state"] in ("ready_future_api","awaiting_mint")])):
        if not subset: (p/name).write_text("",encoding="utf-8"); continue
        with (p/name).open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(subset[0])); w.writeheader(); w.writerows(subset)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--db",default="data/live.sqlite"); ap.add_argument("--model",default="models/axiom24/latest.joblib"); ap.add_argument("--out",default="data/axiom_predictions_24h.csv"); ap.add_argument("--rank-by",default="p_hit_plus100_by_12h"); ap.add_argument("--all-latest",action="store_true"); ap.add_argument("--no-handoff",action="store_true")
    args=ap.parse_args(); rows=predict_rows(args.db,args.model,args.all_latest); write_csv(rows,args.out,args.rank_by)
    handoff={} if args.no_handoff else plan_handoff(args.db,rows)
    if not args.no_handoff: export_handoff(args.db)
    print(json.dumps({"predictions":len(rows),"out":args.out,"handoff":handoff},indent=2))
if __name__=="__main__": main()
