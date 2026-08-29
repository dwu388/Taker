from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from typing import Any

import numpy as np

from .common import json_dumps, parse_iso, safe_float
from .db import connect, migrate

RAW_NUMERIC = [
    "age_minutes","image_reuse_count","market_cap_usd","volume_usd","fees_sol","txns","holders",
    "pro_traders","kols","dev_migrations","dev_creations","recent_visitors","top10_holders_pct",
    "funding_time_minutes","sniper_pct","insider_pct","bundler_pct","dex_paid"
]
TRAJECTORY_FIELDS = [
    "market_cap_usd","volume_usd","fees_sol","txns","holders","pro_traders","kols","recent_visitors",
    "top10_holders_pct","sniper_pct","insider_pct","bundler_pct"
]
WINDOWS = [5,10,15,30,60,90,180]

EXCLUDED_MODEL_KEYS = {
    "capture_count","visibility_bonus","consecutive_capture_count","max_consecutive_capture_count",
    "reappearance_count","observation_index","interval_minutes","obs_count_60m","obs_count_90m","obs_count_180m"
}


def _ratio(a: Any,b: Any) -> float|None:
    a=safe_float(a); b=safe_float(b)
    if a is None or b is None or b==0: return None
    return a/b


def _pct(a: Any,b: Any) -> float|None:
    a=safe_float(a); b=safe_float(b)
    if a is None or b is None or b==0: return None
    return (a/b)-1.0


def _nearest_before(history:list[dict[str,Any]], current_dt, minutes:int) -> dict[str,Any]|None:
    target=current_dt-timedelta(minutes=minutes)
    candidates=[r for r in history if r["_dt"]<=target]
    if not candidates: return None
    # nearest causal row at or before target, with tolerance that grows modestly with window
    r=max(candidates,key=lambda x:x["_dt"])
    tolerance=max(4, min(20, minutes*.35))
    if (target-r["_dt"]).total_seconds()/60 > tolerance: return None
    return r


def _window_rows(history:list[dict[str,Any]], current_dt, minutes:int)->list[dict[str,Any]]:
    start=current_dt-timedelta(minutes=minutes)
    return [r for r in history if start<=r["_dt"]<=current_dt]


def _path_stats(rows:list[dict[str,Any]], field:str)->dict[str,float|None]:
    vals=[]
    for r in rows:
        v=safe_float(r.get(field))
        if v is not None and v>0: vals.append(v)
    if len(vals)<3:
        return {"efficiency":None,"positive_ratio":None,"volatility":None,"largest_gain":None,"largest_loss":None}
    rets=np.diff(np.log(np.asarray(vals,dtype=float)))
    total=float(np.sum(np.abs(rets)))
    net=float(np.log(vals[-1]/vals[0]))
    return {
        "efficiency": net/total if total>0 else 0.0,
        "positive_ratio": float(np.mean(rets>0)),
        "volatility": float(np.std(rets)),
        "largest_gain": float(np.max(rets)),
        "largest_loss": float(np.min(rets)),
    }


def compute_feature_row(current:dict[str,Any], prior_and_current:list[dict[str,Any]])->dict[str,Any]:
    current_dt=parse_iso(current["snapshot_at"])
    if current_dt is None: raise ValueError("invalid snapshot_at")
    history=[]
    for r in prior_and_current:
        dt=parse_iso(r.get("snapshot_at"))
        if dt is not None and dt<=current_dt:
            rr=dict(r); rr["_dt"]=dt; history.append(rr)
    history.sort(key=lambda r:r["_dt"])
    feat:dict[str,Any]={}
    for f in RAW_NUMERIC:
        v=current.get(f)
        feat[f]=safe_float(v)
        feat[f+"_missing"]=1.0 if v is None else 0.0
    feat["tracked_dev_status_ds"] = 1.0 if str(current.get("tracked_dev_status_raw") or "").upper()=="DS" else 0.0
    feat["volume_mcap"]=_ratio(current.get("volume_usd"),current.get("market_cap_usd"))
    feat["fees_per_tx"]=_ratio(current.get("fees_sol"),current.get("txns"))
    feat["volume_per_tx"]=_ratio(current.get("volume_usd"),current.get("txns"))
    feat["pro_trader_holder_ratio"]=_ratio(current.get("pro_traders"),current.get("holders"))
    feat["kol_holder_ratio"]=_ratio(current.get("kols"),current.get("holders"))
    feat["recent_visitor_holder_ratio"]=_ratio(current.get("recent_visitors"),current.get("holders"))
    feat["dev_migrations_per_creation"]=_ratio(current.get("dev_migrations"),current.get("dev_creations"))

    for w in WINDOWS:
        prior=_nearest_before(history,current_dt,w)
        for f in TRAJECTORY_FIELDS:
            ch=_pct(current.get(f),prior.get(f)) if prior else None
            feat[f+f"_change_{w}m"]=ch
            feat[f+f"_slope_{w}m"]=ch/w if ch is not None else None
        if w in (15,30,60,90,180):
            ps=_path_stats(_window_rows(history,current_dt,w),"market_cap_usd")
            for k,v in ps.items(): feat[f"mc_path_{k}_{w}m"]=v

    # Curvature / acceleration using only causal slopes.
    def sub(a,b):
        x=safe_float(feat.get(a)); y=safe_float(feat.get(b)); return None if x is None or y is None else x-y
    feat["mc_curvature_5v30"]=sub("market_cap_usd_slope_5m","market_cap_usd_slope_30m")
    feat["mc_curvature_15v60"]=sub("market_cap_usd_slope_15m","market_cap_usd_slope_60m")
    feat["mc_curvature_30v180"]=sub("market_cap_usd_slope_30m","market_cap_usd_slope_180m")
    feat["volume_accel_15v60"]=sub("volume_usd_slope_15m","volume_usd_slope_60m")
    feat["tx_accel_15v60"]=sub("txns_slope_15m","txns_slope_60m")
    feat["holders_accel_15v60"]=sub("holders_slope_15m","holders_slope_60m")

    for w in (15,30,60,180):
        pairs=[("volume_usd","market_cap_usd","volume_vs_mc"),("txns","market_cap_usd","tx_vs_mc"),("holders","market_cap_usd","holders_vs_mc"),("pro_traders","holders","pro_vs_holders"),("fees_sol","txns","fees_vs_tx"),("recent_visitors","holders","visitors_vs_holders")]
        for a,b,name in pairs:
            x=safe_float(feat.get(f"{a}_slope_{w}m")); y=safe_float(feat.get(f"{b}_slope_{w}m"))
            feat[f"{name}_momentum_{w}m"] = None if x is None or y is None else x-y

    first=history[0] if history else current
    for f,short in (("market_cap_usd","mc"),("volume_usd","volume"),("txns","tx"),("holders","holders")):
        feat[f"{short}_multiple_from_first"]=_ratio(current.get(f),first.get(f))
    mc_vals=[safe_float(r.get("market_cap_usd")) for r in history]
    mc_vals=[v for v in mc_vals if v is not None and v>0]
    if mc_vals and safe_float(current.get("market_cap_usd")) is not None:
        high=max(mc_vals); cur=float(current["market_cap_usd"])
        feat["mc_fraction_of_observed_high"]=cur/high if high else None
        feat["mc_drawdown_from_observed_high"]=(cur/high)-1 if high else None
    else:
        feat["mc_fraction_of_observed_high"]=None; feat["mc_drawdown_from_observed_high"]=None
    return {k:v for k,v in feat.items() if k not in EXCLUDED_MODEL_KEYS}


def rebuild_features(db_path:str)->dict[str,int]:
    migrate(db_path); con=connect(db_path)
    try:
        rows=[dict(r) for r in con.execute("SELECT * FROM axiom_observations ORDER BY token_key,snapshot_at")]
        by_token:dict[str,list[dict[str,Any]]]={}
        for r in rows: by_token.setdefault(r["token_key"],[]).append(r)
        count=0
        con.execute("DELETE FROM axiom_features_v18")
        for token, hist in by_token.items():
            prefix=[]
            for r in hist:
                prefix.append(r)
                feat=compute_feature_row(r,prefix)
                con.execute("INSERT OR REPLACE INTO axiom_features_v18(observation_id,token_key,snapshot_at,feature_json) VALUES(?,?,?,?)",(r["observation_id"],token,r["snapshot_at"],json_dumps(feat)))
                count+=1
        con.commit(); return {"features_rebuilt":count,"tokens":len(by_token)}
    finally: con.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--db",default="data/live.sqlite"); args=ap.parse_args(); print(json.dumps(rebuild_features(args.db),indent=2))
if __name__=="__main__": main()
