from __future__ import annotations

import argparse
import json
import math
from datetime import timedelta
from typing import Any

import numpy as np

from .common import json_dumps, parse_iso
from .db import connect, migrate

HORIZONS = {"1h":60,"4h":240,"12h":720,"24h":1440}
UP_THRESHOLDS = {"plus30":.30,"plus50":.50,"plus100":1.0,"plus200":2.0,"plus400":4.0}
DOWN_THRESHOLDS = {"minus30":-.30,"minus50":-.50}
AGEOUT_DEATH_MAX_MINUTES = 23*60
DEATH_MISSED_CYCLES = 10


def _capture_cycles(con):
    return [dict(r) for r in con.execute("SELECT cycle_id,captured_at,completed FROM capture_cycles WHERE completed=1 ORDER BY cycle_id")]


def compute_death_events(con) -> dict[str,list[dict[str,Any]]]:
    cycles=_capture_cycles(con)
    obs=[dict(r) for r in con.execute("SELECT cycle_id,token_key,snapshot_at,age_minutes FROM axiom_observations ORDER BY cycle_id")]
    by_cycle:dict[int,dict[str,dict[str,Any]]]={}
    for r in obs: by_cycle.setdefault(int(r["cycle_id"]),{})[r["token_key"]]=r
    tokens=sorted({r["token_key"] for r in obs}); out={t:[] for t in tokens}
    for token in tokens:
        seen=False; misses=0; last_seen=None; already_dead=False
        for c in cycles:
            cid=int(c["cycle_id"]); row=by_cycle.get(cid,{}).get(token)
            if row:
                # A later reappearance starts a new observable lifecycle after a prior death.
                if already_dead: already_dead=False
                seen=True; misses=0; last_seen=row
                continue
            if not seen or already_dead: continue
            misses+=1
            if misses>=DEATH_MISSED_CYCLES:
                age=last_seen.get("age_minutes") if last_seen else None
                if age is not None and int(age)>=AGEOUT_DEATH_MAX_MINUTES:
                    out[token].append({"time":c["captured_at"],"cycle_id":cid,"reason":"age_out_ambiguity","last_seen":last_seen["snapshot_at"]})
                else:
                    out[token].append({"time":c["captured_at"],"cycle_id":cid,"reason":"dead_after_10_missed_cycles","last_seen":last_seen["snapshot_at"]})
                already_dead=True; seen=False; misses=0
    return out


def _first_cross(path:list[tuple[Any,float]], threshold:float, up:bool=True):
    for dt,ret in path:
        if (up and ret>=threshold) or ((not up) and ret<=threshold): return dt
    return None


def _minutes(a,b)->float:
    return (b-a).total_seconds()/60.0


def _known_binary(event_time, decision_dt, horizon_min:int, latest_dt, terminal_dt=None, ambiguous_terminal=False):
    end=decision_dt+timedelta(minutes=horizon_min)
    if event_time is not None and event_time<=end:
        return 1
    # A real death is terminal evidence; age-out ambiguity is not.
    if terminal_dt is not None and terminal_dt<=end and not ambiguous_terminal:
        return 0
    if latest_dt>=end:
        return 0
    return None


def targets_for_decision(decision:dict[str,Any], future:list[dict[str,Any]], latest_dt, death_events:list[dict[str,Any]])->tuple[str,str|None,str|None,dict[str,Any]]:
    dt=parse_iso(decision["snapshot_at"]); entry=decision.get("market_cap_usd")
    if dt is None or entry is None or float(entry)<=0:
        return "incomplete","missing_entry_market_cap",None,{}
    entry=float(entry); end24=dt+timedelta(hours=24)
    death=None
    for ev in death_events:
        edt=parse_iso(ev["time"])
        if edt and dt<edt<=end24:
            death={**ev,"_dt":edt}; break
    ambiguous=bool(death and death["reason"]=="age_out_ambiguity")
    terminal_dt = None if death is None else death["_dt"]
    effective_end = min(end24, terminal_dt) if terminal_dt and not ambiguous else end24
    obs=[]
    for r in future:
        rdt=parse_iso(r["snapshot_at"]); mc=r.get("market_cap_usd")
        if rdt and dt<=rdt<=effective_end and mc is not None and float(mc)>0:
            obs.append((rdt,float(mc)))
    if not obs or obs[0][0]!=dt:
        obs.insert(0,(dt,entry))
    obs.sort(key=lambda x:x[0])
    path=[(t,(mc/entry)-1.0) for t,mc in obs]
    t:dict[str,Any]={}
    up_events={name:_first_cross(path,thr,True) for name,thr in UP_THRESHOLDS.items()}
    down_events={name:_first_cross(path,thr,False) for name,thr in DOWN_THRESHOLDS.items()}
    for hk,hmin in HORIZONS.items():
        for name in UP_THRESHOLDS:
            t[f"hit_{name}_by_{hk}"]=_known_binary(up_events[name],dt,hmin,latest_dt,terminal_dt,ambiguous)
        for name in DOWN_THRESHOLDS:
            t[f"hit_{name}_by_{hk}"]=_known_binary(down_events[name],dt,hmin,latest_dt,terminal_dt,ambiguous)
    for hk,hmin in (("1h",60),("4h",240),("12h",720)):
        if death and not ambiguous and terminal_dt<=dt+timedelta(minutes=hmin): t[f"dead_within_{hk}_24h"]=1
        elif latest_dt>=dt+timedelta(minutes=hmin) or (death and not ambiguous): t[f"dead_within_{hk}_24h"]=0
        else: t[f"dead_within_{hk}_24h"]=None

    up100=up_events["plus100"]; up50=up_events["plus50"]; dn50=down_events["minus50"]; dn30=down_events["minus30"]
    real_death_dt=terminal_dt if death and not ambiguous else None
    def competing(success_dt, risk_dt):
        events=[(x,k) for x,k in ((success_dt,"success"),(risk_dt,"risk"),(real_death_dt,"death")) if x is not None]
        if events:
            first=min(events,key=lambda x:x[0])
            return 1 if first[1]=="success" else 0
        if latest_dt>=end24: return 0
        return None
    t["success_2x_before_50dd_or_dead_24h"]=competing(up100,dn50)
    t["success_50_before_30dd_or_dead_24h"]=competing(up50,dn30)
    if real_death_dt is not None:
        t["dead_before_2x_24h"]=1 if up100 is None or real_death_dt<up100 else 0
    elif up100 is not None: t["dead_before_2x_24h"]=0
    elif latest_dt>=end24: t["dead_before_2x_24h"]=0
    else: t["dead_before_2x_24h"]=None

    full_terminal = bool(real_death_dt is not None or latest_dt>=end24)
    if full_terminal:
        prices=[mc for _,mc in obs]
        peak_idx=int(np.argmax(prices)); peak=prices[peak_idx]
        peak_dt=obs[peak_idx][0]
        peak_mult=peak/entry
        pre=prices[:peak_idx+1]
        min_pre=min(pre)
        dd_entry=max(0.0,1.0-min_pre/entry)
        running=pre[0]; max_dd=0.0
        for p in pre:
            running=max(running,p)
            if running>0: max_dd=max(max_dd,1.0-p/running)
        t["peak_multiple_24h"]=peak_mult
        t["log_peak_multiple_24h"]=math.log(max(peak_mult,1e-8))
        mins=_minutes(dt,peak_dt)
        t["time_to_peak_before_terminal_minutes_24h"]=mins
        t["log1p_time_to_peak_minutes_24h"]=math.log1p(max(0.0,mins))
        t["drawdown_before_peak_pct_24h"]=dd_entry
        t["peak_to_trough_before_peak_pct_24h"]=max_dd
        t["max_return_before_terminal_24h"]=max(r for _,r in path)
        t["min_return_before_terminal_24h"]=min(r for _,r in path)
        t["return_at_terminal_observation_24h"]=path[-1][1]
        t["observable_lifetime_minutes_24h"]=_minutes(dt,obs[-1][0])
        if real_death_dt is not None:
            md=_minutes(dt,real_death_dt)
            t["time_to_death_minutes_if_death_24h"]=md
            t["log1p_time_to_death_minutes_if_death_24h"]=math.log1p(max(0.0,md))
        else:
            t["time_to_death_minutes_if_death_24h"]=None; t["log1p_time_to_death_minutes_if_death_24h"]=None
    else:
        for key in ("peak_multiple_24h","log_peak_multiple_24h","time_to_peak_before_terminal_minutes_24h","log1p_time_to_peak_minutes_24h","drawdown_before_peak_pct_24h","peak_to_trough_before_peak_pct_24h","max_return_before_terminal_24h","min_return_before_terminal_24h","return_at_terminal_observation_24h","observable_lifetime_minutes_24h","time_to_death_minutes_if_death_24h","log1p_time_to_death_minutes_if_death_24h"):
            t[key]=None
    known=sum(v is not None for v in t.values())
    if full_terminal: status="complete"
    elif known: status="partial"
    else: status="immature"
    reason = death["reason"] if death else ("full_24h_observed" if latest_dt>=end24 else "awaiting_future")
    return status,reason,(terminal_dt.isoformat() if terminal_dt else None),t


def refresh(db_path:str)->dict[str,int]:
    migrate(db_path); con=connect(db_path)
    try:
        observations=[dict(r) for r in con.execute("SELECT * FROM axiom_observations ORDER BY token_key,snapshot_at")]
        cycles=_capture_cycles(con)
        latest_dt=max((parse_iso(c["captured_at"]) for c in cycles if parse_iso(c["captured_at"])),default=None)
        if latest_dt is None:
            return {"complete":0,"partial":0,"immature":0,"incomplete":0}
        deaths=compute_death_events(con)
        by_token={}
        for r in observations: by_token.setdefault(r["token_key"],[]).append(r)
        con.execute("DELETE FROM axiom_labels_24h_v18")
        counts={"complete":0,"partial":0,"immature":0,"incomplete":0}
        for token,hist in by_token.items():
            for i,r in enumerate(hist):
                status,reason,terminal,t=targets_for_decision(r,hist[i:],latest_dt,deaths.get(token,[]))
                if status not in counts: counts[status]=0
                counts[status]+=1
                con.execute("INSERT OR REPLACE INTO axiom_labels_24h_v18(observation_id,token_key,decision_time,label_status,terminal_reason,terminal_time,target_json) VALUES(?,?,?,?,?,?,?)",(r["observation_id"],token,r["snapshot_at"],status,reason,terminal,json_dumps(t)))
        con.commit(); return counts
    finally: con.close()


def status(db_path:str)->dict[str,Any]:
    migrate(db_path); con=connect(db_path)
    try:
        labels=[dict(r) for r in con.execute("SELECT * FROM axiom_labels_24h_v18")]
        out={"total_rows":len(labels),"complete":0,"partial":0,"immature":0,"incomplete":0}
        target_counts={}; terminal={}; mixed={}
        per_token={}
        for r in labels:
            out[r["label_status"]]=out.get(r["label_status"],0)+1
            terminal[r.get("terminal_reason")]=terminal.get(r.get("terminal_reason"),0)+1
            t=json.loads(r["target_json"])
            for k,v in t.items():
                if v is not None:
                    rec=target_counts.setdefault(k,{"known":0,"positive":0})
                    rec["known"]+=1
                    if v==1: rec["positive"]+=1
            p=t.get("success_2x_before_50dd_or_dead_24h")
            if p is not None: per_token.setdefault(r["token_key"],set()).add(int(p))
        out["tokens_with_mixed_success_and_failure_decision_points"]=sum(1 for s in per_token.values() if len(s)>1)
        out["terminal_event_breakdown"]={str(k):v for k,v in terminal.items()}
        out["max_inference_target_availability"]=target_counts
        return out
    finally: con.close()


def main():
    ap=argparse.ArgumentParser(description="V18 24-hour lifecycle labels and training")
    sub=ap.add_subparsers(dest="cmd",required=True)
    for c in ("refresh","status"):
        p=sub.add_parser(c); p.add_argument("--db",default="data/live.sqlite")
    p=sub.add_parser("train"); p.add_argument("--db",default="data/live.sqlite"); p.add_argument("--output-dir",default="models/axiom24"); p.add_argument("--allow-small",action="store_true")
    args=ap.parse_args()
    if args.cmd=="refresh": print(json.dumps(refresh(args.db),indent=2))
    elif args.cmd=="status": print(json.dumps(status(args.db),indent=2))
    else:
        from .axiom_train_24h import train_models
        print(json.dumps(train_models(args.db,args.output_dir,args.allow_small),indent=2,default=str))
if __name__=="__main__": main()
