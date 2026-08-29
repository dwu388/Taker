from __future__ import annotations

import argparse, json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

SESSION_TABLE="axiom_v24_collection_run_sessions"
CENSOR_TABLE="axiom_v24_collection_censors"
PEAK_SNAPSHOT_TABLE="axiom_v24_collection_censor_peak_snapshots"
DEFAULT_ACTIVE_LOOKBACK_MINUTES=50.0


def _utc(v:Any)->pd.Timestamp:
    t=pd.Timestamp(v); return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
def _now_iso()->str: return datetime.now(timezone.utc).isoformat()
def _exists(c:sqlite3.Connection,n:str)->bool: return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(n,)).fetchone() is not None
def _cols(c:sqlite3.Connection,n:str)->set[str]: return {str(r[1]) for r in c.execute(f'PRAGMA table_info("{n}")')} if _exists(c,n) else set()


def migrate(c:sqlite3.Connection)->None:
    c.executescript(f"""
    CREATE TABLE IF NOT EXISTS {SESSION_TABLE}(session_id TEXT PRIMARY KEY,source TEXT NOT NULL,started_at TEXT NOT NULL,last_capture_at TEXT,stopped_at TEXT,censor_at TEXT,stop_reason TEXT,status TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_{SESSION_TABLE}_status ON {SESSION_TABLE}(status,started_at);
    CREATE TABLE IF NOT EXISTS {CENSOR_TABLE}(session_id TEXT NOT NULL,token_key TEXT NOT NULL,last_seen_at TEXT NOT NULL,censor_at TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(session_id,token_key));
    CREATE INDEX IF NOT EXISTS idx_{CENSOR_TABLE}_token_time ON {CENSOR_TABLE}(token_key,censor_at);
    CREATE TABLE IF NOT EXISTS {PEAK_SNAPSHOT_TABLE}(session_id TEXT NOT NULL,token_key TEXT NOT NULL,decision_at TEXT NOT NULL,row_json TEXT NOT NULL,censor_at TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(session_id,token_key,decision_at));
    """); c.commit()


def _obs(c:sqlite3.Connection)->pd.DataFrame:
    from . import axiom_peak_structure as peak
    x,_=peak.load_observations(c)
    if x.empty:return x
    x=x.copy(); x["token_key"]=x.token_key.astype(str); x["snapshot_at"]=pd.to_datetime(x.snapshot_at,utc=True,errors="coerce")
    return x.dropna(subset=["snapshot_at"])


def _active(c:sqlite3.Connection,at:pd.Timestamp,lookback:float)->dict[str,pd.Timestamp]:
    x=_obs(c)
    if x.empty:return {}
    x=x[(x.snapshot_at<=at)&(x.snapshot_at>=at-pd.Timedelta(minutes=float(lookback)))]
    if x.empty:return {}
    s=x.groupby("token_key",sort=False).snapshot_at.max(); return {str(k):_utc(v) for k,v in s.items()}


def _snapshot_labels(c:sqlite3.Connection,sid:str,active:dict[str,pd.Timestamp],at:pd.Timestamp,reason:str)->int:
    table="axiom_peak_structure_labels_v21"
    if not active or not _exists(c,table):return 0
    cols=_cols(c,table)
    if not {"token_key","decision_at"}.issubset(cols):return 0
    names=[str(r[1]) for r in c.execute(f'PRAGMA table_info("{table}")')]; unfinished="AND COALESCE(label_finalized,0)=0" if "label_finalized" in cols else ""; n=0
    for token in active:
        for row in c.execute(f'SELECT * FROM "{table}" WHERE token_key=? AND decision_at<=? {unfinished}',(token,at.isoformat())):
            rec=dict(zip(names,row)); c.execute(f"INSERT OR IGNORE INTO {PEAK_SNAPSHOT_TABLE}(session_id,token_key,decision_at,row_json,censor_at,reason,created_at) VALUES(?,?,?,?,?,?,?)",(sid,token,str(rec["decision_at"]),json.dumps(rec,sort_keys=True,default=str),at.isoformat(),reason,_now_iso())); n+=int(c.execute("SELECT changes()").fetchone()[0]>0)
    return n


def _censor_paper_state(c:sqlite3.Connection,at:pd.Timestamp,reason:str)->dict[str,int]:
    out={"paper_positions_censored":0,"pending_entries_cancelled":0}
    if _exists(c,"axiom_paper_positions_v20"):
        cols=_cols(c,"axiom_paper_positions_v20"); sets=["status='censored'"]; vals:list[Any]=[]
        if "closed_at" in cols:sets.append("closed_at=?");vals.append(at.isoformat())
        if "close_reason" in cols:sets.append("close_reason=?");vals.append(reason)
        if "exit_kind" in cols:sets.append("exit_kind='collection_censored'")
        for name in ("reward","observed_reward","execution_reward","net_return_pct","observed_net_return_pct","execution_net_return_pct","exit_mc_observed","exit_mc_execution_proxy"):
            if name in cols:sets.append(f'"{name}"=NULL')
        c.execute(f"UPDATE axiom_paper_positions_v20 SET {','.join(sets)} WHERE status='open'",vals); out["paper_positions_censored"]=int(c.execute("SELECT changes()").fetchone()[0])
    if _exists(c,"axiom_paper_pending_entries_v24"):
        cols=_cols(c,"axiom_paper_pending_entries_v24"); sets=["status='cancelled'"]; vals=[]
        if "cancelled_at" in cols:sets.append("cancelled_at=?");vals.append(at.isoformat())
        if "cancel_reason" in cols:sets.append("cancel_reason=?");vals.append(reason)
        c.execute(f"UPDATE axiom_paper_pending_entries_v24 SET {','.join(sets)} WHERE status='pending'",vals); out["pending_entries_cancelled"]=int(c.execute("SELECT changes()").fetchone()[0])
    return out


def _stop(c:sqlite3.Connection,sid:str,reason:str,stopped_at:Any|None=None,lookback:float=DEFAULT_ACTIVE_LOOKBACK_MINUTES)->dict[str,Any]:
    migrate(c); row=c.execute(f"SELECT last_capture_at,status FROM {SESSION_TABLE} WHERE session_id=?",(sid,)).fetchone()
    if not row:return {"stopped":False,"reason":"session_not_found","session_id":sid}
    if str(row[1])!="active":return {"stopped":False,"reason":"already_stopped","session_id":sid}
    stop_at=_utc(stopped_at or pd.Timestamp.now(tz="UTC")); censor_at=_utc(row[0]) if row[0] else stop_at; active=_active(c,censor_at,lookback)
    for token,last_seen in active.items():c.execute(f"INSERT OR REPLACE INTO {CENSOR_TABLE}(session_id,token_key,last_seen_at,censor_at,reason,created_at) VALUES(?,?,?,?,?,?)",(sid,token,last_seen.isoformat(),censor_at.isoformat(),reason,_now_iso()))
    snaps=_snapshot_labels(c,sid,active,censor_at,reason); paper=_censor_paper_state(c,censor_at,reason)
    c.execute(f"UPDATE {SESSION_TABLE} SET stopped_at=?,censor_at=?,stop_reason=?,status='stopped' WHERE session_id=?",(stop_at.isoformat(),censor_at.isoformat(),reason,sid)); c.commit()
    return {"stopped":True,"session_id":sid,"stopped_at":stop_at.isoformat(),"censor_at":censor_at.isoformat(),"reason":reason,"active_tokens_censored":len(active),"peak_rows_snapshotted":snaps,**paper}


def start_collection_session(db:str,*,source:str="axiom_migrated_runner",started_at:Any|None=None)->str:
    Path(db).parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db) as c:
        migrate(c)
        for (sid,) in c.execute(f"SELECT session_id FROM {SESSION_TABLE} WHERE status='active' ORDER BY started_at").fetchall():_stop(c,str(sid),"unclean_restart_censored")
        sid=str(uuid.uuid4()); t=_utc(started_at or pd.Timestamp.now(tz="UTC")); c.execute(f"INSERT INTO {SESSION_TABLE}(session_id,source,started_at,status,created_at) VALUES(?,?,?,'active',?)",(sid,source,t.isoformat(),_now_iso())); c.commit(); return sid


def note_successful_capture(db:str,sid:str,capture_at:Any|None=None)->None:
    with sqlite3.connect(db) as c:
        migrate(c)
        if capture_at is None:
            r=c.execute("SELECT MAX(captured_at) FROM capture_cycles").fetchone(); capture_at=r[0] if r and r[0] else None
        if capture_at is not None:c.execute(f"UPDATE {SESSION_TABLE} SET last_capture_at=? WHERE session_id=? AND status='active'",(_utc(capture_at).isoformat(),sid));c.commit()


def stop_collection_session(db:str,sid:str|None=None,*,reason:str="manual_stop_censored",stopped_at:Any|None=None)->dict[str,Any]:
    with sqlite3.connect(db) as c:
        migrate(c)
        if sid is None:
            r=c.execute(f"SELECT session_id FROM {SESSION_TABLE} WHERE status='active' ORDER BY started_at DESC LIMIT 1").fetchone()
            if not r:return {"stopped":False,"reason":"no_active_session"}
            sid=str(r[0])
        return _stop(c,sid,reason,stopped_at)


def first_collection_boundary_between(c:sqlite3.Connection,start:Any,end:Any)->pd.Timestamp|None:
    if not _exists(c,SESSION_TABLE):return None
    a,b=_utc(start),_utc(end); r=c.execute(f"SELECT censor_at FROM {SESSION_TABLE} WHERE status='stopped' AND censor_at IS NOT NULL AND censor_at>=? AND censor_at<=? ORDER BY censor_at LIMIT 1",(a.isoformat(),b.isoformat())).fetchone(); return _utc(r[0]) if r else None


def censors_by_token(c:sqlite3.Connection)->dict[str,list[dict[str,Any]]]:
    if not _exists(c,CENSOR_TABLE):return {}
    out:dict[str,list[dict[str,Any]]]={}
    for token,at,reason,last,sid in c.execute(f"SELECT token_key,censor_at,reason,last_seen_at,session_id FROM {CENSOR_TABLE} ORDER BY token_key,censor_at"):out.setdefault(str(token),[]).append({"censor_at":_utc(at),"reason":str(reason),"last_seen_at":_utc(last),"session_id":str(sid)})
    return out


def censor_from_map(m:dict[str,list[dict[str,Any]]],token:str,start:Any,end:Any)->dict[str,Any]|None:
    a,b=_utc(start),_utc(end)
    for rec in m.get(str(token),[]):
        if a<=_utc(rec["censor_at"])<=b:return rec
    return None


def apply_peak_label_censors(c:sqlite3.Connection)->dict[str,int]:
    migrate(c); table="axiom_peak_structure_labels_v21"
    if not _exists(c,table):return {"restored":0,"neutralized":0}
    cols=_cols(c,table); restored=neutralized=0; snap_keys:set[tuple[str,str]]=set()
    for raw,at,reason in c.execute(f"SELECT row_json,censor_at,reason FROM {PEAK_SNAPSHOT_TABLE} ORDER BY censor_at"):
        try:o=json.loads(raw)
        except Exception:continue
        token,decision=str(o.get("token_key") or ""),str(o.get("decision_at") or "")
        if not token or not decision:continue
        snap_keys.add((token,decision)); u={k:v for k,v in o.items() if k in cols and k not in {"token_key","decision_at","config_json","schema_version","target_fingerprint"}}
        if "terminal_at" in cols:u["terminal_at"]=at
        if "terminal_reason" in cols:u["terminal_reason"]=reason
        if "path_end_at" in cols:u["path_end_at"]=at
        if "label_finalized" in cols:u["label_finalized"]=0
        if "label_status_next_peak" in cols and not o.get("has_next_substantial_peak_before_terminal_72h"):u["label_status_next_peak"]="censored_collection_stop"
        if u:c.execute(f'UPDATE "{table}" SET '+",".join(f'"{k}"=?' for k in u)+" WHERE token_key=? AND decision_at=?",[*u.values(),token,decision]);restored+=int(c.execute("SELECT changes()").fetchone()[0]>0)
    targets=[x for x in cols if x.startswith(("has_next_","next_substantial_peak_","later_higher_peak_","post_next_peak_","time_to_"))]
    for token,recs in censors_by_token(c).items():
        for rec in recs:
            at=_utc(rec["censor_at"]); reason=str(rec["reason"])
            for (decision,) in c.execute(f'SELECT decision_at FROM "{table}" WHERE token_key=? AND decision_at<=?',(token,at.isoformat())).fetchall():
                if (token,str(decision)) in snap_keys:continue
                d=_utc(decision)
                if d+pd.Timedelta(hours=24)<=at:continue
                sets=[f'"{x}"=NULL' for x in targets]; params:list[Any]=[]
                if "label_finalized" in cols:sets.append('"label_finalized"=0')
                if "label_status_next_peak" in cols:sets.append("\"label_status_next_peak\"='censored_collection_stop'")
                if "terminal_at" in cols:sets.append('"terminal_at"=?');params.append(at.isoformat())
                if "terminal_reason" in cols:sets.append('"terminal_reason"=?');params.append(reason)
                if "path_end_at" in cols:sets.append('"path_end_at"=?');params.append(at.isoformat())
                if sets:c.execute(f'UPDATE "{table}" SET '+",".join(sets)+" WHERE token_key=? AND decision_at=?",[*params,token,decision]);neutralized+=int(c.execute("SELECT changes()").fetchone()[0]>0)
    c.commit(); return {"restored":restored,"neutralized":neutralized}


def status(db:str)->dict[str,Any]:
    with sqlite3.connect(db) as c:
        migrate(c); a=c.execute(f"SELECT COUNT(*) FROM {SESSION_TABLE} WHERE status='active'").fetchone()[0];s=c.execute(f"SELECT COUNT(*) FROM {SESSION_TABLE} WHERE status='stopped'").fetchone()[0];n=c.execute(f"SELECT COUNT(*) FROM {CENSOR_TABLE}").fetchone()[0];r=c.execute(f"SELECT session_id,source,started_at,last_capture_at,stopped_at,censor_at,stop_reason,status FROM {SESSION_TABLE} ORDER BY started_at DESC LIMIT 1").fetchone();keys=["session_id","source","started_at","last_capture_at","stopped_at","censor_at","stop_reason","status"];return {"active_sessions":int(a),"stopped_sessions":int(s),"neutral_token_censors":int(n),"latest_session":dict(zip(keys,r)) if r else None}


def main()->None:
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    for name in ("stop","status"):p=sub.add_parser(name);p.add_argument("--db",default="data/live.sqlite")
    x=ap.parse_args();print(json.dumps(stop_collection_session(x.db) if x.cmd=="stop" else status(x.db),indent=2,default=str))
if __name__=="__main__":main()
