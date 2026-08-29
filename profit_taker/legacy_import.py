from __future__ import annotations

import argparse,json,re
from collections import defaultdict
from typing import Any

from .common import normalize_token_key,parse_compact_number,parse_duration_minutes,safe_float,safe_int,json_dumps
from .db import connect,migrate
from .visibility import update_visibility

ALIASES={
 'snapshot_at':['snapshot_at','captured_at','observed_at','timestamp','decision_time','created_at'],
 'token_key':['token_key','token_id','mint','token_address','address'],
 'token_address':['token_address','mint','mint_address','contract_address'],
 'name':['name','token_name','symbol','ticker'],
 'short_address_hint':['short_address_hint','short_address','address_hint'],
 'age_minutes':['age_minutes','age_min','token_age_minutes','age'],
 'image_reuse_count':['image_reuse_count','image_reuse'],
 'market_cap_usd':['market_cap_usd','market_cap','mcap','mc'],
 'volume_usd':['volume_usd','volume','vol'],
 'fees_sol':['fees_sol','fees','fee'],
 'txns':['txns','transactions','transaction_count','tx'],
 'holders':['holders','holder_count'],
 'pro_traders':['pro_traders','pro_trader_count'],
 'kols':['kols','kol_count'],
 'dev_migrations':['dev_migrations'], 'dev_creations':['dev_creations'], 'recent_visitors':['recent_visitors','visitors'],
 'top10_holders_pct':['top10_holders_pct','top10_pct','top_10_pct'], 'tracked_dev_status_raw':['tracked_dev_status_raw','dev_status'],
 'funding_time_raw':['funding_time_raw','funding_time'], 'funding_time_minutes':['funding_time_minutes'],
 'sniper_pct':['sniper_pct'], 'insider_pct':['insider_pct'], 'bundler_pct':['bundler_pct'], 'dex_paid':['dex_paid']
}
NEW_TABLES={'capture_cycles','axiom_observations','axiom_features_v18','axiom_visibility','axiom_labels_24h_v18','axiom_api_handoff_queue','wallet_history_snapshots_v12','helius_token_observations','normalized_trades','provider_usage','wallet_backfill_queue','shyft_wallet_prescreens','token_sync_state','wallet_sync_state','helius_wallet_sync_state','helius_token_sync_state'}

def _tables(con):return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
def _cols(con,t):return [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
def _pick(cols,names):
    low={c.lower():c for c in cols}
    for n in names:
        if n.lower() in low:return low[n.lower()]
    return None

def discover(con):
    cands=[]
    for t in _tables(con):
        if t in NEW_TABLES:continue
        cols=_cols(con,t);snap=_pick(cols,ALIASES['snapshot_at']);mc=_pick(cols,ALIASES['market_cap_usd']);tk=_pick(cols,ALIASES['token_key']);name=_pick(cols,ALIASES['name']);hint=_pick(cols,ALIASES['short_address_hint'])
        score=sum(x is not None for x in (snap,mc,tk,name,hint))
        if snap and score>=3:cands.append({'table':t,'columns':cols,'score':score})
    return sorted(cands,key=lambda x:x['score'],reverse=True)

def _coerce(field,v):
    if v is None:return None
    if field in ('market_cap_usd','volume_usd'):
        return parse_compact_number(v) if isinstance(v,str) else safe_float(v)
    if field in ('fees_sol','top10_holders_pct','sniper_pct','insider_pct','bundler_pct'):return safe_float(str(v).replace('%',''))
    if field in ('txns','holders','pro_traders','kols','dev_migrations','dev_creations','recent_visitors','image_reuse_count','funding_time_minutes'):return safe_int(v)
    if field=='age_minutes':
        if isinstance(v,str) and re.search(r'[smhdwoy]',v,re.I):return parse_duration_minutes(v.strip().lower())
        return safe_int(v)
    if field=='dex_paid':
        if isinstance(v,str):return 1 if v.strip().lower() in ('1','true','yes','paid') else 0
        return 1 if bool(v) else 0
    return str(v) if v is not None else None

def import_legacy(db,auto=False,table=None):
    migrate(db);con=connect(db)
    try:
        existing=int(con.execute('SELECT COUNT(*) FROM axiom_observations').fetchone()[0])
        cands=discover(con)
        if auto and existing>0:return {'status':'skipped','reason':'axiom_observations already populated','existing_rows':existing,'candidates':cands}
        if not table:
            if not cands:return {'status':'no_candidate','existing_rows':existing,'candidates':[]}
            table=cands[0]['table']
        cols=_cols(con,table);mapping={f:_pick(cols,a) for f,a in ALIASES.items()};snapcol=mapping['snapshot_at']
        if not snapcol:raise RuntimeError(f'{table} has no recognizable timestamp column')
        raw=[dict(r) for r in con.execute(f'SELECT * FROM "{table}" ORDER BY "{snapcol}"')]
        cycles={};imported=0;seen=set()
        for rr in raw:
            snap=str(rr.get(snapcol) or '').strip()
            if not snap:continue
            if snap not in cycles:
                cycles[snap]=con.execute('INSERT INTO capture_cycles(captured_at,rows_detected,completed) VALUES(?,0,1)',(snap,)).lastrowid
            row={f:_coerce(f,rr.get(c)) if c else None for f,c in mapping.items()}
            token_key=row.get('token_key') or normalize_token_key(row.get('name'),row.get('short_address_hint'),row.get('token_address'))
            if not token_key:continue
            # Ignore old synthetic disappearance placeholders: there must be at least one real observed market/card value.
            if not any(row.get(f) is not None for f in ('market_cap_usd','volume_usd','txns','holders','fees_sol')):continue
            sig=(token_key,snap)
            if sig in seen:continue
            seen.add(sig)
            fields=['cycle_id','token_key','token_address','name','short_address_hint','snapshot_at','age_minutes','image_reuse_count','market_cap_usd','volume_usd','fees_sol','txns','holders','pro_traders','kols','dev_migrations','dev_creations','recent_visitors','top10_holders_pct','tracked_dev_status_raw','funding_time_raw','funding_time_minutes','sniper_pct','insider_pct','bundler_pct','dex_paid','source_json']
            vals=[cycles[snap],token_key,row.get('token_address'),row.get('name'),row.get('short_address_hint'),snap,row.get('age_minutes'),row.get('image_reuse_count'),row.get('market_cap_usd'),row.get('volume_usd'),row.get('fees_sol'),row.get('txns'),row.get('holders'),row.get('pro_traders'),row.get('kols'),row.get('dev_migrations'),row.get('dev_creations'),row.get('recent_visitors'),row.get('top10_holders_pct'),row.get('tracked_dev_status_raw'),row.get('funding_time_raw'),row.get('funding_time_minutes'),row.get('sniper_pct'),row.get('insider_pct'),row.get('bundler_pct'),row.get('dex_paid'),json_dumps({'legacy_source_table':table})]
            con.execute(f"INSERT OR IGNORE INTO axiom_observations({','.join(fields)}) VALUES({','.join('?' for _ in fields)})",vals);imported+=1
        for snap,cid in cycles.items():con.execute('UPDATE capture_cycles SET rows_detected=(SELECT COUNT(*) FROM axiom_observations WHERE cycle_id=?) WHERE cycle_id=?',(cid,cid))
        con.commit()
    finally:con.close()
    # Reconstruct positive-only visibility chronologically from imported completed cycles.
    con=connect(db)
    try: pairs=[tuple(r) for r in con.execute('SELECT cycle_id,token_key,snapshot_at FROM axiom_observations ORDER BY cycle_id,observation_id')]
    finally:con.close()
    for cid,tk,snap in pairs:update_visibility(db,int(cid),tk,snap)
    return {'status':'imported','source_table':table,'rows_imported':imported,'cycles_imported':len(cycles),'candidates':cands}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--auto',action='store_true');ap.add_argument('--table');a=ap.parse_args();print(json.dumps(import_legacy(a.db,a.auto,a.table),indent=2,default=str))
if __name__=='__main__':main()
