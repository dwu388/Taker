from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from .config import load_env,key
from .providers.shyft import ShyftClient
from .db import connect,migrate
from .backfill_queue import enqueue

def _ts(row): return row.get('timestamp') or row.get('blockTime') or row.get('block_time') or row.get('time')
def screen_wallet(db,client,wallet,max_pages=3,promote_score=55):
    rows,used=client.paged_history(wallet,max_pages=max_pages);swaps=[r for r in rows if str(r.get('type') or r.get('transaction_type') or '').upper()=='SWAP']
    ratio=len(swaps)/len(rows) if rows else 0.0
    # Cost-control priority only, never a trading feature.
    score=min(100.0, len(swaps)*1.2 + ratio*45 + min(len(rows),100)/10)
    times=[_ts(r) for r in rows if _ts(r) is not None]
    rec={'wallet':wallet,'transactions_seen':len(rows),'swap_transactions':len(swaps),'swap_ratio':ratio,'oldest_at':str(times[-1]) if times else None,'newest_at':str(times[0]) if times else None,'requests_used':used,'prescreen_score':score,'promoted_to_birdeye_queue':score>=promote_score}
    migrate(db);con=connect(db)
    try:
        con.execute('INSERT INTO shyft_wallet_prescreens(wallet,screened_at,transactions_seen,swap_transactions,swap_ratio,oldest_at,newest_at,requests_used,prescreen_score,promoted,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(wallet,datetime.now(timezone.utc).isoformat(),len(rows),len(swaps),ratio,rec['oldest_at'],rec['newest_at'],used,score,1 if score>=promote_score else 0,json.dumps(rows,default=str)));con.commit()
    finally:con.close()
    if score>=promote_score:enqueue(db,wallet,score,'shyft_prescreen')
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('wallets',nargs='+');ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--env-file',default='.env');ap.add_argument('--max-pages',type=int,default=3);ap.add_argument('--promote-score',type=float,default=55);a=ap.parse_args()
    c=ShyftClient(key(load_env(a.env_file),'SHYFT_API_KEY'));print(json.dumps([screen_wallet(a.db,c,w,a.max_pages,a.promote_score) for w in a.wallets],indent=2))
if __name__=='__main__':main()
