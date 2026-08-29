from __future__ import annotations
import argparse,csv,json,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from .config import load_env,key
from .db import connect,migrate
from .budget import reserve_birdeye,commit_usage
from .providers.birdeye import BirdeyeClient,normalize_swaps
from .providers.helius import HeliusClient,summarize_token

def _unix(dt): return int(dt.timestamp())
def _store_trades(con,trades):
    for t in trades:
        con.execute('INSERT OR IGNORE INTO normalized_trades(trade_id,timestamp,signature,wallet,token_address,side,token_amount,usd_value,price_usd,source,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(t['trade_id'],t['timestamp'],t.get('signature'),t.get('wallet') or '',t['token_address'],t['side'],t.get('token_amount'),t.get('usd_value'),t.get('price_usd'),t['source'],t.get('raw_json')))
def run(tokens_path,db,env_file='.env'):
    env=load_env(env_file);helius=HeliusClient(key(env,'HELIUS_API_KEY'));birdeye=BirdeyeClient(key(env,'BIRDEYE_API_KEY'));migrate(db)
    tokens=[r['token_address'].strip() for r in csv.DictReader(Path(tokens_path).open(encoding='utf-8-sig')) if r.get('token_address')]
    now=datetime.now(timezone.utc);con=connect(db);summary={'tokens':len(tokens),'helius_observations':0,'birdeye_trades':0}
    try:
        for mint in tokens:
            asset=helius.get_asset(mint);acct=helius.get_account_info(mint);largest=helius.get_token_largest_accounts(mint);s=summarize_token(mint,asset,acct,largest)
            con.execute('INSERT OR REPLACE INTO helius_token_observations(token_address,observed_at,price_usd,supply,decimals,mint_authority,freeze_authority,mint_authority_revoked,freeze_authority_revoked,top10_token_account_pct,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(mint,now.isoformat(),s.get('price_usd'),s.get('supply'),s.get('decimals'),s.get('mint_authority'),s.get('freeze_authority'),s.get('mint_authority_revoked'),s.get('freeze_authority_revoked'),s.get('top10_token_account_pct'),json.dumps({'asset':asset,'account':acct,'largest':largest},default=str)));summary['helius_observations']+=1
            st=con.execute('SELECT last_synced_at FROM token_sync_state WHERE token_address=?',(mint,)).fetchone();start=now-timedelta(minutes=90) if not st or not st[0] else datetime.fromisoformat(st[0].replace('Z','+00:00'))-timedelta(seconds=3)
            reserve_birdeye(db,10);payload=birdeye.token_trades(mint,_unix(start),_unix(now),0,100);commit_usage(db,'birdeye',10);tr=normalize_swaps(payload,focus_token=mint);_store_trades(con,tr);summary['birdeye_trades']+=len(tr)
            con.execute('INSERT INTO token_sync_state(token_address,last_synced_at,history_backfilled_to) VALUES(?,?,?) ON CONFLICT(token_address) DO UPDATE SET last_synced_at=excluded.last_synced_at',(mint,now.isoformat(),start.isoformat()))
        con.commit();return summary
    finally:con.close()
def main():
    ap=argparse.ArgumentParser(description='Explicit provider cycle; never called by the default Axiom runner');ap.add_argument('--tokens',default='data/tokens.csv');ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--env-file',default='.env');a=ap.parse_args();print(json.dumps(run(a.tokens,a.db,a.env_file),indent=2))
if __name__=='__main__':main()
