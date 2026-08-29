from __future__ import annotations
import argparse,json,statistics
from collections import defaultdict,deque
from .db import connect,migrate

def rebuild(db,as_of):
    """Build unbiased snapshots from all mature normalized trades; losing exits are retained."""
    migrate(db);con=connect(db)
    try:
        trades=[dict(r) for r in con.execute('SELECT * FROM normalized_trades WHERE timestamp<=? ORDER BY wallet,timestamp',(as_of,))]
        by=defaultdict(list)
        for t in trades:by[t['wallet']].append(t)
        written=0
        for wallet,rows in by.items():
            lots=defaultdict(deque);exits=[]
            for t in rows:
                token=t['token_address'];amt=float(t.get('token_amount') or 0);usd=float(t.get('usd_value') or 0)
                if amt<=0:continue
                if t['side']=='buy':lots[token].append([amt,usd/amt if amt else 0,t['timestamp']])
                elif t['side']=='sell':
                    rem=amt;proceeds_per=usd/amt if amt else 0
                    while rem>0 and lots[token]:
                        lot=lots[token][0];take=min(rem,lot[0]);cost=take*lot[1];proceeds=take*proceeds_per;pnl=proceeds-cost;roi=pnl/cost if cost else 0
                        exits.append((token,pnl,roi,cost,proceeds));lot[0]-=take;rem-=take
                        if lot[0]<=1e-12:lots[token].popleft()
            if not exits:continue
            pnls=[x[1] for x in exits];rois=[x[2] for x in exits];wins=sum(x>0 for x in pnls);losses=sum(x<0 for x in pnls)
            con.execute('INSERT OR REPLACE INTO wallet_history_snapshots_v12(wallet,as_of,mature_exit_count,winning_exit_count,losing_exit_count,win_rate,total_realized_pnl_usd,median_realized_pnl_usd,mean_roi,median_roi,distinct_tokens,total_cost_basis_usd,total_proceeds_usd) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(wallet,as_of,len(exits),wins,losses,wins/len(exits),sum(pnls),statistics.median(pnls),statistics.mean(rois),statistics.median(rois),len({x[0] for x in exits}),sum(x[3] for x in exits),sum(x[4] for x in exits)));written+=1
        con.commit();return {'wallet_snapshots':written}
    finally:con.close()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--as-of',required=True);a=ap.parse_args();print(json.dumps(rebuild(a.db,a.as_of),indent=2))
if __name__=='__main__':main()
