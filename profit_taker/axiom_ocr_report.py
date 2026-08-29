from __future__ import annotations

import argparse,json
from collections import Counter
from .db import connect,migrate

FIELDS=["market_cap_usd","volume_usd","fees_sol","txns","holders","pro_traders","kols","dev_migrations","dev_creations","recent_visitors","top10_holders_pct","funding_time_minutes","sniper_pct","insider_pct","bundler_pct","dex_paid"]

def report(db):
    migrate(db); con=connect(db)
    try: rows=[dict(r) for r in con.execute("SELECT * FROM axiom_observations ORDER BY observation_id DESC LIMIT 5000")]
    finally: con.close()
    out={"observations":len(rows),"field_coverage":{},"ocr_rejection_reasons":Counter()}
    for f in FIELDS: out["field_coverage"][f]=sum(r.get(f) is not None for r in rows)/len(rows) if rows else 0
    for r in rows:
        try: d=json.loads(r.get("raw_ocr_json") or "{}")
        except: d={}
        for rec in d.values():
            if isinstance(rec,dict) and rec.get("reason") and rec.get("reason")!="accepted_consensus": out["ocr_rejection_reasons"][rec["reason"]]+=1
    out["ocr_rejection_reasons"]=dict(out["ocr_rejection_reasons"])
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--db",default="data/live.sqlite"); a=ap.parse_args(); print(json.dumps(report(a.db),indent=2))
if __name__=="__main__":main()
