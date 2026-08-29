from __future__ import annotations
from .db import connect,migrate

def enqueue(db,wallet,priority,reason='shyft_prescreen',stage='96h'):
    migrate(db); con=connect(db)
    try:
        con.execute("""INSERT INTO wallet_backfill_queue(wallet,priority,stage,state,reason) VALUES(?,?,?,?,?)
        ON CONFLICT(wallet) DO UPDATE SET priority=MAX(priority,excluded.priority),reason=excluded.reason,updated_at=CURRENT_TIMESTAMP""",(wallet,float(priority),stage,'queued',reason));con.commit()
    finally:con.close()

def next_items(db,limit=25):
    con=connect(db)
    try:return [dict(r) for r in con.execute("SELECT * FROM wallet_backfill_queue WHERE state='queued' ORDER BY priority DESC,queued_at LIMIT ?",(int(limit),))]
    finally:con.close()
