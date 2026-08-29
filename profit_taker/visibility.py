from __future__ import annotations

from .db import connect, migrate


def visibility_bonus(capture_count:int)->float:
    # Capture 1=0.00, each additional capture +0.02, capped at +0.12.
    return min(.12,max(0,capture_count-1)*.02)


def update_visibility(db_path:str, cycle_id:int, token_key:str, snapshot_at:str)->dict:
    migrate(db_path); con=connect(db_path)
    try:
        row=con.execute("SELECT * FROM axiom_visibility WHERE token_key=?",(token_key,)).fetchone()
        if row is None:
            vals={"first_seen_at":snapshot_at,"last_seen_at":snapshot_at,"capture_count":1,"consecutive_capture_count":1,"max_consecutive_capture_count":1,"reappearance_count":0,"visibility_bonus":0.0,"last_cycle_id":cycle_id}
            con.execute("INSERT INTO axiom_visibility(token_key,first_seen_at,last_seen_at,capture_count,consecutive_capture_count,max_consecutive_capture_count,reappearance_count,visibility_bonus,last_cycle_id) VALUES(?,?,?,?,?,?,?,?,?)",(token_key,*vals.values()))
        else:
            d=dict(row); consecutive=int(d["consecutive_capture_count"])+1 if d.get("last_cycle_id")==cycle_id-1 else 1
            reappear=int(d["reappearance_count"])+(1 if d.get("last_cycle_id") is not None and d.get("last_cycle_id")<cycle_id-1 else 0)
            count=int(d["capture_count"])+1; bonus=visibility_bonus(count); maxcon=max(int(d["max_consecutive_capture_count"]),consecutive)
            con.execute("UPDATE axiom_visibility SET last_seen_at=?,capture_count=?,consecutive_capture_count=?,max_consecutive_capture_count=?,reappearance_count=?,visibility_bonus=?,last_cycle_id=? WHERE token_key=?",(snapshot_at,count,consecutive,maxcon,reappear,bonus,cycle_id,token_key))
            vals={"first_seen_at":d["first_seen_at"],"last_seen_at":snapshot_at,"capture_count":count,"consecutive_capture_count":consecutive,"max_consecutive_capture_count":maxcon,"reappearance_count":reappear,"visibility_bonus":bonus,"last_cycle_id":cycle_id}
        con.commit(); return vals
    finally: con.close()
