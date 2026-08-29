from __future__ import annotations
from datetime import datetime,timezone
from .db import connect,migrate

BIRDEYE_MONTHLY_MAX=30000.0
BIRDEYE_DAILY_SOFT=900.0
BIRDEYE_DAILY_HARD=1000.0


def usage(db,provider='birdeye',day=None):
    migrate(db); day=day or datetime.now(timezone.utc).date().isoformat(); con=connect(db)
    try:
        daily=float(con.execute('SELECT COALESCE(SUM(units),0) FROM provider_usage WHERE provider=? AND usage_date=?',(provider,day)).fetchone()[0])
        month=day[:7]+'%'; monthly=float(con.execute('SELECT COALESCE(SUM(units),0) FROM provider_usage WHERE provider=? AND usage_date LIKE ?',(provider,month)).fetchone()[0])
        return {'daily':daily,'monthly':monthly}
    finally:con.close()


def reserve_birdeye(db,units=10.0,allow_over_soft=False):
    u=usage(db,'birdeye')
    if u['monthly']+units>BIRDEYE_MONTHLY_MAX: raise RuntimeError('Birdeye monthly 30,000-CU circuit breaker reached')
    if u['daily']+units>BIRDEYE_DAILY_HARD: raise RuntimeError('Birdeye daily 1,000-CU hard circuit breaker reached')
    if not allow_over_soft and u['daily']+units>BIRDEYE_DAILY_SOFT: raise RuntimeError('Birdeye daily 900-CU soft budget reached')


def commit_usage(db,provider,units):
    migrate(db); day=datetime.now(timezone.utc).date().isoformat(); con=connect(db)
    try:
        con.execute('INSERT INTO provider_usage(provider,usage_date,units) VALUES(?,?,?) ON CONFLICT(provider,usage_date) DO UPDATE SET units=units+excluded.units',(provider,day,float(units))); con.commit()
    finally:con.close()
