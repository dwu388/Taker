from __future__ import annotations
import hashlib,json,time
from typing import Any
import requests

BASE='https://public-api.birdeye.so'
QUOTE_MINTS={
 'So11111111111111111111111111111111111111112','EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v','Es9vMFrzaCERmJfrF4H2FYDqAYhB8p7RGeWrnmxNmHaL'
}

class BirdeyeClient:
    def __init__(self,key,timeout=30,min_interval=1.05):self.key=key;self.timeout=timeout;self.min_interval=min_interval;self._last=0.0
    def _get(self,path,params):
        wait=self.min_interval-(time.monotonic()-self._last)
        if wait>0:time.sleep(wait)
        r=requests.get(BASE+path,params=params,headers={'X-API-KEY':self.key,'accept':'application/json','x-chain':'solana'},timeout=self.timeout);self._last=time.monotonic();r.raise_for_status();return r.json()
    def token_trades(self,address,time_from,time_to,offset=0,limit=100):
        return self._get('/defi/txs/token/seek_by_time',{'address':address,'tx_type':'swap','time_from':int(time_from),'time_to':int(time_to),'offset':offset,'limit':limit})
    def trader_trades(self,wallet,time_from,time_to,offset=0,limit=100):
        return self._get('/trader/txs/seek_by_time',{'address':wallet,'tx_type':'swap','time_from':int(time_from),'time_to':int(time_to),'offset':offset,'limit':limit})

def _items(payload):
    d=payload.get('data',payload) if isinstance(payload,dict) else payload
    if isinstance(d,dict):
        for k in ('items','txs','transactions','data'):
            if isinstance(d.get(k),list):return d[k]
    return d if isinstance(d,list) else []

def _leg_mint(leg): return leg.get('address') or leg.get('mint') or leg.get('token_address') or leg.get('tokenAddress')
def _leg_amount(leg):
    for k in ('ui_amount','uiAmount','amount','token_amount','tokenAmount'):
        try:
            if leg.get(k) is not None:return float(leg[k])
        except:pass
    return None

def normalize_swaps(payload,focus_token=None,wallet_hint=None):
    out=[]
    for idx,item in enumerate(_items(payload)):
        base=item.get('base') or {};quote=item.get('quote') or {}
        legs=[x for x in (base,quote) if isinstance(x,dict)]
        wallet=item.get('owner') or item.get('wallet') or item.get('address') or wallet_hint
        sig=item.get('tx_hash') or item.get('txHash') or item.get('signature') or item.get('tx')
        ts=item.get('block_unix_time') or item.get('blockUnixTime') or item.get('timestamp') or item.get('block_time')
        for li,leg in enumerate(legs):
            mint=_leg_mint(leg)
            if not mint or mint in QUOTE_MINTS:continue
            if focus_token and mint!=focus_token:continue
            typ=str(leg.get('type_swap') or leg.get('typeSwap') or leg.get('direction') or '').lower()
            side='buy' if typ in ('to','in','receive','received') else ('sell' if typ in ('from','out','send','sent') else None)
            if side is None:continue
            amount=_leg_amount(leg)
            price=None;usd=None
            for k in ('price','price_usd','priceUsd','token_price'):
                try:
                    if leg.get(k) is not None:price=float(leg[k]);break
                except:pass
            for k in ('value_usd','valueUsd','usd_value','usdValue'):
                try:
                    if leg.get(k) is not None:usd=float(leg[k]);break
                except:pass
            if usd is None and amount is not None and price is not None:usd=amount*price
            rawid=f'{sig}:{item.get("tx_index",item.get("txIndex",idx))}:{item.get("instruction_index",item.get("instructionIndex",0))}:{li}:{mint}'
            tid=hashlib.sha256(rawid.encode()).hexdigest()
            out.append({'trade_id':tid,'timestamp':str(ts),'signature':sig,'wallet':wallet,'token_address':mint,'side':side,'token_amount':amount,'usd_value':usd,'price_usd':price,'source':'birdeye','raw_json':json.dumps(item,default=str)})
    return out
