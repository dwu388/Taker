"""Compatibility location for forward-trade normalization.

The default Axiom collector never imports this module. Provider execution is explicit.
"""
from __future__ import annotations
import hashlib,json

def normalize_balance_change(signature,wallet,token_address,token_delta,quote_usd_delta,timestamp):
    if not token_delta:return None
    side='buy' if token_delta>0 else 'sell';amount=abs(float(token_delta));usd=abs(float(quote_usd_delta)) if quote_usd_delta is not None else None
    price=usd/amount if usd is not None and amount else None;raw=f'{signature}:{wallet}:{token_address}:{side}'
    return {'trade_id':hashlib.sha256(raw.encode()).hexdigest(),'timestamp':timestamp,'signature':signature,'wallet':wallet,'token_address':token_address,'side':side,'token_amount':amount,'usd_value':usd,'price_usd':price,'source':'helius','raw_json':json.dumps({'token_delta':token_delta,'quote_usd_delta':quote_usd_delta})}
