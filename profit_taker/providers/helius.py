from __future__ import annotations
import requests

class HeliusClient:
    def __init__(self,key,timeout=30):self.url=f'https://mainnet.helius-rpc.com/?api-key={key}';self.timeout=timeout;self._id=0
    def rpc(self,method,params):
        self._id+=1;r=requests.post(self.url,json={'jsonrpc':'2.0','id':self._id,'method':method,'params':params},timeout=self.timeout);r.raise_for_status();p=r.json()
        if p.get('error'):raise RuntimeError(f'Helius RPC {method}: {p["error"]}')
        return p.get('result')
    def get_asset(self,mint):return self.rpc('getAsset',{'id':mint,'displayOptions':{'showFungible':True}})
    def get_account_info(self,mint):return self.rpc('getAccountInfo',[mint,{'encoding':'jsonParsed'}])
    def get_token_largest_accounts(self,mint):return self.rpc('getTokenLargestAccounts',[mint])

def summarize_token(mint,asset,account,largest):
    token_info=(asset or {}).get('token_info') or (asset or {}).get('tokenInfo') or {}
    parsed=(((account or {}).get('value') or {}).get('data') or {}).get('parsed') or {}; info=parsed.get('info') or {}
    supply=token_info.get('supply') or info.get('supply');dec=token_info.get('decimals') or info.get('decimals')
    price=token_info.get('price_info') or token_info.get('priceInfo') or {};price=price.get('price_per_token') or price.get('pricePerToken') or price.get('price')
    vals=(largest or {}).get('value') or []; total=0.0
    try:
        denom=float(supply or 0)/(10**int(dec or 0))
        if denom>0:
            total=sum(float((v.get('uiAmount') if v.get('uiAmount') is not None else (v.get('uiAmountString') or 0))) for v in vals[:10])/denom*100
    except:total=0.0
    return {'token_address':mint,'price_usd':price,'supply':supply,'decimals':dec,'mint_authority':info.get('mintAuthority'),'freeze_authority':info.get('freezeAuthority'),'mint_authority_revoked':1 if info.get('mintAuthority') is None else 0,'freeze_authority_revoked':1 if info.get('freezeAuthority') is None else 0,'top10_token_account_pct':total}
