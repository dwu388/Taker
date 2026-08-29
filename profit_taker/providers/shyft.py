from __future__ import annotations
import time,requests
BASE='https://api.shyft.to/sol/v1/transaction/history'
class ShyftClient:
    def __init__(self,key,timeout=30):self.key=key;self.timeout=timeout
    def history(self,wallet,limit=100,before=None,until=None):
        params={'network':'mainnet-beta','account':wallet,'tx_num':min(100,int(limit))}
        if before:params['before_tx_signature']=before
        if until:params['until_tx_signature']=until
        for attempt in range(4):
            r=requests.get(BASE,params=params,headers={'x-api-key':self.key},timeout=self.timeout)
            if r.status_code==429 and attempt<3:time.sleep(1.5*(attempt+1));continue
            r.raise_for_status();p=r.json();return p.get('result') or p.get('data') or []
        return []
    def paged_history(self,wallet,max_pages=3):
        rows=[];before=None;requests_used=0
        for _ in range(max_pages):
            page=self.history(wallet,100,before=before);requests_used+=1
            if not page:break
            rows.extend(page);last=page[-1];before=last.get('signatures',[None])[0] if isinstance(last.get('signatures'),list) else last.get('signature')
            if len(page)<100 or not before:break
        return rows,requests_used
