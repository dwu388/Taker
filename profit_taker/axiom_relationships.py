from __future__ import annotations
import argparse,json
import numpy as np
import pandas as pd
from .axiom_train_24h import load_training_frame,feature_columns

def analyze(db,target='success_2x_before_50dd_or_dead_24h',bins=8,top=40):
    df=load_training_frame(db);cols=feature_columns(df);out={}
    if target not in df:return {'error':f'missing target {target}'}
    for c in cols[:]:
        sub=df[[c,target]].dropna()
        if len(sub)<30 or sub[c].nunique()<3:continue
        try:sub['bin']=pd.qcut(sub[c],q=min(bins,sub[c].nunique()),duplicates='drop')
        except:continue
        g=sub.groupby('bin',observed=True).agg(n=(target,'size'),target_rate=(target,'mean'),feature_mean=(c,'mean')).reset_index();
        spread=float(g['target_rate'].max()-g['target_rate'].min()) if len(g)>1 else 0
        out[c]={'spread':spread,'bins':[{**r,'bin':str(r['bin'])} for r in g.to_dict('records')]}
    ranked=dict(sorted(out.items(),key=lambda kv:kv[1]['spread'],reverse=True)[:top]);return {'target':target,'features':ranked}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--target',default='success_2x_before_50dd_or_dead_24h');ap.add_argument('--out',default='data/axiom_relationships_v18.json');a=ap.parse_args();r=analyze(a.db,a.target);open(a.out,'w',encoding='utf-8').write(json.dumps(r,indent=2,default=str));print(a.out)
if __name__=='__main__':main()
