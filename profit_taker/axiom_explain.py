from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np
from .axiom_predict_24h import load_current_rows,_matrix_row

def explain(db,model,target='success_2x_before_50dd_or_dead_24h',limit=20):
    b=joblib.load(model);rec=b['heads'].get(target)
    if not rec:return {'error':f'head not trained: {target}'}
    rows=load_current_rows(db,False)
    if not rows:return {'error':'no current rows'}
    try:import shap
    except Exception as e:return {'error':f'shap is not installed: {e}'}
    explainer=shap.TreeExplainer(rec['lgb']);out=[]
    for r in rows:
        feat=json.loads(r['feature_json']);X=_matrix_row(feat,b['feature_columns'],rec['medians']);sv=explainer.shap_values(X)
        arr=np.asarray(sv[-1] if isinstance(sv,list) else sv).reshape(-1);pairs=sorted(zip(b['feature_columns'],arr),key=lambda kv:abs(kv[1]),reverse=True)[:limit]
        out.append({'token_key':r['token_key'],'snapshot_at':r['snapshot_at'],'top_shap':[{'feature':f,'shap':float(v),'value':feat.get(f)} for f,v in pairs]})
    return {'target':target,'rows':out}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/live.sqlite');ap.add_argument('--model',default='models/axiom24/latest.joblib');ap.add_argument('--target',default='success_2x_before_50dd_or_dead_24h');ap.add_argument('--out',default='data/axiom_shap_v18.json');a=ap.parse_args();r=explain(a.db,a.model,a.target);Path(a.out).write_text(json.dumps(r,indent=2,default=str),encoding='utf-8');print(a.out)
if __name__=='__main__':main()
