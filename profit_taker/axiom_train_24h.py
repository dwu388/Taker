from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_absolute_error, mean_pinball_loss, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from .common import parse_iso
from .db import connect, migrate

HORIZONS=("1h","4h","12h","24h")
UP=("plus30","plus50","plus100","plus200","plus400")
DOWN=("minus30","minus50")
CLASS_TARGETS=[f"hit_{u}_by_{h}" for h in HORIZONS for u in UP] + [f"hit_{d}_by_{h}" for h in HORIZONS for d in DOWN] + [
    "success_2x_before_50dd_or_dead_24h","success_50_before_30dd_or_dead_24h","dead_before_2x_24h",
    "dead_within_1h_24h","dead_within_4h_24h","dead_within_12h_24h",
]
REG_TARGETS=[
    "log_peak_multiple_24h","log1p_time_to_peak_minutes_24h","drawdown_before_peak_pct_24h",
    "peak_to_trough_before_peak_pct_24h","log1p_time_to_death_minutes_if_death_24h",
]
QUANTILE_ALPHAS=(.10,.25,.50,.75,.90)
CAPACITY={
    "compact":{"num_leaves":15,"max_depth":3,"min_child":10},
    "balanced":{"num_leaves":31,"max_depth":5,"min_child":8},
    "expressive":{"num_leaves":63,"max_depth":7,"min_child":5},
}


def load_training_frame(db_path:str)->pd.DataFrame:
    migrate(db_path); con=connect(db_path)
    try:
        rows=con.execute("""
            SELECT o.observation_id,o.token_key,o.snapshot_at,f.feature_json,l.label_status,l.target_json
            FROM axiom_observations o
            JOIN axiom_features_v18 f USING(observation_id)
            JOIN axiom_labels_24h_v18 l USING(observation_id)
            ORDER BY o.snapshot_at
        """).fetchall()
    finally: con.close()
    recs=[]
    for r in rows:
        feat=json.loads(r["feature_json"]); targets=json.loads(r["target_json"])
        rec={"observation_id":r["observation_id"],"token_key":r["token_key"],"snapshot_at":r["snapshot_at"],"label_status":r["label_status"],**feat,**targets}
        recs.append(rec)
    return pd.DataFrame(recs)


def feature_columns(df:pd.DataFrame)->list[str]:
    forbidden={"observation_id","token_key","snapshot_at","label_status",*CLASS_TARGETS,*REG_TARGETS,
               "peak_multiple_24h","time_to_peak_before_terminal_minutes_24h","time_to_death_minutes_if_death_24h",
               "max_return_before_terminal_24h","min_return_before_terminal_24h","return_at_terminal_observation_24h","observable_lifetime_minutes_24h"}
    cols=[]
    for c in df.columns:
        if c in forbidden: continue
        if c.startswith("hit_") or c.startswith("success_") or c.startswith("dead_"): continue
        if pd.api.types.is_numeric_dtype(df[c]): cols.append(c)
    # Defensively exclude operational visibility/capture features even if a future schema accidentally joins them.
    blocked=("capture_count","visibility_bonus","consecutive_capture_count","reappearance_count","observation_index","interval_minutes","obs_count_")
    return [c for c in cols if not any(c==b or c.startswith(b) for b in blocked)]


def _token_first_times(df):
    return df.groupby("token_key")["snapshot_at"].min().sort_values()


def split_masks(df:pd.DataFrame, allow_small:bool=False, purge_hours:int=144):
    times=_token_first_times(df)
    tokens=list(times.index)
    if len(tokens)<4: raise RuntimeError("Need at least 4 distinct tokens for a token-grouped validation split")
    cut=max(1,min(len(tokens)-1,int(len(tokens)*.8)))
    val_tokens=set(tokens[cut:]); val_start=min(parse_iso(times[t]) for t in val_tokens)
    purge_cut=val_start-timedelta(hours=purge_hours)
    train_tokens={t for t in tokens[:cut] if parse_iso(times[t])<=purge_cut}
    if (not train_tokens or len(val_tokens)<1) and allow_small:
        train_tokens=set(tokens[:cut])
    if not train_tokens: raise RuntimeError("144-hour purge left no training cohort; collect more data or use --allow-small for a plumbing experiment")
    return df["token_key"].isin(train_tokens).to_numpy(), df["token_key"].isin(val_tokens).to_numpy(), {"train_tokens":len(train_tokens),"val_tokens":len(val_tokens),"val_start":val_start.isoformat(),"purge_hours":purge_hours if train_tokens else 0}


def _weights(tokens:pd.Series)->np.ndarray:
    counts=tokens.value_counts(); w=tokens.map(lambda t:1.0/counts[t]).to_numpy(float)
    return w/np.mean(w)


def _matrix(df, cols, medians=None):
    X=df[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    if medians is None:
        medians=X.median(axis=0).fillna(0.0)
    return X.fillna(medians).to_numpy(np.float32), medians


def _make_pair(kind:str, profile:str, n_estimators:int, alpha:float|None=None):
    p=CAPACITY[profile]
    common_lgb=dict(n_estimators=n_estimators,learning_rate=.035,num_leaves=p["num_leaves"],min_child_samples=p["min_child"],subsample=.85,colsample_bytree=.85,reg_lambda=1.0,random_state=42,verbosity=-1,n_jobs=-1)
    common_xgb=dict(n_estimators=n_estimators,learning_rate=.035,max_depth=p["max_depth"],min_child_weight=max(1,p["min_child"]//2),subsample=.85,colsample_bytree=.85,reg_lambda=1.0,random_state=42,n_jobs=-1,tree_method="hist")
    if kind=="classification":
        return LGBMClassifier(objective="binary",**common_lgb), XGBClassifier(objective="binary:logistic",eval_metric="logloss",**common_xgb)
    if kind=="regression":
        return LGBMRegressor(objective="regression",**common_lgb), XGBRegressor(objective="reg:squarederror",**common_xgb)
    if kind=="quantile":
        assert alpha is not None
        return LGBMRegressor(objective="quantile",alpha=alpha,**common_lgb), XGBRegressor(objective="reg:quantileerror",quantile_alpha=alpha,**common_xgb)
    raise ValueError(kind)


def _fit_pair(kind,profile,n_estimators,X,y,w,alpha=None):
    lgb,xgb=_make_pair(kind,profile,n_estimators,alpha)
    lgb.fit(X,y,sample_weight=w); xgb.fit(X,y,sample_weight=w)
    return lgb,xgb


def _score_pair(kind, pair, Xv,yv,alpha=None):
    lgb,xgb=pair
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if kind=="classification":
            p=(lgb.predict_proba(Xv)[:,1]+xgb.predict_proba(Xv)[:,1])/2
            return float(log_loss(yv,np.clip(p,1e-6,1-1e-6)))
        pred=(lgb.predict(Xv)+xgb.predict(Xv))/2
    if kind=="quantile": return float(mean_pinball_loss(yv,pred,alpha=alpha))
    return float(mean_absolute_error(yv,pred))


def _production_guard(df:pd.DataFrame):
    complete=df[df["label_status"]=="complete"].copy()
    if complete.empty: raise RuntimeError("No complete labels")
    dts=pd.to_datetime(complete["snapshot_at"],utc=True)
    span=(dts.max()-dts.min()).total_seconds()/86400
    positives=int(pd.to_numeric(df.get("success_2x_before_50dd_or_dead_24h"),errors="coerce").fillna(-1).eq(1).sum())
    tokens=int(df["token_key"].nunique())
    if span<30 or positives<500 or tokens<30:
        raise RuntimeError(f"Production guard not met: complete_label_span_days={span:.1f} (need 30), primary_positives={positives} (need 500), distinct_tokens={tokens} (need 30). Use --allow-small only for experiments.")


def _select_profile(df,cols,n_estimators,allow_small):
    target=next((t for t in ["success_2x_before_50dd_or_dead_24h",*CLASS_TARGETS] if t in df and df[t].notna().sum()>=12 and df[t].dropna().nunique()>=2),None)
    if target is None: return "balanced",{"reason":"no_viable_classifier_for_capacity_selection"}
    sub=df[df[target].notna()].copy(); tr,va,meta=split_masks(sub,allow_small)
    Xt,med=_matrix(sub.loc[tr],cols); Xv,_=_matrix(sub.loc[va],cols,med)
    yt=sub.loc[tr,target].astype(int).to_numpy(); yv=sub.loc[va,target].astype(int).to_numpy(); wt=_weights(sub.loc[tr,"token_key"])
    if len(np.unique(yt))<2 or len(np.unique(yv))<2: return "balanced",{"reason":"single_class_split","target":target,**meta}
    scores={}
    for profile in CAPACITY:
        try:
            pair=_fit_pair("classification",profile,n_estimators,Xt,yt,wt)
            scores[profile]=_score_pair("classification",pair,Xv,yv)
        except Exception as e:
            scores[profile]=float("inf")
    best=min(scores,key=scores.get)
    return best,{"target":target,"validation_logloss":scores,**meta}


def train_models(db_path:str, output_dir:str, allow_small:bool=False)->dict[str,Any]:
    df=load_training_frame(db_path)
    if df.empty: raise RuntimeError("No joined feature/label rows. Run prepare_v18_max_inference.bat first.")
    if not allow_small: _production_guard(df)
    cols=feature_columns(df)
    if not cols: raise RuntimeError("No numeric model features")
    n_estimators=140 if allow_small else 700
    profile,profile_meta=_select_profile(df,cols,min(n_estimators,180),allow_small)
    bundle={"version":"V18-max-inference-rebuilt","feature_columns":cols,"capacity_profile":profile,"capacity_selection":profile_meta,"heads":{},"quantiles":{},"metadata":{"allow_small":allow_small,"n_estimators":n_estimators}}
    trained=0; skipped={}

    def train_head(target,kind,alpha=None):
        nonlocal trained
        if target not in df: skipped[target]="missing_column"; return
        sub=df[df[target].notna()].copy()
        min_rows=12 if allow_small else 100
        min_tokens=4 if allow_small else 20
        if len(sub)<min_rows or sub["token_key"].nunique()<min_tokens:
            skipped[target]=f"insufficient_rows_or_tokens:{len(sub)}/{sub['token_key'].nunique()}"; return
        if kind=="classification" and sub[target].nunique()<2:
            skipped[target]="single_class"; return
        try: tr,va,split_meta=split_masks(sub,allow_small)
        except Exception as e: skipped[target]=str(e); return
        if tr.sum()<5 or va.sum()<2:
            skipped[target]="split_too_small"; return
        Xt,med=_matrix(sub.loc[tr],cols); Xv,_=_matrix(sub.loc[va],cols,med)
        ytr=sub.loc[tr,target].astype(float).to_numpy(); yv=sub.loc[va,target].astype(float).to_numpy(); wt=_weights(sub.loc[tr,"token_key"])
        if kind=="classification" and (len(np.unique(ytr))<2 or len(np.unique(yv))<2):
            skipped[target]="single_class_train_or_validation"; return
        try:
            pair=_fit_pair(kind,profile,n_estimators,Xt,ytr.astype(int) if kind=="classification" else ytr,wt,alpha)
            score=_score_pair(kind,pair,Xv,yv.astype(int) if kind=="classification" else yv,alpha)
        except Exception as e:
            skipped[target]=f"fit_failed:{type(e).__name__}:{e}"; return
        rec={"kind":kind,"lgb":pair[0],"xgb":pair[1],"medians":med.to_dict(),"validation_score":score,"split":split_meta,"rows":len(sub),"tokens":int(sub['token_key'].nunique())}
        if alpha is not None: rec["alpha"]=alpha
        if kind=="classification":
            try:
                p=(pair[0].predict_proba(Xv)[:,1]+pair[1].predict_proba(Xv)[:,1])/2
                rec["validation_auc"]=float(roc_auc_score(yv,p))
            except Exception: pass
        if alpha is None: bundle["heads"][target]=rec
        else: bundle["quantiles"][f"q{int(alpha*100):02d}"]=rec
        trained+=1

    for t in CLASS_TARGETS: train_head(t,"classification")
    for t in REG_TARGETS: train_head(t,"regression")
    # Peak distribution quantiles all learn the same log-peak target independently.
    for alpha in QUANTILE_ALPHAS: train_head("log_peak_multiple_24h","quantile",alpha)

    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    model_path=out/"latest.joblib"; joblib.dump(bundle,model_path,compress=3)
    summary={"model":str(model_path),"trained_heads":trained,"standard_heads":len(bundle["heads"]),"quantile_heads":len(bundle["quantiles"]),"capacity_profile":profile,"features":len(cols),"skipped":skipped}
    (out/"latest_training_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    return summary
