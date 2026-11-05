#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OIBC No.28 site-level low-RH residual correction, strictly local/offline.

Two independent methods: Ridge (squared-loss baseline) and LightGBM quantile q=.5.
All calibration labels: 146 GroupOpt OOS days. 5-fold delivery-day OOF correction
for grouping. Independent Holdout: only prediction/evaluation, never fitting.

Relies on unchanged previously-delivered `no28_rh_conditional_analysis.py`
for exact No.28 validation and forecast feature joins, and ORIGINAL
`baseline_optimizer.py` for identical Greedy optimizer.

From data root:
  python3 -u codes/no28_rh_residual_models.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from no28_rh_conditional_analysis import (load_no28_rows, attach_weather,
    delivery_day, KST, WX_COLS, REVISION_COLS, BINS)

BASELINE_W = {'groupopt': 64637423.578663826, 'holdout': 63669335.6497201}
BASELINE_N = {'groupopt': 678254, 'holdout': 709834}
BASELINE_NH = {'groupopt': 3504, 'holdout': 3528}
MIN_CAP = 20000.0
ERR_EPS = 1e-9
RESEARCH_VERSION = 'no28_rh_site_residual_20261104_v1'


def log(*p):
    print('[RH-RESID]', *p, flush=True)


def json_ready(x):
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [json_ready(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, (float,)) and not np.isfinite(x): return None
    return x


def store_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def store_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.csv')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(path)


def load_tables(root):
    st = time.monotonic()
    site, src = load_no28_rows(root)
    log('No28 OOS residual rows verified from', src)
    site = attach_weather(site, root)
    cache = root / 'output/newcand_no28/_shared/no28_seed42_nat_feat.parquet'
    if not cache.exists():
        raise FileNotFoundError(f'No28 OOS forecast cache required for site correction: {cache}')
    preds = pd.read_parquet(cache, columns=['site_id', 'ts', 'pred_kwh'])
    from no28_rh_conditional_analysis import kst
    preds['ts'] = kst(preds.ts)
    if preds.duplicated(['site_id', 'ts']).any():
        raise ValueError('Duplicate cached No28 prediction keys')
    pre_n = len(site)
    site = site.merge(preds, on=['site_id', 'ts'], how='left', validate='one_to_one', sort=False)
    if len(site) != pre_n or site.pred_kwh.isna().any():
        raise ValueError('Missing/duplicated No28 OOS predictions on site residual labels')
    if (site.pred_kwh < -1e-6).any():
        raise ValueError('Base No28 unexpectedly has negative predictions; stop')
    site['pred_kwh'] = site.pred_kwh.clip(lower=0).astype('float32')
    site['actual_kwh'] = (site['pred_kwh'].astype(float) - site.e_50.astype(float)).astype('float32')
    if (site.actual_kwh < -0.02).any():
        raise ValueError('Negative actual energy recovered; No28 prediction/residual caches may differ')
    # Double-check e against recovered ground truth before using it.
    if not np.allclose(site.pred_kwh.to_numpy(dtype=float)-site.actual_kwh.to_numpy(dtype=float),
                       site.e_50.to_numpy(dtype=float), rtol=1.e-4, atol=1.e-3):
        raise ValueError('No28 residual/parquet prediction conflict')
    site['actual_kwh'] = site.actual_kwh.clip(lower=0)
    site['delivery_day'] = delivery_day(site.ts).dt.strftime('%Y-%m-%d')
    site['rh_band'] = pd.cut(site.wb_rel_hum,
        [-np.inf,20,40,60,np.inf], labels=['RH_le20','RH_20_40','RH_40_60','RH_gt60']).astype('object').fillna('MISSING')
    site['correctable'] = site['rh_band'].isin(['RH_20_40','RH_40_60'])
    if len(site) != sum(BASELINE_N.values()):
        raise ValueError('Unexpected total OOS row count')
    log('Rows and RH correction coverage:', site.groupby('split', observed=True).agg(
         rows=('correctable','size'), correctable=('correctable','sum')).to_dict('index'))
    log('Forecasts/labels/weather prepared in', round(time.monotonic()-st,1),'sec')
    return site


def feature_matrix(site):
    """Only pre-issued NWP+No28 forecast and static capacity/time, NO labels."""
    p = site.pred_kwh.to_numpy(dtype=np.float64)
    cap = site.capacity_kw.to_numpy(dtype=np.float64)
    ts = site.ts
    hr = ts.dt.hour.to_numpy(dtype=np.float64)
    doy = ts.dt.dayofyear.to_numpy(dtype=np.float64)
    month = ts.dt.month.to_numpy(dtype=np.float64)
    features = {
        'no28_pred_cf': p / np.maximum(cap,1),
        'no28_pred_kwh': p,
        'capacity_kw': cap,
        'log_capacity': np.log1p(cap),
        'hour': hr,
        'hour_sin': np.sin(2*np.pi*hr/24),
        'hour_cos': np.cos(2*np.pi*hr/24),
        'doy_sin': np.sin(2*np.pi*doy/365.25),
        'doy_cos': np.cos(2*np.pi*doy/365.25),
        'season_winter': np.isin(month,[12,1,2]).astype(float),
        'season_spring': np.isin(month,[3,4,5]).astype(float),
        'season_summer': np.isin(month,[6,7,8]).astype(float),
        'season_autumn': np.isin(month,[9,10,11]).astype(float),
    }
    wxcols = WX_COLS['wa'] + WX_COLS['wb']
    gapcols = ['cloud_diff_wa_minus_wb','rh_diff_wa_minus_wb','temp_diff_wa_minus_wb']
    revcols = ['revision_'+v for vv in REVISION_COLS.values() for v in vv]
    for col in wxcols + gapcols + revcols:
        features[col] = pd.to_numeric(site[col], errors='coerce').to_numpy(dtype=np.float32)
    cols = list(features)
    X = pd.DataFrame(features, index=np.arange(len(site))).astype('float32')
    if np.isinf(X.to_numpy(copy=False)).any():
        raise ValueError('Infinite feature values')
    if X.shape[1] != len(cols): raise AssertionError('Feature count mismatch')
    log('Feature columns:',len(cols), cols)
    return X


def day_folds(d, nfold, seed):
    from sklearn.model_selection import KFold
    days=np.array(sorted(d.delivery_day.unique()))
    if len(days)<nfold*2: raise ValueError('Not enough GroupOpt delivery days for folds')
    folds=[]
    kk=KFold(n_splits=nfold,shuffle=True,random_state=seed)
    for j,(tr,va) in enumerate(kk.split(days),1):
        train_day=set(days[tr]); valid_day=set(days[va]);
        train_idx=np.flatnonzero(d.delivery_day.isin(train_day).to_numpy())
        val_idx=np.flatnonzero(d.delivery_day.isin(valid_day).to_numpy())
        if np.intersect1d(train_idx,val_idx).size: raise AssertionError('Fold overlap')
        folds.append((j,train_idx,val_idx))
        log(f'DayFold{j}: train={len(train_day)} days / {len(train_idx):,} rows, '
            f'valid={len(valid_day)} days / {len(val_idx):,} rows')
    seen=np.concatenate([f[2] for f in folds])
    if len(seen)!=len(d) or not np.array_equal(np.sort(seen),np.arange(len(d))):
        raise ValueError('GroupOpt OOF validation coverage not exactly once')
    return folds


def build_model(name,threads,trees):
    if name=='ridge':
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        return Pipeline([
            ('impute',SimpleImputer(strategy='median',add_indicator=True)),
            ('scale',StandardScaler()),
            ('ridge',Ridge(alpha=40.0,solver='lsqr',tol=1.e-4)),
        ])
    if name=='lightgbm_q50':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective='quantile',alpha=0.5,learning_rate=0.05,
            n_estimators=trees,num_leaves=11,max_depth=5,min_child_samples=280,
            colsample_bytree=1.0,reg_lambda=2.0,max_bin=127,n_jobs=threads,
            verbosity=-1,random_state=42)
    raise ValueError(name)


def fit_predict_groupopt(name, X, y_cf, weights, folds, out, args, inputs_tag):
    """OOF on GroupOpt, then full fit for strictly label-free Holdout predict."""
    from joblib import dump, load
    model_out=out/name
    model_out.mkdir(parents=True,exist_ok=True)
    oof_cache=model_out/'groupopt_oof_pred_correction_cf.npy'
    final_cache=model_out/'groupopt_full_model.joblib'
    meta_cache=model_out/'cache_config.json'
    config=dict(version=RESEARCH_VERSION,method=name,seed=args.seed,folds=args.folds,
                trees=args.trees,features=list(X.columns),shrinkage=args.shrinkage,
                max_correction_cf=args.max_correction_cf, inputs_tag=inputs_tag)
    resume = (not args.fresh) and meta_cache.exists() and \
             json.loads(meta_cache.read_text(encoding='utf-8'))==config
    if resume and oof_cache.is_file() and final_cache.is_file():
        z=np.load(oof_cache)
        if z.shape==(len(X),) and np.isfinite(z).all():
            log(name,'RESUME validated OOF + final model cache')
            return z,load(final_cache)
    store_json(meta_cache,config)
    oof=np.full(len(X),np.nan,dtype='float32')
    for fold,train,val in folds:
        model=build_model(name,args.threads,args.trees)
        t0=time.monotonic()
        model.fit(X.iloc[train],y_cf[train],
                  **({'ridge__sample_weight':weights[train]} if name=='ridge' else
                     {'sample_weight':weights[train]}))
        q=model.predict(X.iloc[val])
        if not np.isfinite(q).all(): raise ValueError(f'{name}: nonfinite Fold{fold} predictions')
        oof[val]=np.asarray(q,dtype='float32')
        log(name,f'fold {fold}/{args.folds} done in {time.monotonic()-t0:.1f}s')
    if not np.isfinite(oof).all():raise ValueError('OOF contains missing predictions')
    np.save(oof_cache,oof)
    log(name,'Training final model on all 146 GroupOpt days (Holdout never fitted)')
    model=build_model(name,args.threads,args.trees)
    model.fit(X,y_cf, **({'ridge__sample_weight':weights} if name=='ridge' else
                          {'sample_weight':weights}))
    dump(model,final_cache,compress=2)
    return oof,model


def corrected_residuals(site, model_r_cf, args):
    """Use a site-level Round17 WB RH gate; no correction outside (20,60]."""
    cap=site.capacity_kw.to_numpy(dtype=np.float64)
    prior=site.pred_kwh.to_numpy(dtype=np.float64)
    e=site.e_50.to_numpy(dtype=np.float64)
    use=site.correctable.to_numpy(dtype=bool) & np.isfinite(model_r_cf)
    correction_cf=np.clip(np.asarray(model_r_cf,dtype=np.float64),
                          -args.max_correction_cf,+args.max_correction_cf)
    correction_kwh=np.where(use,args.shrinkage*correction_cf*cap,0.0)
    new_pred=np.maximum(0.,prior+correction_kwh)
    # Some old No28 cache predictions may be zero at night; post-clipping handles them.
    new_res=e+(new_pred-prior)
    return new_res,new_pred,np.where(use,new_pred-prior,0.)


def residual_matrix_for_split(site, values, sites):
    """Observed site residuals only; missing site-hours contribute 0, never actual=0."""
    if len(values)!=len(site): raise ValueError('Residual array must align rows')
    times=pd.DatetimeIndex(site.ts.unique()).sort_values()
    nsite=len(sites)
    ti=times.get_indexer(site.ts)
    if (ti<0).any():raise ValueError('Timestamp indexing failure')
    si=site.site_id.map(pd.Series(np.arange(nsite),index=sites.site_id)).to_numpy()
    if pd.isna(si).any():raise ValueError('Site index mismatch')
    si=si.astype(np.int64)
    if len(np.unique(ti*nsite+si))!=len(site):
        raise ValueError('Duplicate site-hour residual keys')
    matrix=np.zeros((len(times),nsite),dtype=np.float64)
    np.add.at(matrix,(ti,si),values)
    wkdf=site[['ts','wk']].groupby('ts',sort=True).agg(first=('wk','first'),
                                                       max=('wk','max'),min=('wk','min'))
    if len(wkdf)!=len(times) or not np.allclose(wkdf['first'],wkdf['max'],atol=1e-6) or \
                                    not np.allclose(wkdf['first'],wkdf['min'],atol=1e-6):
        raise ValueError('Weight inconsistency at time')
    wk=wkdf.reindex(times)['first'].to_numpy(dtype=np.float64)
    return matrix,times,wk


def matrix_scores(mat,wk,mask):
    total=mat.sum(axis=1)
    e1=mat[:,mask].sum(axis=1)
    e2=total-e1
    w=float(.1*np.dot(wk,np.abs(e1)+np.abs(e2)))
    tlb=float(.1*np.dot(wk,np.abs(total)))
    return w,tlb


def summarise_sites(df, error_after, predicted_r_cf, name, split):
    cap=df.capacity_kw.to_numpy(dtype=np.float64)
    wk=df.wk.to_numpy(dtype=np.float64)
    err=df.e_50.to_numpy(dtype=np.float64)
    actual_corr=-err/cap
    pc=np.asarray(predicted_r_cf,dtype=float)
    # Direction accuracy only when true error >= 1% capacity and predicted
    # correction >= 0.5% capacity. Exclude near-zero labels and abstentions.
    significant=np.abs(actual_corr)>=.01
    decisive=np.abs(pc)>=.005
    rows=[]
    masks=[('ALL',np.ones(len(df),dtype=bool)),
           ('RH_20_40',df.rh_band.eq('RH_20_40').to_numpy()),
           ('RH_40_60',df.rh_band.eq('RH_40_60').to_numpy()),
           ('RH_other',~df.correctable.to_numpy(dtype=bool))]
    for label,m in masks:
        if not m.any():continue
        val=np.sum(wk[m])
        selection=m&significant
        predicting=selection&decisive
        rows.append(dict(method=name,split=split,rh_band=label,n_site_hours=int(m.sum()),
             n_days=int(df.loc[m,'delivery_day'].nunique()),
             base_WMAE_kwh=float(np.dot(wk[m],np.abs(err[m]))/val),
             corrected_WMAE_kwh=float(np.dot(wk[m],np.abs(error_after[m]))/val),
             base_over_pct=float(100*(err[m]>ERR_EPS).mean()),
             corrected_over_pct=float(100*(error_after[m]>ERR_EPS).mean()),
             base_median_error_kwh=float(np.median(err[m])),
             corrected_median_error_kwh=float(np.median(error_after[m])),
             base_mean_error_kwh=float(err[m].mean()),
             corrected_mean_error_kwh=float(error_after[m].mean()),
             normalized_true_residual_MAE_cf=float(np.mean(np.abs(actual_corr[m]))),
             residual_regression_MAE_cf=float(np.mean(np.abs(actual_corr[m]-pc[m]))),
             n_nontrivial_true=int(selection.sum()),
             predicted_direction_coverage_pct=float(100*predicting.sum()/max(selection.sum(),1)),
             direction_accuracy_when_decisive_pct=float(100*(np.sign(pc[predicting])==
                                             np.sign(actual_corr[predicting])).mean()) if predicting.any() else np.nan))
    return pd.DataFrame(rows)


def conditional_rows(df,error_after,predicted_r_cf,name,split):
    """One-feature-at-a-time low-RH sign diagnostics; not claims of causal effects."""
    if split!='groupopt':
        # We still save Holdout as descriptive, not for candidate selection.
        pass
    p=df.copy()
    p['post_error_kwh']=error_after
    p['predicted_residual_cf']=predicted_r_cf
    p=p.loc[p.correctable].copy()
    if len(p)==0:return pd.DataFrame()
    p['hour_bin']=pd.cut(p.ts.dt.hour,[-1,5,8,10,12,15,18,23],
        labels=['00-05','06-08','09-10','11-12','13-15','16-18','19-23']).astype(str)
    months=p.ts.dt.month
    p['season']=np.select([months.isin([12,1,2]),months.isin([3,4,5]),
                  months.isin([6,7,8]),months.isin([9,10,11])],
                  ['winter','spring','summer','autumn'],default='other')
    varnames=['wa_cloud','wb_cloud','wb_uv_idx',
      'cloud_diff_wa_minus_wb','rh_diff_wa_minus_wb','temp_diff_wa_minus_wb'] + \
      ['revision_'+c for vs in REVISION_COLS.values() for c in vs]
    tests=['hour_bin','season']
    for col in varnames:
        edges=BINS.get(col)
        if edges is None:
            edges=[-np.inf,0,1,3,6,9,np.inf]
        p[col+'_bin']=pd.cut(p[col],edges).astype(str)
        tests.append(col+'_bin')
    records=[]
    for col in tests:
        for (rh,cat),g in p.groupby(['rh_band',col],sort=True,dropna=False):
            if not len(g):continue
            old=g.e_50.to_numpy(dtype=np.float64)
            new=g.post_error_kwh.to_numpy(dtype=np.float64)
            wk=g.wk.to_numpy(dtype=np.float64)
            ix=g.index.to_numpy()
            pred=g.predicted_residual_cf.to_numpy(dtype=np.float64)
            true=-old/g.capacity_kw.to_numpy(dtype=np.float64)
            sig=np.abs(true)>=0.01
            decisive=np.abs(pred)>=.005
            valid=sig&decisive
            records.append(dict(method=name,split=split,rh_band=rh,feature=col,
                condition=str(cat),n_site_hours=int(len(g)),n_days=int(g.delivery_day.nunique()),
                original_over_pct=float(100*(old>ERR_EPS).mean()),
                after_over_pct=float(100*(new>ERR_EPS).mean()),
                base_median_signed_kwh=float(np.median(old)),
                after_median_signed_kwh=float(np.median(new)),
                original_WMAE_kwh=float(np.dot(wk,np.abs(old))/wk.sum()) if wk.sum()>0 else np.nan,
                corrected_WMAE_kwh=float(np.dot(wk,np.abs(new))/wk.sum()) if wk.sum()>0 else np.nan,
                significant_n=int(sig.sum()),decisive_n=int(valid.sum()),
                direction_acc_pct=float(100*(np.sign(pred[valid])==np.sign(true[valid])).mean()) if valid.any() else np.nan))
    return pd.DataFrame.from_records(records)


def model_importance(model,name,cols):
    if name=='lightgbm_q50':
        return pd.DataFrame({'model':name,'feature':cols,
           'importance_split':model.feature_importances_.astype(float)}).sort_values(
           'importance_split',ascending=False)
    # Ridge after SimpleImputer may append missingness indicators.
    imp=model.named_steps['impute']
    names=imp.get_feature_names_out(cols)
    return pd.DataFrame({'model':name,'feature':names,
       'standardized_coefficient':model.named_steps['ridge'].coef_.astype(float)}).assign(
       absolute_coefficient=lambda d:d.standardized_coefficient.abs()).sort_values(
       'absolute_coefficient',ascending=False)


def run(args):
    root=Path(args.data_dir).expanduser().resolve()
    out=root/'output/newcand_no28/rh_site_residual_ridge_vs_lgb'
    out.mkdir(parents=True,exist_ok=True)
    import baseline_optimizer as opt
    start=time.monotonic()
    site=load_tables(root)
    sites=pd.read_csv(root/'sites.csv')
    if len(sites)!=253 or sites.site_id.duplicated().any():
        raise ValueError('Expected 253 unique plants')
    cap=sites.capacity_kw.to_numpy(dtype=float)
    base_mask=np.load(root/'output/newcand_no28/_shared/no28_group_seed42_nat_feat.npy').astype(bool)
    opt.check_capacity(base_mask,cap)
    initial=opt.initial_mask(sites).astype(bool)
    opt.check_capacity(initial,cap)
    split_data={}
    for split in ('groupopt','holdout'):
        d=site.loc[site.split==split].reset_index(drop=True)
        if len(d)!=BASELINE_N[split]:raise ValueError('No28 split count mismatch')
        mat,times,wk=residual_matrix_for_split(d,d.e_50.to_numpy(dtype=float),sites)
        if len(times)!=BASELINE_NH[split]:raise ValueError('No28 timestamp count mismatch')
        w,lb=matrix_scores(mat,wk,base_mask)
        if abs(w-BASELINE_W[split])>2:
            raise ValueError(f'No28 base W mismatch on {split}: {w:.8f} expected {BASELINE_W[split]:.8f}')
        # exact parity with user's native optimizer tariff weights.
        native_w=opt.time_weights(root,times)
        if not np.allclose(native_w,wk,rtol=1e-9,atol=1e-6):
            raise ValueError('Saved residual wk disagrees with ORIGINAL baseline_optimizer.time_weights')
        split_data[split]=dict(df=d,mat=mat,times=times,wk=wk,W=w,LB=lb)
        log(f'BASELINE {split} W={w:,.2f}, TLB={lb:,.2f} (strict parity passed)')
    x=feature_matrix(site)
    gmask=site.split.eq('groupopt').to_numpy()
    hmask=~gmask
    Xg=x.loc[gmask].reset_index(drop=True)
    Xh=x.loc[hmask].reset_index(drop=True)
    del x,site
    dg=split_data['groupopt']['df']; dh=split_data['holdout']['df']
    if len(Xg)!=len(dg) or len(Xh)!=len(dh):raise ValueError('Feature-split misalignment')
    yg=(-dg.e_50.to_numpy(dtype=np.float64) /
        dg.capacity_kw.to_numpy(dtype=np.float64)).astype('float32')
    sw=(dg.wk.to_numpy(dtype=np.float64) * dg.capacity_kw.to_numpy(dtype=np.float64))
    if not np.isfinite(sw).all() or (sw<0).any() or sw.mean()<=0:
        raise ValueError('Invalid weight vector')
    sw=np.maximum(sw,0)/sw.mean()
    folds=day_folds(dg,args.folds,args.seed)
    # Identifier protects against accidentally reusing predictions under a different cached split.
    checksum=hashlib.sha256(pd.util.hash_pandas_object(
        dg[['site_id','ts','e_50']],index=False).values.tobytes()).hexdigest()
    tag=checksum[:24]
    log('GroupOpt source hash:',tag)
    baseline={s:{'W_krw':q['W'],'TLB_krw':q['LB']} for s,q in split_data.items()}
    all_rows=[]; all_cond=[]; summaries=[]; importances=[]; failures={}
    for name in ['ridge','lightgbm_q50']:
        try:
            log('\n========== EXPERIMENT',name,'==========')
            moof,final=fit_predict_groupopt(name,Xg,yg,sw,folds,out,args,tag)
            mh=np.asarray(final.predict(Xh),dtype='float32')
            if not np.isfinite(mh).all():raise ValueError('Holdout predictions invalid')
            qresult={}
            method_rows=[]; method_cond=[]
            mdl=out/name
            for split,pred in [('groupopt',moof),('holdout',mh)]:
                obj=split_data[split]
                d=obj['df']
                newres,newpred,cor=corrected_residuals(d,pred,args)
                mat,times,wk=residual_matrix_for_split(d,newres,sites)
                wfixed,lb=matrix_scores(mat,wk,base_mask)
                scores=dict(split=split,baseline_W_krw=obj['W'],baseline_TLB_krw=obj['LB'],
                    adjusted_TLB_krw=lb,TLB_delta_krw=lb-obj['LB'],
                    W_fixed_original_mask_krw=wfixed,
                    fixed_mask_W_delta_krw=wfixed-obj['W'],
                    site_hours_corrected=int((np.abs(cor)>1e-9).sum()),
                    baseline_site_WMAE_kwh=float(np.dot(d.wk.to_numpy(float),np.abs(d.e_50.to_numpy(float)))/d.wk.sum()),
                    corrected_site_WMAE_kwh=float(np.dot(d.wk.to_numpy(float),np.abs(newres))/d.wk.sum()))
                if split=='groupopt':
                    # EXACT original native Greedy implementation, original initial mask.
                    log(name,'Running ORIGINAL baseline_optimizer.greedy_best_improvement on OOF residuals')
                    masknew,meta,trace=opt.greedy_best_improvement(mat,wk,sites,initial)
                    masknew=np.asarray(masknew,dtype=bool)
                    opt.check_capacity(masknew,cap)
                    np.save(mdl/'new_optimized_mask.npy',masknew)
                    store_csv(mdl/'composition.csv',pd.DataFrame({'site_id':sites.site_id,
                                               'group':np.where(masknew,'G1','G2')}))
                    store_csv(mdl/'greedy_trace.csv',pd.DataFrame(trace))
                    wopt,_=matrix_scores(mat,wk,masknew)
                    initial_w,_=matrix_scores(mat,wk,initial)
                    scores.update(W_initial_krw=initial_w, W_greedy_reoptimized_krw=wopt,
                        greedy_gain_vs_fixed_krw=wopt-wfixed,
                        greedy_delta_vs_no28_groupopt_krw=wopt-split_data['groupopt']['W'],
                        changed_sites_vs_no28=int((masknew!=base_mask).sum()),
                        G1_capacity_kw=float(cap[masknew].sum()),
                        G2_capacity_kw=float(cap[~masknew].sum()))
                    log(name,f'GroupOpt: baseline={obj["W"]:,.0f} fixed={wfixed:,.0f} '
                             f'Greedy={wopt:,.0f}, TLB={lb:,.0f}')
                else:
                    masknew=np.load(mdl/'new_optimized_mask.npy').astype(bool)
                    opt.check_capacity(masknew,cap)
                    wnew,_=matrix_scores(mat,wk,masknew)
                    scores.update(W_new_group_krw=wnew, delta_vs_no28_holdout_W_krw=wnew-obj['W'],
                                  changed_sites_vs_no28=int((masknew!=base_mask).sum()))
                    log(name,f'Holdout: baseline={obj["W"]:,.0f} fixed={wfixed:,.0f} '
                             f'new group={wnew:,.0f}, TLB={lb:,.0f}')
                qresult[split]=scores
                rows=summarise_sites(d,newres,pred,name,split)
                method_rows.append(rows)
                if not args.skip_conditional_tables:
                    method_cond.append(conditional_rows(d,newres,pred,name,split))
                if not args.no_save_predictions:
                    pr=d[['site_id','ts','rh_band','pred_kwh']].copy()
                    pr['new_pred_kwh']=newpred.astype('float32')
                    pr['baseline_error_kwh']=d.e_50.to_numpy(dtype='float32')
                    pr['corrected_error_kwh']=newres.astype('float32')
                    pr['model_predicted_residual_cf']=np.asarray(pred,dtype='float32')
                    pr['applied_correction_kwh']=cor.astype('float32')
                    pr.to_parquet(mdl/f'{split}_site_predictions.parquet',index=False)
            summaries.append(dict(model=name,**qresult))
            store_json(mdl/'summary.json',qresult)
            importances.append(model_importance(final,name,list(Xg.columns)))
            all_rows.extend(method_rows); all_cond.extend(method_cond)
            log(name,'DONE in',round(time.monotonic()-start,1),'s since start')
        except Exception as err:
            failures[name]=f'{type(err).__name__}: {err}'
            traceback.print_exc()
            log('FAILED',name,failures[name],'— continuing to the next independent method')
    if not summaries:
        store_json(out/'failures.json',failures)
        raise RuntimeError(f'Both correction models failed: {failures}')
    overview=pd.concat(all_rows,ignore_index=True)
    store_csv(out/'01_site_residual_and_RH_metrics.csv',overview)
    if all_cond:store_csv(out/'02_RH_conditionals_by_feature.csv',pd.concat(all_cond,ignore_index=True))
    store_csv(out/'03_feature_importance.csv',pd.concat(importances,ignore_index=True))
    compact=[]
    for d in summaries:
        for split, q in [('groupopt',d['groupopt']),('holdout',d['holdout'])]:
            compact.append(dict(model=d['model'],**q))
    store_csv(out/'00_comparison_groupopt_holdout.csv',pd.DataFrame(compact))
    result=dict(version=RESEARCH_VERSION,split='No28 random delivery-day seed42 60:20:20, historical prior to 2023-06 used ONLY by base No28',
        correction_training='GroupOpt 146 days (678254 OOS site-hours); 5-fold day OOF for GroupOpt grouping',
        correction_target='(actual-No28_prediction)/site_capacity (normalized site error)',
        correction_gate='WB round17 site-level RH in (20,40] or (40,60]; all other site predictions unchanged',
        correction_clip='pred CF correction bound '+str(args.max_correction_cf)+'; multiply shrinkage '+str(args.shrinkage)+ '; clip generation>=0',
        weighting='training wk*(site capacity); wk = SMP*k, k=3 for hours 11..15',
        matrix_missing='missing site-hour residuals 0 (not missing actual energy=0)',
        holdout_usage='diagnostics only, never used to train or tune; already explored in prior PSAT experiments',
        original_greedy='baseline_optimizer.greedy_best_improvement',
        groupopt_cache_hash=tag, feature_names=list(Xg.columns),baseline=baseline,
        results={d['model']:{'groupopt':d['groupopt'],'holdout':d['holdout']} for d in summaries},
        failures=failures,
        wall_seconds=time.monotonic()-start)
    store_json(out/'summary.json',result)
    log('\nCOMPLETE:',out)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--trees',type=int,default=170)
    ap.add_argument('--shrinkage',type=float,default=0.75)
    ap.add_argument('--max-correction-cf',type=float,default=0.08)
    ap.add_argument('--no-save-predictions',action='store_true')
    ap.add_argument('--skip-conditional-tables',action='store_true')
    ap.add_argument('--fresh',action='store_true',help='Ignore existing fitted OOF/final model caches')
    args=ap.parse_args()
    if not 0<args.shrinkage<=1 or not 0<args.max_correction_cf<=1:
        ap.error('Invalid clipping or shrinkage')
    if args.folds<3 or args.threads<1 or args.trees<20:
        ap.error('Invalid folds/threads/trees')
    run(args)


if __name__=='__main__':
    main()
