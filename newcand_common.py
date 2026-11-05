"""Checkpointed OIBC No.28 experiments. Requires existing local codes/ No.28 modules.

All 10 candidates share the exact No.28 51-feature backbone and random seed 42 split.
No hidden evaluation labels are read. Checkpoints are atomic, per-model/round and per-stage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

CANDIDATES = {
  1: ('site_response', 'Site-specific weather-to-power response'),
  2: ('nearby_stations', 'Nearby-station spatial NWP features'),
  3: ('group_median', 'Group-level conditional median regression'),
  4: ('direct_reconciliation', 'Direct group+total forecasts with reconciliation'),
  5: ('analog_ensemble', 'Solar-time analog residual ensemble'),
  6: ('bounded_local_global', 'Constrained local-global group correction'),
  7: ('nwp_shift', 'Test NWP covariate-shift density reweighting'),
  8: ('total_contrast', 'Group weighted total-contrast direct loss'),
  9: ('clearsky_target', 'Clear-sky index target transformation'),
 10: ('weather_moe', 'Weather-regime specialist residual models'),
}
FEATURE_CAND = {1,2,7,9,10}
GROUP_CAND = {3,4,6,8}


def log(*args): print(time.strftime('[%H:%M:%S]'), *args, flush=True)


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf8') as f: json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(temp,path)


def atomic_parquet(df, path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    df.to_parquet(temp,index=False,compression='zstd')
    os.replace(temp,path)


def atomic_pickle(obj,path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f: pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp,path)


def atomic_npy(array,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f: np.save(f,array)
    os.replace(temp,path)


def resume_parquet(path, validate=None):
    if Path(path).is_file():
        try:
            result=pd.read_parquet(path)
            if validate is None or validate(result):
                log('RESUME',path.name);return result
        except Exception as e: log('Unreadable checkpoint, rebuilding',path,e)
    return None


def resume_pickle(path):
    if Path(path).is_file():
        try:
            with Path(path).open('rb') as f: obj=pickle.load(f)
            log('RESUME',path.name);return obj
        except Exception as e: log('Unreadable checkpoint, rebuilding',path,e)
    return None


def fit_variant_model(kind, X, y, sample_weight, seed=42, threads=4, trees=600):
    """No.28 native base parameters; `trees=600` reproduces No.28.
    LightGBM regression_l1; CatBoost MAE; normalized cap*SMP*k weights.
    """
    if kind == 'lgb':
        import lightgbm as lgb
        model=lgb.LGBMRegressor(n_estimators=trees,learning_rate=0.06,
            num_leaves=127,min_child_samples=200,subsample=0.8,
            subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=threads,
            objective='regression_l1',random_state=seed)
    elif kind == 'cat':
        from catboost import CatBoostRegressor
        model=CatBoostRegressor(loss_function='MAE',iterations=trees,
            learning_rate=0.06,depth=8,random_seed=seed,verbose=False,
            thread_count=threads,allow_writing_files=False)
    else: raise ValueError(kind)
    model.fit(X,y,sample_weight=sample_weight)
    return model


def get_imports(root):
    root=Path(root)
    sys.path.insert(0,str(root/'codes'))
    import baseline_validation as val
    import baseline_optimizer as opt
    import nat_feat as nf
    # Import validation_timesplit only for its documentation, not its mutable ROOT.
    return val,opt,nf


@dataclass
class Context:
    root: Path
    out: Path
    shared: Path
    candidate: int
    args: object
    val: object
    opt: object
    nf: object
    base: object
    sites: pd.DataFrame
    fit: pd.DataFrame
    group_actual: pd.DataFrame
    hold_actual: pd.DataFrame
    fa: pd.DataFrame
    fb: pd.DataFrame
    smp: pd.Series
    baseline_group: pd.DataFrame | None = None
    baseline_hold: pd.DataFrame | None = None
    original_features: object = None


def make_base(root,val,nf):
    base=val.load_module(root/'baseline.py','newcand_no28_baseline')
    for c in list(nf.A_COLS)+list(nf.B_COLS):
        base.FEATS.extend([c+'_prev',c+'_next',c+'_daymean'])
    base.FEATS.extend(['cloud_diff']+list(nf.NAT_FEATS))
    assert len(base.FEATS)==51, ('Expected No.28 51 features',len(base.FEATS))
    orig=base.features
    def wrapped(*args,**kwargs):
        d=orig(*args,**kwargs)
        d['cloud_diff']=d['wa_cloud']-d['wb_cloud']
        return d
    base.features=wrapped
    return base


def prepare_context(args,candidate):
    root=Path(args.data_dir).expanduser().resolve()
    val,opt,nf=get_imports(root)
    base=make_base(root,val,nf)
    slug=CANDIDATES[candidate][0]
    out=root/'output'/'newcand_no28'/f'{candidate:02d}_{slug}'
    shared=root/'output'/'newcand_no28'/'_shared'
    for p in (out,shared): p.mkdir(parents=True,exist_ok=True)
    # Prevent stale resume after config/code change.
    manifest={'version':'2026-11-04-no28-v1','candidate':candidate,
              'seed':args.seed,'threads':args.threads,
              'feature_parity':args.feature_parity,'no28_feat_count':51,
              'root':str(root)}
    mpath=out/'manifest.json'
    if mpath.exists():
        old=json.loads(mpath.read_text(encoding='utf8'))
        if old!=manifest:
            raise RuntimeError(f'Checkpoint settings changed in {out}. Use --fresh (moves old checkpoint dir) or preserve settings.')
    atomic_json(mpath,manifest)
    sites=pd.read_csv(root/'sites.csv')[["site_id","current_group","capacity_kw","lat","lon","wa_station","wb_station"]]
    assert len(sites)==253 and sites.site_id.is_unique
    gen=pd.read_parquet(root/'train'/'generation.parquet',columns=['site_id','ts','energy'])
    gen['ts']=val.to_kst(gen.ts)
    if gen.duplicated(['site_id','ts']).any(): raise ValueError('Duplicate generation keys')
    recent=gen.loc[gen.ts>=base.TRAIN_FROM].copy()
    f,g,h,day_split=val.split_602020(recent,args.seed)
    old=gen.loc[gen.ts<base.TRAIN_FROM]
    f=pd.concat([old,f],ignore_index=True)
    del gen,recent,old
    split_shared=shared/f'split_days_seed{args.seed}.csv'
    if split_shared.exists():
        prior=pd.read_csv(split_shared)
        if not prior.equals(day_split): raise ValueError('Existing split differs; remove shared cache')
    else:
        tmp=split_shared.with_name(split_shared.name+'.tmp')
        day_split.to_csv(tmp,index=False);os.replace(tmp,split_shared)
    fa,fb=val.load_weather(root,base,f.ts.min())
    fa,_=nf.add_time_feats(fa,'wa_station',nf.A_COLS)
    fb,_=nf.add_time_feats(fb,'wb_station',nf.B_COLS)
    if args.feature_parity=='submission_safe':
        # Submit_29 removes next at day boundary; apply to BOTH train/validation.
        for c in nf.A_COLS: fa.loc[fa.ts.dt.hour==0,c+'_next']=np.nan
        for c in nf.B_COLS: fb.loc[fb.ts.dt.hour==0,c+'_next']=np.nan
    fa,fb=nf.add_national(fa,fb)
    raw_smp=pd.read_csv(root/'train'/'smp.csv',usecols=['ts','smp'])
    raw_smp['ts']=val.to_kst(raw_smp.ts)
    smp=raw_smp.set_index('ts').smp
    ctx=Context(root,out,shared,candidate,args,val,opt,nf,base,sites,f,g,h,fa,fb,smp)
    return ctx


def features(ctx, actual, r, extra=None):
    df=ctx.base.features(actual,ctx.sites,ctx.fa,ctx.fb,None,r)
    if len(df)!=len(actual):raise ValueError(f'Feature join changed rowcount round{r}: {len(df)} != {len(actual)}')
    if extra is not None:df=extra(df,r)
    return df


def row_weights(ctx, df):
    w=df.ts.map(ctx.smp).to_numpy(np.float64)
    if not np.isfinite(w).all(): raise ValueError('Missing SMP at model-fit timestamps')
    w *= np.where(df.ts.dt.hour.between(11,15),3.,1.)
    w *= df.cap.to_numpy(np.float64)
    if (w<0).any() or w.mean()<=0:raise ValueError('Nonpositive/invalid weights')
    return w/w.mean()


def clear_sky_factor(df):
    # Haurwitz-inspired transmissivity proxy, no test actuals or external radiation.
    # Midnight and low sun clamped to stabilize inverse target transform.
    elev=np.asarray(df.elev,dtype='float64')
    sin_e=np.maximum(np.sin(np.radians(elev)),0.)
    mass=1/(sin_e + 0.50572*np.maximum(elev+6.07995,1e-2)**(-1.6364))
    clearness=1.361*sin_e*(0.75**(np.maximum(mass,0)**0.678))
    return np.clip(clearness,0.08,1.15).astype('float32')


def feature_frame(ctx,actual,r,extras=None):
    d=features(ctx,actual,r,extras)
    return d


def train_predict_four(ctx,kind_key, extra=None, extra_cols=(), target_mode='cf',
                       train_weight_fn=None, train_filter=None, blend_base=None,
                       lgb_trees=600,cat_trees=600):
    """Checkpoint each of LGB/Cat round 10/17; predict combined group+hold per unit.
    When blending, only the novel expert correction is applied to existing No.28.
    """
    cached=resume_parquet(ctx.out/f'{kind_key}_eval_preds.parquet',
        lambda d:set(['site_id','ts','pred_kwh']).issubset(d.columns))
    if cached is not None:return split_predictions(ctx,cached)
    specs=[('lgb',10),('lgb',17),('cat',10),('cat',17)]
    per=[]
    eval_actual=pd.concat([ctx.group_actual,ctx.hold_actual],ignore_index=True)
    feature_names=list(ctx.base.FEATS)+list(extra_cols)
    for model_kind,r in specs:
        key=f'{kind_key}_{model_kind}_r{r}'
        model_path=ctx.out/f'{key}.pkl'
        model=resume_pickle(model_path)
        if model is None:
            log('FIT',key)
            frame=feature_frame(ctx,ctx.fit,r,extra)
            frame=frame.loc[frame.elev>0].copy()
            if train_filter is not None:
                frame=frame.loc[train_filter(frame)].copy()
            X=frame[feature_names].astype('float32')
            y=(frame.energy/frame.cap).clip(0,1.2).to_numpy(dtype='float32')
            if target_mode=='clearsky':
                y=np.clip(y/clear_sky_factor(frame),0,3.).astype('float32')
            w=row_weights(ctx,frame)
            if train_weight_fn is not None:w=train_weight_fn(frame,w,r)
            model=fit_variant_model(model_kind,X,y,w,seed=ctx.args.seed,
                            threads=ctx.args.threads,
                            trees=lgb_trees if model_kind=='lgb' else cat_trees)
            atomic_pickle(model,model_path)
            log('SAVED MODEL',model_path.name)
            del frame,X,y,w
        pred_path=ctx.out/f'{key}_eval.parquet'
        p=resume_parquet(pred_path,lambda d:len(d)==len(eval_actual) and 'pred_kwh' in d)
        if p is None:
            log('PREDICT',key)
            df=feature_frame(ctx,eval_actual,r,extra)
            X=df[feature_names].astype('float32')
            cf=model.predict(X)
            if target_mode=='clearsky':cf=cf*clear_sky_factor(df)
            p=df[['site_id','ts']].copy()
            p['pred_kwh']=np.where(df.elev>0,np.maximum(cf,0),0)*df.cap.to_numpy()
            if not np.isfinite(p.pred_kwh).all():raise ValueError('Nonfinite eval prediction')
            atomic_parquet(p,pred_path)
            del df,X,cf
        p=p.rename(columns={'pred_kwh':key})
        per.append(p)
        del model
    base=per[0]
    for v in per[1:]:base=base.merge(v,on=['site_id','ts'],validate='one_to_one')
    model_keys=[x for x in base if x not in ('site_id','ts')]
    base['pred_kwh']=base[model_keys].mean(axis=1)
    result=base[['site_id','ts','pred_kwh']].copy()
    if blend_base is not None:
        br=blend_base.merge(result,on=['site_id','ts'],suffixes=('_base','_cand'),validate='one_to_one')
        result=br[['site_id','ts']].copy()
        result['pred_kwh']=0.75*br.pred_kwh_base+0.25*br.pred_kwh_cand
    atomic_parquet(result,ctx.out/f'{kind_key}_eval_preds.parquet')
    return split_predictions(ctx,result)


def split_predictions(ctx,pred):
    dp=ctx.val.days_of(pred.ts)
    gd=ctx.val.days_of(ctx.group_actual.ts).drop_duplicates()
    hd=ctx.val.days_of(ctx.hold_actual.ts).drop_duplicates()
    a=pred.loc[dp.isin(gd)].copy()
    b=pred.loc[dp.isin(hd)].copy()
    if len(a)!=len(ctx.group_actual) or len(b)!=len(ctx.hold_actual):
        raise ValueError('Prediction split mismatch')
    return a,b


def no28_predictions(ctx):
    if ctx.baseline_group is not None:return ctx.baseline_group,ctx.baseline_hold
    p=ctx.shared/f'no28_seed{ctx.args.seed}_{ctx.args.feature_parity}.parquet'
    pred=resume_parquet(p, lambda d:'pred_kwh' in d and len(d)==len(ctx.group_actual)+len(ctx.hold_actual))
    if pred is None and ctx.args.feature_parity=='nat_feat' and ctx.args.seed==42:
        # Direct reuse of exact nat_feat.py generated OOS predictions if present.
        folder=ctx.root/'output'/'nat_feat'/'random'
        a=folder/'l1_cap_smpk_eval_pred.parquet'
        b=folder/'cat_l1_cap_smpk_eval_pred.parquet'
        if a.exists() and b.exists():
            pa=pd.read_parquet(a);pb=pd.read_parquet(b)
            m=pa.merge(pb,on=['site_id','ts'],suffixes=('_a','_b'),validate='one_to_one')
            pred=m[['site_id','ts']].copy()
            pred['pred_kwh']=.5*m.pred_kwh_a+.5*m.pred_kwh_b
            if len(pred)!=len(ctx.group_actual)+len(ctx.hold_actual):
                log('Existing No28 prediction count mismatch, training fresh');pred=None
            else:atomic_parquet(pred,p)
    if pred is None:
        saved=ctx.out
        # Baseline training checkpoint is shared and not tied to a specific candidate.
        ctx.out=ctx.shared/f'no28_training_seed{ctx.args.seed}_{ctx.args.feature_parity}'
        ctx.out.mkdir(parents=True,exist_ok=True)
        try:ga,ha=train_predict_four(ctx,'no28',lgb_trees=600,cat_trees=600)
        finally:ctx.out=saved
        pred=pd.concat([ga,ha],ignore_index=True)
        atomic_parquet(pred,p)
    ctx.baseline_group,ctx.baseline_hold=split_predictions(ctx,pred)
    return ctx.baseline_group,ctx.baseline_hold


def baseline_assignment(ctx):
    p=ctx.shared/f'no28_group_seed{ctx.args.seed}_{ctx.args.feature_parity}.npy'
    if p.exists():
        mask=np.load(p)
        if len(mask)==len(ctx.sites):
            log('RESUME No28 optimized group');return mask
    pg,_=no28_predictions(ctx)
    mat,times,_=ctx.val.residual_matrix(ctx.group_actual,pg,ctx.sites.site_id,'NO28_GROUP_OPT')
    init=ctx.opt.initial_mask(ctx.sites)
    mask,meta,trace=ctx.opt.greedy_best_improvement(mat,ctx.opt.time_weights(ctx.root,times),ctx.sites,init)
    ctx.opt.check_capacity(mask,ctx.sites.capacity_kw.to_numpy())
    atomic_npy(mask,p)
    trace.to_csv(ctx.shared/f'no28_group_seed{ctx.args.seed}_{ctx.args.feature_parity}_trace.csv',index=False)
    return mask


def evaluate(ctx,pred_group,pred_hold,mask=None,group_model=False,notes=None):
    """Full checkpoint for stage GROUP_OPT+HOLDOUT using existing exact scoring."""
    path=ctx.out/'summary.json'
    if path.is_file():
        log('RESUME COMPLETED',path);return json.loads(path.read_text(encoding='utf8'))
    if group_model:
        if mask is None:mask=baseline_assignment(ctx)
        bg,bh=no28_predictions(ctx)
        bg_mat,bg_times,_=ctx.val.residual_matrix(ctx.group_actual,bg,ctx.sites.site_id,'BASELINE_OPT')
        bh_mat,bh_times,_=ctx.val.residual_matrix(ctx.hold_actual,bh,ctx.sites.site_id,'BASELINE_HOLD')
        bg_s=ctx.val.score_group(ctx.opt,ctx.root,bg_mat,bg_times,mask,mask)
        bh_s=ctx.val.score_group(ctx.opt,ctx.root,bh_mat,bh_times,mask,mask)
        # pred_group/hold already group-level: columns ts,group,pred_kwh
        def group_penalty(pred,actual):
            ass=pd.Series(np.where(mask,'G1','G2'),index=ctx.sites.site_id)
            true=actual[['site_id','ts','energy']].copy()
            true['group']=true.site_id.map(ass)
            tg=true.groupby(['group','ts'],observed=True,as_index=False).energy.sum()
            pp=pred[['group','ts','pred_kwh']]
            merged=tg.merge(pp,on=['group','ts'],validate='one_to_one')
            if len(merged)!=len(tg): raise ValueError('Missing group predictions')
            ww=ctx.opt.time_weights(ctx.root,pd.DatetimeIndex(merged.ts))
            return float(ctx.opt.PENALTY_RATE*np.dot(ww,np.abs(merged.pred_kwh-merged.energy)))
        sg=group_penalty(pred_group,ctx.group_actual)
        sh=group_penalty(pred_hold,ctx.hold_actual)
        report={'experiment':CANDIDATES[ctx.candidate][0],
           'holdout_W_krw':sh,'group_opt_W_krw':sg,
           'no28_same_group_holdout_W_krw':bh_s['W_optimized_krw'],
           'no28_group_opt_W_krw':bg_s['W_optimized_krw'],
           'no28_holdout_LB_krw':bh_s['W_lower_bound_krw'],
           'delta_vs_no28_krw':sh-bh_s['W_optimized_krw'],
           'notes':notes or {},'type':'group_predictor_fixed_no28_groups'}
        atomic_parquet(pred_group,ctx.out/'group_opt_group_predictions.parquet')
        atomic_parquet(pred_hold,ctx.out/'holdout_group_predictions.parquet')
    else:
        atomic_parquet(pred_group,ctx.out/'group_opt_site_predictions.parquet')
        atomic_parquet(pred_hold,ctx.out/'holdout_site_predictions.parquet')
        matrix,times,_=ctx.val.residual_matrix(ctx.group_actual,pred_group,ctx.sites.site_id,'CAND_GROUP_OPT')
        matrix_h,th,_=ctx.val.residual_matrix(ctx.hold_actual,pred_hold,ctx.sites.site_id,'CAND_HOLDOUT')
        mask_path=ctx.out/'optimized_mask.npy'
        if mask_path.exists():
            optimized=np.load(mask_path);log('RESUME optimized group')
        else:
            initial=ctx.opt.initial_mask(ctx.sites)
            optimized,meta,trace=ctx.opt.greedy_best_improvement(matrix,ctx.opt.time_weights(ctx.root,times),ctx.sites,initial)
            ctx.opt.check_capacity(optimized,ctx.sites.capacity_kw.to_numpy())
            atomic_npy(optimized,mask_path)
            trace.to_csv(ctx.out/'greedy_trace.csv',index=False)
        initial=ctx.opt.initial_mask(ctx.sites)
        sg=ctx.val.score_group(ctx.opt,ctx.root,matrix,times,initial,optimized)
        sh=ctx.val.score_group(ctx.opt,ctx.root,matrix_h,th,initial,optimized)
        m=ctx.val.model_metrics(ctx.root,ctx.hold_actual,pred_hold)
        atomic_parquet(pd.DataFrame({'site_id':ctx.sites.site_id,
                       'group':np.where(optimized,'G1','G2')}),ctx.out/'composition.parquet')
        report={'experiment':CANDIDATES[ctx.candidate][0],
            'holdout_W_krw':sh['W_optimized_krw'],
            'group_opt_W_krw':sg['W_optimized_krw'],
            'holdout_LB_krw':sh['W_lower_bound_krw'],
            'weighted_MAE':m['weighted_MAE_kwh'],
            'initial_W_krw':sh['W_initial_krw'],
            'notes':notes or {},'type':'site_predictor_reoptimized_groups'}
        _,bh=no28_predictions(ctx)
        mask0=baseline_assignment(ctx)
        bh_mat,bh_times,_=ctx.val.residual_matrix(ctx.hold_actual,bh,ctx.sites.site_id,'NO28_HOLD')
        bs=ctx.val.score_group(ctx.opt,ctx.root,bh_mat,bh_times,mask0,mask0)
        report['no28_holdout_W_krw']=bs['W_optimized_krw']
        report['delta_vs_no28_krw']=report['holdout_W_krw']-report['no28_holdout_W_krw']
    atomic_json(path,report)
    log('FINISHED',ctx.candidate,'W',round(report['holdout_W_krw']),
        'delta',round(report.get('delta_vs_no28_krw',0)))
    return report


def base_aggregation(ctx, mask):
    """Group-wise time predictors, using only No.28 site OOS forecasts (calibration/hold)."""
    pg,ph=no28_predictions(ctx)
    ids=ctx.sites.site_id.to_numpy()
    group_of=pd.Series(np.where(mask,'G1','G2'),index=ids)
    def agg(actual,pred):
        d=actual[['site_id','ts','energy']].merge(pred,on=['site_id','ts'],validate='one_to_one')
        d['group']=d.site_id.map(group_of)
        d['cap']=d.site_id.map(ctx.sites.set_index('site_id').capacity_kw)
        a=d.groupby(['group','ts'],observed=True,as_index=False).agg(
           actual=('energy','sum'),no28=('pred_kwh','sum'),capacity=('cap','sum'))
        a['hour']=a.ts.dt.hour
        a['doy']=a.ts.dt.dayofyear
        a['hour_sin']=np.sin(a.hour*2*np.pi/24)
        a['hour_cos']=np.cos(a.hour*2*np.pi/24)
        a['doy_sin']=np.sin(a.doy*2*np.pi/365.25)
        a['doy_cos']=np.cos(a.doy*2*np.pi/365.25)
        a['group_id']=(a.group=='G1').astype(int)
        a['pred_cf']=a.no28/a.capacity.clip(lower=1)
        return a
    return agg(ctx.group_actual,pg),agg(ctx.hold_actual,ph)


def grouped_columns():
    return ['no28','capacity','hour_sin','hour_cos','doy_sin','doy_cos','group_id','pred_cf']


def grouped_weights(ctx,d):
    smp=d.ts.map(ctx.smp)
    if smp.isna().any():raise ValueError('Missing SMP for group-calibrator')
    ww=smp.to_numpy(np.float64)*np.where(d.hour.between(11,15),3.,1.)
    return ww/ww.mean()


def group_prediction_frames(ctx,cal,hold,calc,holdc):
    pg=cal[['ts','group']].copy();pg['pred_kwh']=np.clip(np.asarray(calc),0,None)
    ph=hold[['ts','group']].copy();ph['pred_kwh']=np.clip(np.asarray(holdc),0,None)
    return pg,ph


def load_or_fit_group_model(ctx,key,cal,fitfunc):
    p=ctx.out/f'{key}_group_model.pkl'
    model=resume_pickle(p)
    if model is None:
        model=fitfunc(cal)
        atomic_pickle(model,p)
    return model


def run_candidate(candidate,args):
    if args.fresh:
        root=Path(args.data_dir).expanduser().resolve()
        out=root/'output'/'newcand_no28'/f'{candidate:02d}_{CANDIDATES[candidate][0]}'
        if out.exists():
            import shutil
            backup=out.with_name(out.name+time.strftime('_backup_%Y%m%d_%H%M%S'))
            shutil.move(out,backup);log('BACKUP previous checkpoints',backup)
    ctx=prepare_context(args,candidate)
    if (ctx.out/'summary.json').exists():
        return json.loads((ctx.out/'summary.json').read_text())
    import newcand_experiments as exp
    return exp.dispatch(ctx)


def cli(candidate):
    p=argparse.ArgumentParser(description=f'No28 independent candidate {candidate}: {CANDIDATES[candidate][1]}')
    p.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--fresh',action='store_true',help='Move old candidate checkpoint directory aside; retrain')
    p.add_argument('--feature-parity',choices=['nat_feat','submission_safe'],default='nat_feat',
       help='nat_feat=exact No28 validation; submission_safe=remove 00:00 next both train and eval')
    args=p.parse_args()
    try:run_candidate(candidate,args)
    except Exception:
        log('FAILED candidate',candidate,'(re-run same command to resume)')
        traceback.print_exc();raise
