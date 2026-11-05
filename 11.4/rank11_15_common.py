#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Five independent No.28 candidates ranked 11-15 in the supplied 39-idea list.
No contest data are uploaded; all computation is local.
Requires existing codes/newcand_common.py and standard No.28 local caches.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import newcand_common as nc

SPEC = {
    11: (47, 'seasonal_portfolio', 'season-conditioned robust residual portfolio'),
    12: (48, 'ramp_aware', 'ramp-aware weighted L1 training'),
    13: (49, 'nwp_phase', 'forecast temporal-phase features ±2 hours'),
    14: (50, 'clip_curtail', 'clipping and curtailment proxy separation'),
    15: (51, 'group_error_dist', 'group-specific conditional error quantile distribution'),
}
for _rank, (_cid, _slug, _desc) in SPEC.items():
    nc.CANDIDATES[_cid] = (_slug, f'Rank {_rank}: {_desc}')

EXPECTED = 63_669_335.6497201


def log(msg):
    nc.log('[RANK11-15]',msg)


def safe_score(ctx, pg, ph, mask, title):
    m, t, missing_g = ctx.val.residual_matrix(ctx.group_actual, pg, ctx.sites.site_id, title+'_GROUP')
    mh, th, missing_h = ctx.val.residual_matrix(ctx.hold_actual, ph, ctx.sites.site_id, title+'_HOLD')
    initial = ctx.opt.initial_mask(ctx.sites)
    sg = ctx.val.score_group(ctx.opt, ctx.root, m, t, initial, mask)
    sh = ctx.val.score_group(ctx.opt, ctx.root, mh, th, initial, mask)
    metrics = ctx.val.model_metrics(ctx.root, ctx.hold_actual, ph)
    return {
        'group_opt_W_krw':sg['W_optimized_krw'],
        'holdout_W_krw':sh['W_optimized_krw'],
        'initial_W_krw':sh['W_initial_krw'],
        'holdout_LB_krw':sh['W_lower_bound_krw'],
        'weighted_MAE':metrics['weighted_MAE_kwh'],
        'groupopt_missing_site_hours':missing_g,
        'holdout_missing_site_hours':missing_h,
    }


def reference(ctx):
    base_g, base_h = nc.no28_predictions(ctx)
    mask = nc.baseline_assignment(ctx)
    ref = safe_score(ctx,base_g,base_h,mask,'REF_NO28')
    diff = abs(ref['holdout_W_krw'] - EXPECTED)
    log(f'No.28 cache reference W={ref["holdout_W_krw"]:,.2f}, diff={diff:.2f}')
    if ctx.args.seed == 42 and ctx.args.feature_parity == 'nat_feat' and diff > 5.0:
        raise ValueError('No.28 OOS reference mismatch. Refuse to compare unlike prediction caches.')
    return base_g,base_h,mask,ref


def persist(ctx,rank,rep,notes,base_ref):
    row={
       'rank':rank,'candidate':SPEC[rank][1],
       'experiment':SPEC[rank][2],
       'type':notes.get('type',''),
       'holdout_W_krw':rep['holdout_W_krw'],
       'group_opt_W_krw':rep['group_opt_W_krw'],
       'initial_W_krw':rep['initial_W_krw'],
       'holdout_LB_krw':rep['holdout_LB_krw'],
       'weighted_MAE':rep['weighted_MAE'],
       'no28_holdout_W_krw':base_ref['holdout_W_krw'],
       'no28_group_opt_W_krw':base_ref['group_opt_W_krw'],
       'delta_vs_no28_krw':rep['holdout_W_krw']-base_ref['holdout_W_krw'],
       'missing_site_hours':{
          'group_opt':rep['groupopt_missing_site_hours'],
          'holdout':rep['holdout_missing_site_hours'],
       },
       'details':notes,
    }
    nc.atomic_json(ctx.out/'summary.json',row)
    log(f'FINISHED rank={rank} holdout W={row["holdout_W_krw"]:,.2f} '
        f'delta={row["delta_vs_no28_krw"]:+,.2f} KRW')
    return row


def season_index(times):
    month=np.asarray(times.month)
    return ((month%12)//3).astype(np.int8)  # 0 winter 1 spring 2 summer 3 autumn


def season_loss(mat,weights,mask,times,season_lam=.35):
    """No.28 W + lambda * equal-season mean per-hour W, scaled to W units.
    Only group-optimization residuals; no holdout involvement.
    """
    e1=mat[:,mask].sum(axis=1)
    e2=mat[:,~mask].sum(axis=1)
    per=0.1*weights*(np.abs(e1)+np.abs(e2))
    base=per.sum()
    si=season_index(times)
    parts=np.array([per[si==i].mean() if np.any(si==i) else 0.0 for i in range(4)])
    robust=float(len(per)*parts.mean())
    return float((1-season_lam)*base+season_lam*robust)


def rank11(ctx):
    """Grouping ONLY: start from the fixed No.28 group, seasonal-robust objective.
    Group assignment must be static on test; no per-season test group switching.
    """
    base_g,base_h,original,ref=reference(ctx)
    mat,times,_=ctx.val.residual_matrix(ctx.group_actual,base_g,ctx.sites.site_id,'R11_GROUP_OPT')
    ww=ctx.opt.time_weights(ctx.root,times)
    cap=ctx.sites.capacity_kw.to_numpy(dtype='float64')
    mask=original.astype(bool).copy()
    best=season_loss(mat,ww,mask,times)
    log(f'Rank11 starting seasonal objective={best:,.2f}')
    # Vectorized candidates: move and swap among the most season-imbalanced sites.
    # Keep runtime bounded; all proposals evaluated on GROUP_OPT only.
    trace=[]
    for iteration in range(5):
        group_idx=np.flatnonzero(mask)
        other_idx=np.flatnonzero(~mask)
        candidates=[]
        c1=cap[mask].sum();c2=cap[~mask].sum()
        # All single-site moves; capacity checks explicit.
        for idx in range(len(mask)):
            if mask[idx] and c1-cap[idx] >= 20000:
                candidates.append(('move',idx,-1))
            elif (not mask[idx]) and c2-cap[idx] >= 20000:
                candidates.append(('move',idx,-1))
        # First-pass swap shortlisting from seasonal site error signatures.
        seasons=season_index(times)
        signature=np.vstack([mat[seasons==s].mean(axis=0) if (seasons==s).any() else np.zeros(mat.shape[1]) for s in range(4)])
        center=signature-signature.mean(axis=1,keepdims=True)
        strength=np.linalg.norm(center,axis=0)
        picks1=group_idx[np.argsort(strength[group_idx])[-25:]]
        picks2=other_idx[np.argsort(strength[other_idx])[-25:]]
        for a in picks1:
            for b in picks2:
                if c1-cap[a]+cap[b] >= 20000 and c2-cap[b]+cap[a]>=20000:
                    candidates.append(('swap',int(a),int(b)))
        # Efficient loss from delta of baseline group sums (avoids slicing 253-site matrices per move).
        e1=mat[:,mask].sum(axis=1)
        total=mat.sum(axis=1)
        season=season_index(times)
        season_cnt=np.bincount(season,minlength=4).astype(float)
        coeff=(1-.35)+.35*(len(times)/4)/np.maximum(season_cnt,1)
        eff=0.1*ww*coeff[season]
        best_action=None
        new_best=best
        for how,a,b in candidates:
            if how=='move':
                change= -mat[:,a] if mask[a] else mat[:,a]
            else:
                change= -mat[:,a]+mat[:,b]
            x=e1+change
            obj=float(np.dot(eff,np.abs(x)+np.abs(total-x)))
            if obj < new_best-1e-5:
                new_best=obj;best_action=(how,a,b)
        if best_action is None:
            log('Rank11 local search converged; no further seasonal objective gain')
            break
        how,a,b=best_action
        mask[a]=~mask[a]
        if how=='swap':mask[b]=~mask[b]
        ctx.opt.check_capacity(mask,cap)
        trace.append({'step':iteration+1,'type':how,'site_a':ctx.sites.site_id.iloc[a],
                      'site_b':str(ctx.sites.site_id.iloc[b]) if b>=0 else '',
                      'seasonal_objective':new_best})
        best=new_best
        log(f'Rank11 step {iteration+1}: {how}, seasonal objective={best:,.2f}')
    report=safe_score(ctx,base_g,base_h,mask,'R11_PORTFOLIO')
    nc.atomic_npy(mask,ctx.out/'optimized_mask.npy')
    pd.DataFrame({'site_id':ctx.sites.site_id,'group':np.where(mask,'G1','G2')}).to_csv(
        ctx.out/'composition.csv',index=False)
    pd.DataFrame(trace,columns=['step','type','site_a','site_b','seasonal_objective']).to_csv(
        ctx.out/'search_trace.csv',index=False)
    # Explicit seasonal diagnostic with same scoring for each season.
    for split,actual,pred in [('group_opt',ctx.group_actual,base_g),('holdout',ctx.hold_actual,base_h)]:
        m,t,_=ctx.val.residual_matrix(actual,pred,ctx.sites.site_id,'R11_SEASON_'+split)
        w=ctx.opt.time_weights(ctx.root,t)
        seasons=season_index(t)
        rows=[]
        for s,name in enumerate(['winter','spring','summer','autumn']):
            ix=seasons==s
            if not ix.any():continue
            for label,mm in [('no28',original),('seasonal',mask)]:
                e1=m[ix][:,mm].sum(axis=1);e2=m[ix][:,~mm].sum(axis=1)
                rows.append({'season':name,'group':label,'n_hours':int(ix.sum()),
                             'W_krw':float(0.1*np.dot(w[ix],np.abs(e1)+np.abs(e2)))})
        pd.DataFrame(rows).to_csv(ctx.out/(split+'_season_metrics.csv'),index=False)
    return persist(ctx,11,report,{
       'type':'group_optimization_only','lambda_equal_season':.35,
       'max_search_rounds':5,'accepted_steps':len(trace),
       'fixed_single_test_composition':True,'model_retrained':False,
       'comment':'Season-specific objective; a SINGLE group mask is used for all seasons.'
    },ref)


def rank12(ctx):
    """Ramp-weighted L1: sole alteration is training row weights.
    Training labels only, and ONLY on the No28 training partition.
    """
    _,_,orig,ref=reference(ctx)
    def ramp_weights(frame,original_w,r):
        # Label-derived weights are legal for TRAINING rows; never used in prediction features.
        x=frame[['site_id','ts','energy','cap']].copy()
        x=x.sort_values(['site_id','ts'])
        prior=x.groupby('site_id',sort=False).energy.shift(1)
        prior_ts=x.groupby('site_id',sort=False).ts.shift(1)
        consecutive=(x.ts-prior_ts).eq(pd.Timedelta(hours=1))
        ramp=((x.energy-prior).abs()/x.cap.clip(lower=1)).where(consecutive,0).fillna(0)
        # 0.2 capacity in 1h saturates the supplemental ramp emphasis.
        ampl=1.+1.5*np.minimum(ramp.to_numpy(dtype=float)/.20,1.)
        # Restore original frame order after computing per-site adjacency.
        out=pd.Series(ampl,index=x.index).reindex(frame.index).to_numpy()
        weights=np.asarray(original_w,dtype=float)*out
        log(f'Rank12 r{r}: ramp weighted train fraction high={(out>1.5).mean():.2%}')
        return weights/weights.mean()
    pg,ph=nc.train_predict_four(ctx,'ramp_weighted',train_weight_fn=ramp_weights)
    # Standard candidate evaluation reoptimizes a GROUP_OPT-only mask exactly as other site-predictor tests.
    return nc.evaluate(ctx,pg,ph,notes={
       'rank':12,'change':'Train row weights *= 1+1.5*min(|deltaCF_prev_hour|/0.20,1)',
       'original_weight':'capacity*SMP*k','no28_validation':'same seed 42 60:20:20',
       'no_test_labels':True,'reference_no28_holdout_W_krw':ref['holdout_W_krw'],
    })


def add_pm2(df,station,columns):
    """Use only same forecast issue round and same right-closed delivery day."""
    out=df.sort_values([station,'round','ts']).copy()
    day=(out.ts-pd.Timedelta(hours=1)).dt.normalize()
    groups=[out[station],out['round'],day]
    for c in columns:
        g=out.groupby(groups,sort=False,observed=True)[c]
        left=g.shift(2); right=g.shift(-2)
        # Enforce true 2h adjacency, not just row position.
        tprev=out.groupby(groups,sort=False,observed=True).ts.shift(2)
        tnext=out.groupby(groups,sort=False,observed=True).ts.shift(-2)
        out['phase_'+c+'_m2']=left.where((out.ts-tprev).eq(pd.Timedelta(hours=2))).astype('float32')
        out['phase_'+c+'_p2']=right.where((tnext-out.ts).eq(pd.Timedelta(hours=2))).astype('float32')
    return out


def rank13(ctx):
    """Phase-aware forecast-only features (±2h) added to unchanged 51-feature No28.
    This is a proxy allowing model to learn timing shifts, NOT true obs-aligned phase correction.
    """
    _,_,original,ref=reference(ctx)
    log('Rank13 generating station/round/day safe ±2h forecast features')
    ctx.fa=add_pm2(ctx.fa,'wa_station',['wa_cloud'])
    ctx.fb=add_pm2(ctx.fb,'wb_station',['wb_cloud','wb_uv_idx'])
    cols=[f'phase_{c}_{direction}' for c in ['wa_cloud','wb_cloud','wb_uv_idx'] for direction in ['m2','p2']]
    pg,ph=nc.train_predict_four(ctx,'phase_2h',extra_cols=cols)
    return nc.evaluate(ctx,pg,ph,notes={
        'rank':13,'change':'Six new same-round/same-delivery-day ±2h forecast values',
        'features':cols,'note':'Forecast-only phase-shift proxy; no real-time observed weather.',
        'reference_no28_holdout_W_krw':ref['holdout_W_krw'],
    })


def rank14(ctx):
    """Exploratory *proxy* separation; not ground-truth curtailment detection.
    Learn fixed plant-specific empirical clipping caps only from TRAIN labels.
    Clear-weather low output frequency is treated as an unreliable curtailment proxy,
    with a conservative bounded correction on out-of-sample forecasts.
    """
    bg,bh,mask,ref=reference(ctx)
    sites=ctx.sites.set_index('site_id')
    fit=ctx.fit[['site_id','ts','energy']].copy()
    fit['cap']=fit.site_id.map(sites.capacity_kw)
    fit['cf']=fit.energy/fit.cap
    day=fit.loc[fit.ts.dt.hour.between(10,16) & fit.cf.gt(0.02)].copy()
    capstats=day.groupby('site_id').cf.agg(n='count',q995=lambda v:float(v.quantile(.995)),
                                          q99=lambda v:float(v.quantile(.99)))
    capstats['ceiling_cf']=np.where(capstats.n.ge(70),
                      np.clip(capstats.q995*1.02,.2,1.2),1.2)
    # Forecast-based high-sun low-production proxy. Training labels ONLY.
    f=nc.features(ctx,ctx.fit,17)
    sunny=(f.elev.to_numpy(dtype=float)>=25)&(f.wb_uv_idx.to_numpy(dtype=float)>=6)&\
          (f.wa_cloud.to_numpy(dtype=float)<=30)&(f.wb_cloud.to_numpy(dtype=float)<=30)
    f['is_sunny']=sunny
    f['is_low']=((f.energy/f.cap)<.12)&sunny
    sunny_stats=f.groupby('site_id').agg(n_sunny=('is_sunny','sum'),n_low=('is_low','sum'))
    capstats=capstats.join(sunny_stats,how='outer').fillna({'n_sunny':0,'n_low':0,'ceiling_cf':1.2})
    # Explicitly conservative proxy: max 5% reduction in strong-sun regime.
    raw_rate=np.where(capstats.n_sunny.to_numpy()>=40,
                      capstats.n_low.to_numpy()/np.maximum(capstats.n_sunny.to_numpy(),1),0.)
    capstats['derate_rate']=np.clip(raw_rate-.10,0,.05)
    capstats[['n','q995','q99','ceiling_cf','n_sunny','n_low','derate_rate']].to_csv(
         ctx.out/'train_only_proxy_statistics.csv')
    log(f'Rank14 clipping caps < 1.0: {(capstats.ceiling_cf<1).sum()} sites; '
        f'nonzero curtailment proxy: {(capstats.derate_rate>0).sum()} sites')
    def apply(actual,pred):
        d=pred[['site_id','ts','pred_kwh']].copy()
        cap=d.site_id.map(sites.capacity_kw).to_numpy(dtype=float)
        ceiling=d.site_id.map(capstats.ceiling_cf).fillna(1.2).to_numpy(dtype=float)
        derate=d.site_id.map(capstats.derate_rate).fillna(0).to_numpy(dtype=float)
        g=nc.features(ctx,actual,17)
        if not np.array_equal(g[['site_id','ts']].to_numpy(),d[['site_id','ts']].to_numpy()):
            # Do not silently use wrong feature keys/order.
            meta=g[['site_id','ts','elev','wa_cloud','wb_cloud','wb_uv_idx']]
            d=d.merge(meta,on=['site_id','ts'],validate='one_to_one',sort=False)
        else:
            d['elev']=g.elev.to_numpy();d['wa_cloud']=g.wa_cloud.to_numpy()
            d['wb_cloud']=g.wb_cloud.to_numpy();d['wb_uv_idx']=g.wb_uv_idx.to_numpy()
        clear=(d.elev.to_numpy(dtype=float)>=25)&(d.wb_uv_idx.to_numpy(dtype=float)>=6)&\
              (d.wa_cloud.to_numpy(dtype=float)<=30)&(d.wb_cloud.to_numpy(dtype=float)<=30)
        y0=d.pred_kwh.to_numpy(dtype=float)
        # Change only clipping + suspected derating regime; no other model change.
        corrected=np.minimum(y0,ceiling*cap)
        corrected=np.where(clear,corrected*(1-derate),corrected)
        p=d[['site_id','ts']].copy();p['pred_kwh']=np.clip(corrected,0,None)
        return p
    pg=apply(ctx.group_actual,bg)
    ph=apply(ctx.hold_actual,bh)
    nc.atomic_parquet(pg,ctx.out/'group_opt_site_predictions.parquet')
    nc.atomic_parquet(ph,ctx.out/'holdout_site_predictions.parquet')
    rep=safe_score(ctx,pg,ph,mask,'R14_CLIP_CURTAIL')
    return persist(ctx,14,rep,{
       'type':'site_postprocess_fixed_no28_groups',
       'model_retrained':False,
       'clipping_cap':'TRAIN site daytime q99.5*1.02, only if >=70 points',
       'curtailment':'UNVERIFIED proxy: clear NWP + site low-output frequency; at most 5% reduction',
       'ground_truth_curtailment_available':False,
       'caution':'Do NOT interpret suspected curtailment as measured curtailment.',
    },ref)


def rank15(ctx):
    """Group-specific conditional signed-error quantile distributions (q.1/.5/.9).
    Fit on group_opt OOS outcomes only; holdout never enters fit or selection.
    """
    from lightgbm import LGBMRegressor
    bg,bh,mask,ref=reference(ctx)
    cal,hold=nc.base_aggregation(ctx,mask)
    cols=nc.grouped_columns()
    qs=(.1,.5,.9)
    fits={}
    for group in ['G1','G2']:
        c=cal.loc[cal.group.eq(group)].copy()
        ww=nc.grouped_weights(ctx,c)
        yy=((c.actual-c.no28)/c.capacity.clip(lower=1)).to_numpy(dtype='float32')
        X=c[cols].astype('float32')
        for q in qs:
            key=f'g{group}_quantile_{int(q*100)}'
            model=nc.resume_pickle(ctx.out/(key+'.pkl'))
            if model is None:
                log(f'Rank15 fit group={group} quantile={q}')
                model=LGBMRegressor(objective='quantile',alpha=q,n_estimators=260,
                    learning_rate=.035,num_leaves=7,min_child_samples=90,
                    n_jobs=ctx.args.threads,random_state=ctx.args.seed,verbosity=-1)
                model.fit(X,yy,sample_weight=ww)
                nc.atomic_pickle(model,ctx.out/(key+'.pkl'))
            fits[(group,q)]=model
    def apply(d,split):
        dd=d.copy()
        for q in qs:
            dd[f'q{int(q*100)}']=np.nan
        for group in ['G1','G2']:
            maskg=d.group.eq(group)
            x=d.loc[maskg,cols].astype('float32')
            for q in qs:
                dd.loc[maskg,f'q{int(q*100)}']=fits[(group,q)].predict(x)
        a=dd[['q10','q50','q90']].to_numpy(dtype=float)
        a.sort(axis=1)    # quantile crossing guard
        qlo,qmed,qhi=a[:,0],a[:,1],a[:,2]
        capacity=dd.capacity.to_numpy(dtype=float)
        orig=dd.no28.to_numpy(dtype=float)
        # Distribution-dependent shrinkage: uncertain corrections are attenuated.
        spread=(qhi-qlo)*capacity
        gate=1/(1+spread/(.12*capacity.clip(min=1)))
        corr=np.clip(qmed*capacity, -np.maximum(.12*orig,.02*capacity),
                        np.maximum(.12*orig,.02*capacity))
        dd['q10_residual_cf']=qlo
        dd['q50_residual_cf']=qmed
        dd['q90_residual_cf']=qhi
        dd['uncertainty_gate']=gate
        dd['pred_kwh']=(orig+.75*gate*corr).clip(min=0)
        nc.atomic_parquet(dd[['ts','group','no28','actual','pred_kwh','q10_residual_cf',
                              'q50_residual_cf','q90_residual_cf','uncertainty_gate']],
                              ctx.out/(split+'_distribution.parquet'))
        return dd[['ts','group','pred_kwh']].copy()
    pg=apply(cal,'group_opt')
    ph=apply(hold,'holdout')
    rep=safe_group_score(ctx,cal,hold,pg,ph,mask,'R15_GROUPDIST')
    nc.atomic_parquet(pg,ctx.out/'group_opt_group_predictions.parquet')
    nc.atomic_parquet(ph,ctx.out/'holdout_group_predictions.parquet')
    return persist(ctx,15,rep,{
       'type':'group_postprocess_fixed_no28_groups',
       'fit':'GROUP_OPT OOS group signed error quantiles q10/q50/q90',
       'correction':'0.75 * uncertainty_gate * capped conditional median residual',
       'groupopt_metric_warning':'IN-SAMPLE calibrated; only independent Holdout is reliable',
       'holdout_used_for_model_selection':False,
       'model_retrained_no28':False,
    },ref)


def safe_group_score(ctx,cal,hold,pg,ph,mask,title):
    """Evaluate group predictions without inventing missing data or changing group."""
    def get(actual,pred):
        tmp=actual[['ts','group','actual','capacity','no28']].merge(
            pred,on=['ts','group'],validate='one_to_one',how='left')
        if len(tmp)!=len(actual) or tmp.pred_kwh.isna().any():
            raise ValueError('Incomplete group predictions')
        ww=tmp.ts.map(ctx.smp).to_numpy(dtype=float)*np.where(tmp.ts.dt.hour.between(11,15),3.,1.)
        return float(.1*np.dot(ww,np.abs(tmp.pred_kwh.to_numpy()-tmp.actual.to_numpy())))
    # Exact model original site-hour WMAE unchanged for group postprocess.
    bg,bh=nc.no28_predictions(ctx)
    basew=ctx.val.model_metrics(ctx.root,ctx.hold_actual,bh)['weighted_MAE_kwh']
    sc=get(cal,pg);sh=get(hold,ph)
    # Recompute LB from corrected G1/G2 forecasts. No candidate InitialW
    # exists because this experiment only forecasts fixed groups, not sites.
    hh=hold[['ts','group','actual']].merge(ph,on=['ts','group'],how='left',validate='one_to_one')
    hh['err']=hh.pred_kwh-hh.actual
    tt=hh.groupby('ts',observed=True,as_index=False).err.sum()
    w=tt.ts.map(ctx.smp).to_numpy(dtype=float)*np.where(tt.ts.dt.hour.between(11,15),3.,1.)
    lb=float(.1*np.dot(w,np.abs(tt.err.to_numpy(dtype=float))))
    return {
       'group_opt_W_krw':sc,'holdout_W_krw':sh,
       'initial_W_krw':None,
       'holdout_LB_krw':lb,
       'weighted_MAE':basew,
       'groupopt_missing_site_hours':-1,'holdout_missing_site_hours':-1,
    }


def cli(rank):
    if rank not in SPEC:raise ValueError('rank must be in 11..15')
    parser=argparse.ArgumentParser(description=f'No28 rank {rank}: {SPEC[rank][2]}')
    parser.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--feature-parity',choices=['nat_feat','submission_safe'],default='nat_feat')
    parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args()
    cid=SPEC[rank][0]
    if args.fresh:
        out=Path(args.data_dir).expanduser().resolve()/'output'/'newcand_no28'/f'{cid:02d}_{SPEC[rank][1]}'
        if out.exists():
            import shutil
            b=out.with_name(out.name+time.strftime('_backup_%Y%m%d_%H%M%S'))
            shutil.move(out,b)
            log(f'Backed up old checkpoint: {b}')
    ctx=nc.prepare_context(args,cid)
    p=ctx.out/'summary.json'
    if p.exists():
        log(f'RESUME COMPLETE {p}')
        return json.loads(p.read_text(encoding='utf8'))
    funcs={11:rank11,12:rank12,13:rank13,14:rank14,15:rank15}
    return funcs[rank](ctx)
