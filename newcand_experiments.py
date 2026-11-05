"""One hypothesis per experiment; all candidates preserve No.28 ensemble as baseline.

No.28 is always 51 features + L1*cap*SMP*k + 50/50 LGB/Cat + r10/r17.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from newcand_common import (log,atomic_pickle,atomic_parquet,atomic_npy,atomic_json,resume_parquet,
    resume_pickle,features,row_weights,clear_sky_factor,no28_predictions,
    train_predict_four,baseline_assignment,evaluate,base_aggregation,
    grouped_columns,grouped_weights,group_prediction_frames,load_or_fit_group_model)


def profile_experiment(ctx):
    """No.28 + four target-encoded site/season/hour profile features.
    Leave-one-ROW-out statistics in training to avoid direct target self-leakage.
    """
    key=ctx.out/'site_profiles.pkl'
    prof=resume_pickle(key)
    if prof is None:
        z=ctx.fit[['site_id','ts','energy']].merge(ctx.sites[['site_id','capacity_kw']],on='site_id',validate='many_to_one')
        z['cf']=(z.energy/z.capacity_kw).clip(0,1.2)
        z['season']=(z.ts.dt.month%12//3).astype('int8')
        z['hour']=z.ts.dt.hour.astype('int8')
        def ag(cols):
            v=z.groupby(cols,observed=True).cf.agg(['sum','count']).reset_index()
            return v.rename(columns={'sum':'S','count':'N'})
        prof={k:ag(v) for k,v in [('all',['site_id']),('season',['site_id','season']),('hour',['site_id','hour']),('afternoon',['site_id','afternoon'])] if k!='afternoon'}
        z['afternoon']=(z.ts.dt.hour>=12).astype('int8')
        prof['afternoon']=ag(['site_id','afternoon'])
        atomic_pickle(prof,key)
        del z
    newcols=['site_profile_all','site_profile_season','site_profile_hour','site_profile_afternoon']
    def extra(df,r):
        df=df.copy()
        df['season']=(df.ts.dt.month%12//3).astype('int8')
        df['afternoon']=(df.ts.dt.hour>=12).astype('int8')
        q=(df.energy/df.cap).clip(0,1.2) if 'energy' in df else pd.Series(0.,index=df.index)
        istrain='energy' in df and len(df)==len(ctx.fit)
        for label,cols in [('all',['site_id']),('season',['site_id','season']),('hour',['site_id','hour']),('afternoon',['site_id','afternoon'])]:
            v=prof[label]
            ix=pd.MultiIndex.from_frame(df[cols])
            stats=v.set_index(cols).reindex(ix)
            S=stats['S'].to_numpy(dtype='float64')
            N=stats['N'].to_numpy(dtype='float64')
            if istrain:
                S=S-q.to_numpy(dtype='float64')
                N=N-1.
            out=np.divide(S,N,out=np.full(len(df),np.nan),where=N>0)
            if label!='all':
                # fall back on all-site mean if sparse
                baseline=prof['all'].set_index('site_id').reindex(df.site_id)
                backup=np.divide(baseline['S'].to_numpy(),baseline['N'].to_numpy())
                out=np.where(np.isfinite(out),out,backup)
            df['site_profile_'+label]=np.nan_to_num(out,nan=0.0).astype('float32')
        return df.drop(columns=['season','afternoon'])
    pg,ph=train_predict_four(ctx,'site_profile',extra=extra,extra_cols=newcols)
    return evaluate(ctx,pg,ph,notes={'new_features':newcols,'train_profile':'leave_one_row_out'})


def spatial_experiment(ctx):
    """Nearby station raw forecasts, not national PCA; 2km perturbed site coords only."""
    ref=ctx.out/'neighbors.pkl'
    info=resume_pickle(ref)
    if info is None:
        sites=ctx.sites
        info={}
        for kind,col in [('wa','wa_station'),('wb','wb_station')]:
            centers=sites.groupby(col,observed=True)[['lat','lon']].mean()
            cs=centers.to_numpy(dtype=float)
            lat=sites.lat.to_numpy(dtype=float)
            lon=sites.lon.to_numpy(dtype=float)
            xyz=np.column_stack([lat,lon*np.cos(np.deg2rad(lat))])
            centersxy=np.column_stack([cs[:,0],cs[:,1]*np.cos(np.deg2rad(cs[:,0]))])
            distance=((xyz[:,None,:]-centersxy[None,:,:])**2).sum(axis=2)
            order=np.argsort(distance,axis=1)
            ids=centers.index.to_numpy()
            chosen=[]
            for i,r in sites.iterrows():
                assigned=r[col]
                neighbors=[ids[j] for j in order[i] if ids[j]!=assigned][:2]
                if len(neighbors)<2: neighbors+=([assigned]*(2-len(neighbors)))
                chosen.append(neighbors)
            for rank in range(2):
                info[f'{kind}_{rank+1}']=pd.Series([z[rank] for z in chosen],index=sites.site_id)
        atomic_pickle(info,ref)
    cols=['near_wa1_cloud','near_wa2_cloud','near_wa1_rain',
          'near_wb1_cloud','near_wb2_cloud','near_wb1_uv']
    lookup={}
    for r in (10,17):
        a=ctx.fa.loc[ctx.fa['round']==r,['ts','wa_station','wa_cloud','wa_rain']]
        b=ctx.fb.loc[ctx.fb['round']==r,['ts','wb_station','wb_cloud','wb_uv_idx']]
        lookup[r]={'wa':a.set_index(['ts','wa_station']), 'wb':b.set_index(['ts','wb_station'])}
        if not lookup[r]['wa'].index.is_unique or not lookup[r]['wb'].index.is_unique:
            raise ValueError('Duplicate forecast station-time keys')
    def extra(df,r):
        d=df.copy()
        for kind,station,vars in [('wa','wa_station',['wa_cloud','wa_rain']),('wb','wb_station',['wb_cloud','wb_uv_idx'])]:
            for rank in (1,2):
                match=df.site_id.map(info[f'{kind}_{rank}'])
                idx=pd.MultiIndex.from_arrays([df.ts.to_numpy(),match.to_numpy()],names=['ts',station])
                fields=lookup[r][kind].reindex(idx)
                d[f'near_{kind}{rank}_cloud']=fields[f'{kind}_cloud'].to_numpy(dtype='float32')
                if rank==1:
                    target='wa_rain' if kind=='wa' else 'wb_uv_idx'
                    alias='rain' if kind=='wa' else 'uv'
                    d[f'near_{kind}1_{alias}']=fields[target].to_numpy(dtype='float32')
        return d
    pg,ph=train_predict_four(ctx,'neighbor',extra=extra,extra_cols=cols)
    return evaluate(ctx,pg,ph,notes={'new_features':cols,'station_positions':'mean anonymized site coords'})


def group_median(ctx):
    from lightgbm import LGBMRegressor
    mask=baseline_assignment(ctx)
    cal,hold=base_aggregation(ctx,mask)
    def fit(d):
        x=d[grouped_columns()].astype('float32')
        y=d.actual.astype('float32')
        return LGBMRegressor(objective='regression_l1',n_estimators=350,
            learning_rate=.035,num_leaves=15,min_child_samples=80,
            n_jobs=ctx.args.threads,random_state=ctx.args.seed,verbose=-1).fit(
                x,y,sample_weight=grouped_weights(ctx,d))
    model=load_or_fit_group_model(ctx,'group_median',cal,fit)
    predcal=model.predict(cal[grouped_columns()].astype('float32'))
    predhold=model.predict(hold[grouped_columns()].astype('float32'))
    pg,ph=group_prediction_frames(ctx,cal,hold,predcal,predhold)
    return evaluate(ctx,pg,ph,mask,True,notes={'training':'group_opt labels only','method':'pooled group L1, No28 sum feature'})


def direct_reconcile(ctx):
    from lightgbm import LGBMRegressor
    mask=baseline_assignment(ctx)
    cal,hold=base_aggregation(ctx,mask)
    def to_total(df):
        dg=df.groupby('ts',as_index=False).agg(actual=('actual','sum'),no28=('no28','sum'),capacity=('capacity','sum'))
        dt=dg.ts
        dg['hour']=dt.dt.hour
        dg['hour_sin']=np.sin(2*np.pi*dt.dt.hour/24)
        dg['hour_cos']=np.cos(2*np.pi*dt.dt.hour/24)
        dg['doy_sin']=np.sin(2*np.pi*dt.dt.dayofyear/365.25)
        dg['doy_cos']=np.cos(2*np.pi*dt.dt.dayofyear/365.25)
        dg['pred_cf']=dg.no28/dg.capacity
        return dg
    total_cal=to_total(cal); total_hold=to_total(hold)
    cols=['no28','capacity','hour_sin','hour_cos','doy_sin','doy_cos','pred_cf']
    def fit(d):
        xx=d[cols].astype('float32')
        return LGBMRegressor(objective='regression_l1',n_estimators=300,
           learning_rate=.04,num_leaves=15,min_child_samples=40,
           n_jobs=ctx.args.threads,random_state=42,verbose=-1).fit(xx,
           d.actual,sample_weight=grouped_weights(ctx,d))
    models={}
    for gr in ('G1','G2'):
        models[gr]=load_or_fit_group_model(ctx,'group_direct_'+gr,cal.loc[cal.group==gr],fit)
    models['TOTAL']=load_or_fit_group_model(ctx,'group_direct_total',total_cal,fit)
    def calc(d,tot):
        z=d[['ts','group']].copy()
        z['direct']=np.nan
        for gr in ('G1','G2'):
            m=d.group==gr
            z.loc[m,'direct']=models[gr].predict(d.loc[m,cols].astype('float32'))
        sum_direct=z.groupby('ts',observed=True).direct.transform('sum')
        total_pred=pd.Series(models['TOTAL'].predict(tot[cols].astype('float32')),index=tot.ts)
        delta=(z.ts.map(total_pred).to_numpy()-sum_direct.to_numpy())/2.
        return np.maximum(z.direct.to_numpy()+delta,0.0)
    pg,ph=group_prediction_frames(ctx,cal,hold,calc(cal,total_cal),calc(hold,total_hold))
    return evaluate(ctx,pg,ph,mask,True,notes={'method':'G1+G2+total direct L1; equal-share reconciliation'})


def analog_ensemble(ctx):
    from sklearn.neighbors import NearestNeighbors
    baseg,baseh=no28_predictions(ctx)
    cached=resume_parquet(ctx.out/'analog_eval.parquet',lambda d:'analog_pred' in d)
    if cached is None:
        # 17h forecast analog, No.28 10h+17h ensemble kept as 80% anchor.
        r=17
        log('BUILD analog reference train features')
        tr=features(ctx,ctx.fit,r)
        ev=pd.concat([ctx.group_actual,ctx.hold_actual],ignore_index=True)
        te=features(ctx,ev,r)
        def mat(d):
            h=d.ts.dt.hour.to_numpy()
            return np.column_stack([
              np.asarray(d.wa_cloud,dtype=float)/35.,
              np.asarray(d.wb_cloud,dtype=float)/35.,
              np.asarray(d.wa_temp,dtype=float)/15.,
              np.asarray(d.wb_uv_idx,dtype=float)/4.,
              np.asarray(d.elev,dtype=float)/25.,
              np.sin(2*np.pi*h/24)*1.5,np.cos(2*np.pi*h/24)*1.5,
              np.asarray(d.doy_sin,dtype=float)*0.5,np.asarray(d.doy_cos,dtype=float)*0.5,
            ]).astype('float32')
        a=mat(tr);b=mat(te)
        a=np.nan_to_num(a,nan=0.,posinf=0.,neginf=0.)
        b=np.nan_to_num(b,nan=0.,posinf=0.,neginf=0.)
        trsite=tr.site_id.to_numpy();tesite=te.site_id.to_numpy()
        target=(tr.energy/tr.cap).clip(0,1.2).to_numpy(dtype='float32')
        cap=te.cap.to_numpy(dtype='float32')
        elev=te.elev.to_numpy(dtype='float32')
        progress=ctx.out/'analog_partial.npy'
        progress_info=ctx.out/'analog_partial_progress.json'
        done_count=0
        analog=np.zeros(len(te),dtype='float32')
        if progress.exists() and progress_info.exists():
            try:
                previous=np.load(progress)
                meta=json.loads(progress_info.read_text(encoding='utf8'))
                if len(previous)==len(te) and meta.get('n_rows')==len(te):
                    analog=previous
                    done_count=int(meta.get('done_sites',0))
                    log('RESUME analog site loop at site index',done_count)
            except Exception as exc:
                log('Could not read analog partial checkpoint:',exc)
        for j,site in enumerate(ctx.sites.site_id):
            if j<done_count: continue
            it=np.flatnonzero(trsite==site);iq=np.flatnonzero(tesite==site)
            if len(iq)==0:continue
            if len(it)<10:continue
            nn=NearestNeighbors(n_neighbors=min(15,len(it)),algorithm='auto',n_jobs=ctx.args.threads)
            nn.fit(a[it]);dist,idx=nn.kneighbors(b[iq])
            ww=1/(dist+0.1)
            cf=(target[it][idx]*ww).sum(axis=1)/ww.sum(axis=1)
            analog[iq]=np.where(elev[iq]>0,np.maximum(cf,0.),0.)*cap[iq]
            if (j+1)%25==0 or j+1==len(ctx.sites):
                atomic_npy(analog,progress)
                atomic_json(progress_info,{'done_sites':j+1,'n_rows':len(te)})
                log('ANALOG saved partial progress sites',j+1,'/',len(ctx.sites))
        cached=te[['site_id','ts']].copy()
        cached['analog_pred']=analog
        atomic_parquet(cached,ctx.out/'analog_eval.parquet')
        del tr,te,a,b
    original=pd.concat([baseg,baseh],ignore_index=True)
    merged=original.merge(cached,on=['site_id','ts'],validate='one_to_one')
    if len(merged)!=len(original):raise ValueError('Analog key coverage mismatch')
    merged['pred_kwh']=.80*merged.pred_kwh+.20*merged.analog_pred
    # Preserve 0 at night based No.28; both 0 at night.
    pg,ph=ctx.val.days_of(merged.ts),None
    gg=ctx.val.days_of(ctx.group_actual.ts).drop_duplicates()
    hh=ctx.val.days_of(ctx.hold_actual.ts).drop_duplicates()
    group=merged.loc[pg.isin(gg),['site_id','ts','pred_kwh']].copy()
    hold=merged.loc[pg.isin(hh),['site_id','ts','pred_kwh']].copy()
    return evaluate(ctx,group,hold,notes={'ensemble_weights':{'no28':.8,'analog17':.2},'analog_k':15})


def bounded_local_global(ctx):
    from lightgbm import LGBMRegressor
    mask=baseline_assignment(ctx)
    cal,hold=base_aggregation(ctx,mask)
    def fit(d):
        xx=d[grouped_columns()].astype('float32')
        resid=d.actual.to_numpy()-d.no28.to_numpy()
        return LGBMRegressor(objective='regression_l1',n_estimators=200,
           learning_rate=.035,num_leaves=7,min_child_samples=100,
           n_jobs=ctx.args.threads,random_state=42,verbose=-1).fit(xx,resid,
           sample_weight=grouped_weights(ctx,d))
    model=load_or_fit_group_model(ctx,'bounded_residual',cal,fit)
    def calc(d):
        raw=model.predict(d[grouped_columns()].astype('float32'))
        limit=np.maximum(.10*d.no28.to_numpy(),.02*d.capacity.to_numpy())
        return d.no28.to_numpy()+0.5*np.clip(raw,-limit,limit)
    pg,ph=group_prediction_frames(ctx,cal,hold,calc(cal),calc(hold))
    return evaluate(ctx,pg,ph,mask,True,notes={'correction':'0.5*clip(pred_residual, ±max(10% no28, 2% cap))'})


def nwp_shift(ctx):
    """Transductive covariate shift: use allowed TEST forecast inputs only, not labels."""
    from lightgbm import LGBMClassifier
    root=ctx.root/'output'/'preprocessed'/'test'
    if not root.exists(): raise FileNotFoundError('Need preprocessed/test forecast inputs for N17')
    fa_te=pd.read_parquet(root/'forecast_wa.parquet',columns=['wa_station','round','ts']+list(ctx.base.WA_COLS))
    fb_te=pd.read_parquet(root/'forecast_wb.parquet',columns=['wb_station','round','ts']+list(ctx.base.WB_COLS))
    for z in (fa_te,fb_te):z['ts']=ctx.val.to_kst(z.ts)
    # Create No.28 national/temporal features from the provided test NWP as well.
    fa_te,_=ctx.nf.add_time_feats(fa_te,'wa_station',ctx.nf.A_COLS)
    fb_te,_=ctx.nf.add_time_feats(fb_te,'wb_station',ctx.nf.B_COLS)
    if ctx.args.feature_parity=='submission_safe':
        for c in ctx.nf.A_COLS:fa_te.loc[fa_te.ts.dt.hour==0,c+'_next']=np.nan
        for c in ctx.nf.B_COLS:fb_te.loc[fb_te.ts.dt.hour==0,c+'_next']=np.nan
    fa_te,fb_te=ctx.nf.add_national(fa_te,fb_te)
    classifiers={}
    rng=np.random.default_rng(42)
    selected_cols=[c for c in ctx.base.FEATS if c not in ('cap','lat','lon')]
    for r in (10,17):
        key=ctx.out/f'covshift_classifier_r{r}.pkl'
        clf=resume_pickle(key)
        if clf is None:
            log('FIT domain classifier',r)
            tr_sample=ctx.fit.iloc[rng.choice(len(ctx.fit),min(50000,len(ctx.fit)),replace=False)].copy()
            a=features(ctx,tr_sample,r)
            # Sample a reasonable number of actual test timestamps and sites.
            time_choices=pd.DatetimeIndex(fa_te.loc[fa_te['round']==r,'ts'].drop_duplicates().sort_values())
            chosen_ts=time_choices[rng.choice(len(time_choices),min(450,len(time_choices)),replace=False)]
            ssub=ctx.sites.iloc[rng.choice(len(ctx.sites),min(100,len(ctx.sites)),replace=False)]
            test=ctx.base.features(None,ssub,fa_te,fb_te,chosen_ts,r)
            A=a[selected_cols].astype('float32').replace([np.inf,-np.inf],np.nan)
            B=test[selected_cols].astype('float32').replace([np.inf,-np.inf],np.nan)
            x=pd.concat([A,B],ignore_index=True)
            y=np.r_[np.zeros(len(A),dtype='int8'),np.ones(len(B),dtype='int8')]
            clf=LGBMClassifier(n_estimators=140,learning_rate=.04,num_leaves=15,
                min_child_samples=100,verbose=-1,n_jobs=ctx.args.threads,random_state=42)
            clf.fit(x,y)
            atomic_pickle(clf,key)
            del a,A,B,test,x,y
        classifiers[r]=clf
    def weights(df,base_weights,r):
        px=classifiers[r].predict_proba(df[selected_cols].astype('float32'))[:,1]
        ratio=np.clip(px/(1-np.clip(px,0,1-1e-6)),0.25,4.0)
        ww=base_weights*ratio
        return ww/ww.mean()
    pg,ph=train_predict_four(ctx,'covshift',train_weight_fn=weights)
    return evaluate(ctx,pg,ph,notes={'shift_target':'provided 2025-26 test NWP only; no test generation labels','ratio_clip':[.25,4.]})


def total_contrast(ctx):
    """Direct OIBC loss on aggregated *group residual* via lightweight smooth L1.
    No.28 model is frozen; learns only small weather-conditioned site corrections.
    Different from pooled group residual LGB and group median.
    """
    from scipy.optimize import minimize
    mask=baseline_assignment(ctx)
    cal,hold=base_aggregation(ctx,mask)
    pg,ph=no28_predictions(ctx)
    feat_names=['constant','wa_cloud','wb_cloud','wb_uv_idx','elev','cloud_diff','a_cloud_dev','b_cloud_dev']
    def aggregate_basis(actual, pred):
        # Model correction is cap * (linear weather basis) per site, then group sum.
        # Each row uses pre-issued forecast only, no true weather at evaluation.
        ass=pd.Series(np.where(mask,1,0),index=ctx.sites.site_id)
        df=features(ctx,actual,17)
        df['group'] =df.site_id.map(ass)
        v=np.column_stack([
           np.ones(len(df)),np.nan_to_num(df.wa_cloud.to_numpy()/100),
           np.nan_to_num(df.wb_cloud.to_numpy()/100),
           np.nan_to_num(df.wb_uv_idx.to_numpy()/15),
           np.maximum(df.elev.to_numpy(),0)/90,
           np.nan_to_num(df.cloud_diff.to_numpy()/100),
           np.nan_to_num(df.a_cloud_dev.to_numpy()/100),
           np.nan_to_num(df.b_cloud_dev.to_numpy()/100),
        ]).astype('float32')
        v[df.elev.to_numpy()<=0]=0.
        d=df[['ts','group']].copy()
        for j,name in enumerate(feat_names):d[name]=(df.cap.to_numpy()*v[:,j]).astype('float32')
        a=d.groupby(['ts','group'],observed=True,as_index=False)[feat_names].sum()
        a['group']=np.where(a.group==1,'G1','G2')
        return a
    Xg=resume_parquet(ctx.out/'group_basis.parquet',lambda d:all(z in d for z in feat_names))
    if Xg is None:
        Xg=aggregate_basis(ctx.group_actual,pg)
        atomic_parquet(Xg,ctx.out/'group_basis.parquet')
    Xh=resume_parquet(ctx.out/'holdout_basis.parquet',lambda d:all(z in d for z in feat_names))
    if Xh is None:
        Xh=aggregate_basis(ctx.hold_actual,ph)
        atomic_parquet(Xh,ctx.out/'holdout_basis.parquet')
    def joined(ag,X):
        a=ag.merge(X,on=['ts','group'],validate='one_to_one')
        if len(a)!=len(ag):raise ValueError('Missing basis rows')
        a['err']=a.no28-a.actual
        return a
    g=joined(cal,Xg);h=joined(hold,Xh)
    scale=g.capacity.to_numpy().clip(min=1)
    xx=g[feat_names].to_numpy(dtype=float)/scale[:,None]
    error=g.err.to_numpy(dtype=float)/scale
    w=grouped_weights(ctx,g);w=w/w.mean()
    groupidx=(g.group=='G1').to_numpy(dtype=int)
    p=ctx.out/'coefficients.pkl'
    coefficients=resume_pickle(p)
    if coefficients is None:
        def calc(b):
            co=b.reshape(2,-1)
            correction=(xx*co[groupidx]).sum(axis=1)
            residual=error+correction
            loss=np.average(np.sqrt(residual**2+0.002**2),weights=w)
            loss+=.03*np.square(b).sum()
            deri=w/np.sum(w)*residual/np.sqrt(residual**2+0.002**2)
            grad=np.zeros_like(co)
            for grp in (0,1):
                m=groupidx==grp
                grad[grp]=(deri[m,None]*xx[m]).sum(axis=0)
            grad+=.06*co
            return loss,grad.ravel()
        result=minimize(calc,np.zeros(16),jac=True,method='L-BFGS-B',
                        bounds=[(-.10,.10)]*16,options={'maxiter':350,'ftol':1e-11})
        coefficients=result.x.reshape(2,-1)
        atomic_pickle(coefficients,p)
        log('TOTAL-CONTRAST optimizer:',result.success,result.message)
    def predict(a):
        group=(a.group=='G1').to_numpy(dtype=int)
        feat=a[feat_names].to_numpy(dtype=float)
        correction=(feat*coefficients[group]).sum(axis=1)
        return a.no28.to_numpy()+correction
    pc,phold=group_prediction_frames(ctx,g,h,predict(g),predict(h))
    return evaluate(ctx,pc,phold,mask,True,notes={'method':'weighted smoothed L1 over grouped residual','coefficients':coefficients.tolist()})


def clear_sky(ctx):
    pg,ph=train_predict_four(ctx,'clearsky',target_mode='clearsky')
    return evaluate(ctx,pg,ph,notes={'target':'cf / irradiance-derived relative clear-sky proxy; clamp floor .08'})


def weather_moe(ctx):
    """No.28 backbone + cloud-regime expert LGB models, only one new MoE concept."""
    import lightgbm as lgb
    baseg,baseh=no28_predictions(ctx)
    eval_actual=pd.concat([ctx.group_actual,ctx.hold_actual],ignore_index=True)
    def regime(d):
        cloud=(d.wa_cloud.to_numpy(dtype=float)+d.wb_cloud.to_numpy(dtype=float))/2
        return np.where(cloud<30,0,np.where(cloud>65,2,1))
    preds_by_round=[]
    for r in (10,17):
        model_preds={}
        for regime_id in (0,1,2):
            key=f'moe_r{r}_regime{regime_id}'
            ppath=ctx.out/f'{key}_eval.parquet'
            p=resume_parquet(ppath,lambda d:'pred_kwh' in d and len(d)==len(eval_actual))
            if p is None:
                mp=ctx.out/f'{key}_model.pkl'
                model=resume_pickle(mp)
                if model is None:
                    d=features(ctx,ctx.fit,r)
                    keep=(d.elev.to_numpy()>0)&(regime(d)==regime_id)
                    d=d.loc[keep]
                    if len(d)<5000:raise ValueError(f'Too few training rows in regime {regime_id}: {len(d)}')
                    log('FIT MOE round',r,'regime',regime_id,'samples',len(d))
                    model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=350,
                       learning_rate=.045,num_leaves=63,min_child_samples=150,
                       subsample=.8,subsample_freq=1,colsample_bytree=.8,
                       verbose=-1,n_jobs=ctx.args.threads,random_state=42)
                    model.fit(d[ctx.base.FEATS].astype('float32'),
                        (d.energy/d.cap).clip(0,1.2),sample_weight=row_weights(ctx,d))
                    atomic_pickle(model,mp)
                    del d
                d=features(ctx,eval_actual,r)
                forecast=np.maximum(model.predict(d[ctx.base.FEATS].astype('float32')),0)
                forecast=np.where(d.elev>0,forecast,0.)*d.cap.to_numpy()
                p=d[['site_id','ts']].copy();p['pred_kwh']=forecast
                p['regime']=regime(d)
                atomic_parquet(p,ppath)
                del d,model
            model_preds[regime_id]=p
        main=model_preds[0][['site_id','ts','regime']].copy()
        main['expert_pred']=0.
        for reg in (0,1,2):
            pred=model_preds[reg]
            same=(main.site_id.to_numpy()==pred.site_id.to_numpy()).all() and (main.ts.to_numpy()==pred.ts.to_numpy()).all()
            if not same:raise ValueError('MOE prediction row order mismatch')
            ix=main.regime.to_numpy()==reg
            main.loc[ix,'expert_pred']=pred.pred_kwh.to_numpy()[ix]
        preds_by_round.append(main[['site_id','ts','expert_pred']].rename(columns={'expert_pred':f'r{r}'}))
    both=preds_by_round[0].merge(preds_by_round[1],on=['site_id','ts'],validate='one_to_one')
    both['expert']=.5*(both.r10+both.r17)
    br=pd.concat([baseg,baseh],ignore_index=True).merge(both[['site_id','ts','expert']],on=['site_id','ts'],validate='one_to_one')
    br['pred_kwh']=.75*br.pred_kwh+.25*br.expert
    from newcand_common import split_predictions
    pg,ph=split_predictions(ctx,br[['site_id','ts','pred_kwh']])
    return evaluate(ctx,pg,ph,notes={'gate':'mean WA/WB cloud <30 / 30-65 / >65','blend':'75% No28 + 25% expert LGB'})


def dispatch(ctx):
    return {
        1:profile_experiment,2:spatial_experiment,3:group_median,
        4:direct_reconcile,5:analog_ensemble,6:bounded_local_global,
        7:nwp_shift,8:total_contrast,9:clear_sky,10:weather_moe,
    }[ctx.candidate](ctx)
