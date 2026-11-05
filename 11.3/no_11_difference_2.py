#!/usr/bin/env python3
"""OIBC No.11 weather-quantile difference-order-2 experiment, local validation only.

Input to each weather model: BOTH providers, BOTH round=10 and round=17,
station A and B, forecast differences, calendar features.
Outputs: conditional q05/q25/q50/q75/q95 for measured A/B cloud/temp/humidity.
Then append OOF quantile features to the original 24-feature weighted-L1
LightGBM generation model (round 10 and 17; mean predictions).

Split: EXACT random delivery-day 60/20/20 (seed 42), with weather-model
3-fold day OOF INSIDE model_train; group_opt only for Greedy; untouched holdout.
Only train weather observations are read and only for model_train dates.
This DOES NOT create a competition test submission.

Original baseline.py, baseline_optimizer.py, and other experiments unmodified.
Place at data/codes/no_11_difference_2.py and execute with python3.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = HERE.parent if HERE.name == "codes" else HERE
SEED = 42
DIFF_ORDER = 2  # Add 1-hour first difference and centered second difference ONLY inside weather Quantile LightGBM
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
# Only true corresponding observed columns are selected; no guessed mappings.
TARGETS = {
    "a_cloud": ("wa_cloud", "wa_station", 0., 100.),
    "a_temp": ("wa_temp", "wa_station", -50., 55.),
    "a_humidity": ("wa_humidity", "wa_station", 0., 100.),
    "b_cloud": ("wb_cloud", "wb_station", 0., 100.),
    "b_temp": ("wb_temp", "wb_station", -50., 55.),
    "b_humidity": ("wb_rel_hum", "wb_station", 0., 100.),
}
BASE_PARAMS = dict(n_estimators=600, learning_rate=.06, num_leaves=127,
                   min_child_samples=200, subsample=.8, subsample_freq=1,
                   colsample_bytree=.8, verbose=-1, n_jobs=-1)
WX_PARAMS = dict(n_estimators=170, learning_rate=.07, num_leaves=31,
                 min_child_samples=120, colsample_bytree=.9,
                 reg_lambda=2., verbose=-1, n_jobs=-1, random_state=SEED)


def log(s):
    print(f"[WX Quantile] {s}", flush=True)


def kst(s):
    s = pd.to_datetime(s)
    return s.dt.tz_localize('Asia/Seoul') if s.dt.tz is None else s.dt.tz_convert('Asia/Seoul')


def day(s):
    return (s - pd.Timedelta(hours=1)).dt.normalize()


def load_py(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_602020(gen, seed):
    days = pd.DatetimeIndex(day(gen.ts).drop_duplicates().sort_values())
    order = np.random.default_rng(seed).permutation(len(days))
    ntrain, nopt = int(.6*len(days)), int(.2*len(days))
    train_days, opt_days, hold_days = (days[order[:ntrain]],
                  days[order[ntrain:ntrain+nopt]],days[order[ntrain+nopt:]])
    if min(len(train_days),len(opt_days),len(hold_days)) == 0:
        raise ValueError('Too few delivery days')
    d = day(gen.ts)
    train = gen.loc[d.isin(train_days)].copy()
    opt = gen.loc[d.isin(opt_days)].copy()
    hold = gen.loc[d.isin(hold_days)].copy()
    dates = pd.DataFrame({'day_kst':days.strftime('%Y-%m-%d'),
      'split':np.where(days.isin(train_days),'model_train',
              np.where(days.isin(opt_days),'group_opt','holdout'))})
    log(f'60/20/20 seed={seed}, days {len(train_days)}/{len(opt_days)}/{len(hold_days)}; '
        f'rows {len(train):,}/{len(opt):,}/{len(hold):,}')
    return train,opt,hold,dates


def check_keys(df, keys, label):
    if df.duplicated(keys).any():
        raise ValueError(f'{label}: duplicate keys {keys} (first resolve duplicates)')


def load_data(root, base):
    sites = pd.read_csv(root/'sites.csv')[['site_id','current_group','capacity_kw','lat','lon',
                                          'wa_station','wb_station']]
    if len(sites)!=253 or sites.site_id.duplicated().any():
        raise ValueError('sites.csv must contain 253 unique plants')
    gen = pd.read_parquet(root/'train'/'generation.parquet',columns=['site_id','ts','energy'])
    gen['ts']=kst(gen.ts)
    gen=gen.loc[gen.ts>=base.TRAIN_FROM].copy()
    check_keys(gen,['site_id','ts'],'generation')
    p=root/'output'/'preprocessed'/'train'
    fa=pd.read_parquet(p/'forecast_wa.parquet',columns=['wa_station','round','ts']+base.WA_COLS)
    fb=pd.read_parquet(p/'forecast_wb.parquet',columns=['wb_station','round','ts']+base.WB_COLS)
    for d in (fa,fb):
        d['ts']=kst(d.ts)
        d.drop(d.index[d.ts<base.TRAIN_FROM],inplace=True)
    check_keys(fa,['wa_station','round','ts'],'WA forecast')
    check_keys(fb,['wb_station','round','ts'],'WB forecast')
    fa = add_forecast_differences(fa, 'wa_station', base.WA_COLS)
    fb = add_forecast_differences(fb, 'wb_station', base.WB_COLS)
    log(f'Forecast derivatives: WA={len(difference_cols(base.WA_COLS))} cols, '
        f'WB={len(difference_cols(base.WB_COLS))} cols (order {DIFF_ORDER})')
    oa=pd.read_parquet(p/'observed_wa.parquet',columns=['wa_station','ts','wa_cloud','wa_temp','wa_humidity'])
    ob=pd.read_parquet(p/'observed_wb.parquet',columns=['wb_station','ts','wb_cloud','wb_temp','wb_rel_hum'])
    for d in (oa,ob):
        d['ts']=kst(d.ts)
        d.drop(d.index[d.ts<base.TRAIN_FROM],inplace=True)
    # Previously observed: a few conflicting duplicate station-hours. Do NOT
    # select one conflicting label arbitrarily: exclude that station-hour.
    for d,key,name in ((oa,'wa_station','WA observed'),(ob,'wb_station','WB observed')):
        before=len(d)
        duplicate=d.duplicated([key,'ts'],keep=False)
        if duplicate.any():
            unique=d.loc[duplicate].drop_duplicates()
            conflicted=unique.duplicated([key,'ts'],keep=False)
            clean=pd.concat([d.loc[~duplicate],unique.loc[~conflicted]],ignore_index=True)
            d.drop(d.index,inplace=True)
            for col in clean.columns:
                d[col]=clean[col].array
            log(f'{name}: {before-len(d)} duplicate/conflict rows handled')
        check_keys(d,[key,'ts'],name)
    return sites,gen,fa,fb,oa,ob



def difference_cols(cols):
    """New weather-input columns only. No change to generation LightGBM features."""
    d1 = [f'{c}__diff1h' for c in cols]
    d2 = [f'{c}__diff2h' for c in cols] if DIFF_ORDER >= 2 else []
    return d1 + d2


def add_forecast_differences(frame, station_col, cols):
    """Compute time-local forecast derivatives before weather-model joins.

    For each (station, forecast round, delivery day), and only at exactly
    consecutive 1-hour target timestamps:
      first  = F[t] - F[t-1]
      second = F[t+1] - 2*F[t] + F[t-1]
    No observed weather / generation values are used. Forecasts from different
    rounds or delivery days are never mixed. Missing neighbors => NaN; these
    remain missing (LightGBM supports NaN) rather than inventing zero trends.
    """
    if DIFF_ORDER not in (1, 2):
        raise ValueError('DIFF_ORDER must be 1 or 2')
    missing = set([station_col, 'round', 'ts'] + list(cols)) - set(frame.columns)
    if missing:
        raise ValueError(f'Forecast missing columns: {sorted(missing)}')
    if frame.duplicated([station_col, 'round', 'ts']).any():
        raise ValueError('Duplicate station/round/ts before differencing')

    # Work on a copy: the baseline generation feature function still receives
    # the same raw forecast fields, plus harmless extra derivative columns.
    f = frame.sort_values([station_col, 'round', 'ts'], kind='mergesort').reset_index(drop=True).copy()
    delivery_day = day(f['ts'])
    group_keys = [f[station_col], f['round'], delivery_day]
    grouped = f.groupby(group_keys, sort=False, observed=True)
    prev_ts = grouped['ts'].shift(1)
    valid_prev = (f['ts'] - prev_ts).eq(pd.Timedelta(hours=1)).to_numpy()
    prev = grouped[list(cols)].shift(1)
    for c in cols:
        now = pd.to_numeric(f[c], errors='coerce').to_numpy(dtype='float32')
        previous = pd.to_numeric(prev[c], errors='coerce').to_numpy(dtype='float32')
        first = now - previous
        first[~valid_prev] = np.nan
        f[f'{c}__diff1h'] = first

    if DIFF_ORDER >= 2:
        next_ts = grouped['ts'].shift(-1)
        valid_next = (next_ts - f['ts']).eq(pd.Timedelta(hours=1)).to_numpy()
        following = grouped[list(cols)].shift(-1)
        valid_triplet = valid_prev & valid_next
        for c in cols:
            now = pd.to_numeric(f[c], errors='coerce').to_numpy(dtype='float32')
            previous = pd.to_numeric(prev[c], errors='coerce').to_numpy(dtype='float32')
            after = pd.to_numeric(following[c], errors='coerce').to_numpy(dtype='float32')
            second = after - 2.0 * now + previous
            second[~valid_triplet] = np.nan
            f[f'{c}__diff2h'] = second
        del following
    del prev
    return f

def make_site_pairs(sites):
    pairs=sites[['wa_station','wb_station']].drop_duplicates().reset_index(drop=True)
    # IDs stable across every OOF fold and the final fit.
    pairs['station_a_id']=pd.Categorical(pairs.wa_station,
        categories=sorted(sites.wa_station.unique())).codes.astype('int32')
    pairs['station_b_id']=pd.Categorical(pairs.wb_station,
        categories=sorted(sites.wb_station.unique())).codes.astype('int32')
    return pairs


def weather_inputs(keys, pairs, fa, fb, base):
    """One row per (A station,B station,ts), four forecasts, NO measured inputs."""
    keys=keys[['wa_station','wb_station','ts']].drop_duplicates().copy()
    keys=keys.merge(pairs,on=['wa_station','wb_station'],how='left',validate='many_to_one')
    if keys[['station_a_id','station_b_id']].isna().any().any():
        raise ValueError('Unmapped station pair')
    wa_cols = list(base.WA_COLS) + difference_cols(base.WA_COLS)
    wb_cols = list(base.WB_COLS) + difference_cols(base.WB_COLS)
    for r in (10,17):
        a=fa.loc[fa['round']==r,['wa_station','ts']+wa_cols].rename(
            columns={v:f'a{r}_{v}' for v in wa_cols})
        b=fb.loc[fb['round']==r,['wb_station','ts']+wb_cols].rename(
            columns={v:f'b{r}_{v}' for v in wb_cols})
        keys=keys.merge(a,on=['wa_station','ts'],how='left',validate='many_to_one')
        keys=keys.merge(b,on=['wb_station','ts'],how='left',validate='many_to_one')
        del a,b
    for x in ('wa_cloud','wa_temp','wa_humidity','wa_wind_speed'):
        keys[f'a_rev_{x}']=keys[f'a17_{x}']-keys[f'a10_{x}']
    for x in ('wb_cloud','wb_temp','wb_rel_hum','wb_wind_speed','wb_uv_idx'):
        keys[f'b_rev_{x}']=keys[f'b17_{x}']-keys[f'b10_{x}']
    for r in (10,17):
        keys[f'ab_{r}_cloud_gap']=keys[f'a{r}_wa_cloud']-keys[f'b{r}_wb_cloud']
        keys[f'ab_{r}_temp_gap']=keys[f'a{r}_wa_temp']-keys[f'b{r}_wb_temp']
        keys[f'ab_{r}_humidity_gap']=keys[f'a{r}_wa_humidity']-keys[f'b{r}_wb_rel_hum']
        keys[f'ab_{r}_wind_gap_ms']=keys[f'a{r}_wa_wind_speed']-keys[f'b{r}_wb_wind_speed']/3.6
    times=pd.DatetimeIndex(keys.ts)
    doy=times.dayofyear.to_numpy()
    hr=times.hour.to_numpy()
    keys['hour']=hr.astype('float32')
    keys['doy_sin']=np.sin(2*np.pi*doy/365.25).astype('float32')
    keys['doy_cos']=np.cos(2*np.pi*doy/365.25).astype('float32')
    keys['hour_sin']=np.sin(2*np.pi*hr/24).astype('float32')
    keys['hour_cos']=np.cos(2*np.pi*hr/24).astype('float32')
    return keys


def feature_cols(d):
    ignored={'wa_station','wb_station','ts'}
    return [c for c in d.columns if c not in ignored]


def as_X(df, cols):
    x=df[cols].copy()
    for c in cols:
        x[c]=pd.to_numeric(x[c],errors='coerce').astype('int32' if c in ('station_a_id','station_b_id') else 'float32')
    return x


def pair_hour_grid(pairs, ts, chunk_hours=240):
    """Only required *target* timestamps, never chronological future weather actuals."""
    ts=pd.DatetimeIndex(ts).unique().sort_values()
    for start in range(0,len(ts),chunk_hours):
        batch=ts[start:start+chunk_hours]
        g=pd.MultiIndex.from_product([np.arange(len(pairs)),batch],names=['pair_idx','ts']).to_frame(index=False)
        g=g.merge(pairs.reset_index(names='pair_idx'),on='pair_idx',how='left',validate='many_to_one')
        yield g[['wa_station','wb_station','ts']]


def make_training_weather(pairs, times, fa,fb,oa,ob,base, chunk_hours):
    chunks=[]
    for pair_times in pair_hour_grid(pairs,times,chunk_hours):
        d=weather_inputs(pair_times,pairs,fa,fb,base)
        d=d.merge(oa,on=['wa_station','ts'],how='left',validate='many_to_one')
        d=d.merge(ob,on=['wb_station','ts'],how='left',validate='many_to_one')
        chunks.append(d)
    if not chunks:
        raise ValueError('Weather training timestamps empty')
    return pd.concat(chunks,ignore_index=True)


def training_hours(gen):
    # Need only target time periods usable to train daytime PV prediction.
    hours=pd.DatetimeIndex(gen.ts.drop_duplicates().sort_values())
    return hours[(hours.hour>=5)&(hours.hour<=21)]


def fit_quantiles(tr, cols, pairs, max_rows, seed, target_names):
    models={}
    cat=['station_a_id','station_b_id']
    rng=np.random.default_rng(seed)
    # Avoid excessively replicated station labels through multiple station-pairs.
    station_counts={
      'wa_station':pairs.groupby('wa_station').size().to_dict(),
      'wb_station':pairs.groupby('wb_station').size().to_dict(),
    }
    for target in target_names:
        obs_col,station,lower,upper=TARGETS[target]
        mask=np.isfinite(pd.to_numeric(tr[obs_col],errors='coerce').to_numpy(np.float64))
        arr=tr.loc[mask]
        arr=arr.loc[pd.to_numeric(arr[obs_col],errors='coerce').between(lower,upper)]
        if len(arr)<100:
            raise ValueError(f'{target}: too few observed weather labels {len(arr)}')
        if max_rows and len(arr)>max_rows:
            pick=rng.choice(len(arr),size=max_rows,replace=False)
            arr=arr.iloc[np.sort(pick)]
        x=as_X(arr,cols)
        y=arr[obs_col].to_numpy(np.float32)
        counts=arr[station].map(station_counts[station]).to_numpy(np.float64)
        w=(1.0/counts).astype('float32')
        w/=w.mean()
        log(f'  {target}: {len(arr):,} station-pair-hours; fit quantiles {QUANTILES}')
        for q in QUANTILES:
            model=lgb.LGBMRegressor(objective='quantile',alpha=q,**WX_PARAMS)
            model.fit(x,y,sample_weight=w,categorical_feature=cat)
            models[(target,q)]=model
        del arr,x,y,w
        gc.collect()
    return models


def quantile_cols(targets):
    return [f'wx_{t}_q{int(q*100):02d}' for t in targets for q in QUANTILES]


def predict_quantiles(d, cols, models, targets):
    X=as_X(d,cols)
    out=d[['wa_station','wb_station','ts']].copy()
    for t in targets:
        lo,hi=TARGETS[t][2:]
        preds=[]
        for q in QUANTILES:
            p=models[(t,q)].predict(X)
            preds.append(np.clip(p,lo,hi))
        ps=np.sort(np.stack(preds,axis=1),axis=1)  # avoid quantile crossing
        for j,q in enumerate(QUANTILES):
            out[f'wx_{t}_q{int(q*100):02d}']=ps[:,j].astype('float32')
    del X
    return out


def make_quantile_features(keys, pairs, fa,fb,base,cols,models,targets,
                           batch_hours=240):
    """Predict only requested station-pairs×timestamps; no observed input."""
    all_keys=keys[['wa_station','wb_station','ts']].drop_duplicates()
    # Chunk across timestamps to avoid a >2M-row intermediate joined frame.
    ts=pd.DatetimeIndex(all_keys.ts.drop_duplicates()).sort_values()
    chunks=[]
    for start in range(0,len(ts),batch_hours):
        hours=ts[start:start+batch_hours]
        sub=all_keys.loc[all_keys.ts.isin(hours)]
        d=weather_inputs(sub,pairs,fa,fb,base)
        q=predict_quantiles(d,cols,models,targets)
        chunks.append(q)
        del d,q,sub
    out=pd.concat(chunks,ignore_index=True)
    check_keys(out,['wa_station','wb_station','ts'],'quantile features')
    return out


def generate_site_quantiles(gen,sites,pairs,fa,fb,base,cols,models,targets,batch_hours):
    key=gen[['site_id','ts']].merge(sites[['site_id','wa_station','wb_station']],
              on='site_id',how='left',validate='many_to_one')
    weather=make_quantile_features(key,pairs,fa,fb,base,cols,models,targets,batch_hours)
    result=key[['site_id','ts','wa_station','wb_station']].merge(weather,
                on=['wa_station','wb_station','ts'],how='left',validate='many_to_one')
    result=result[['site_id','ts']+quantile_cols(targets)]
    if len(result)!=len(gen) or result[quantile_cols(targets)].isna().any().any():
        raise ValueError('Quantile feature site-hour join missing or nonfinite')
    return result


def weather_oof(model_gen, sites,pairs,fa,fb,oa,ob,base, targets, folds, max_rows, seed, batch_hours):
    days=pd.DatetimeIndex(day(model_gen.ts).drop_duplicates().sort_values())
    perm=np.random.default_rng(seed+202).permutation(len(days))
    slots=np.array_split(perm,folds)
    target_cols=quantile_cols(targets)
    oofs=[]
    cols=None
    for fi,positions in enumerate(slots,1):
        valdays=days[positions]
        trgen=model_gen.loc[~day(model_gen.ts).isin(valdays)]
        tegen=model_gen.loc[day(model_gen.ts).isin(valdays)]
        log(f'Weather OOF fold {fi}/{folds}: train {len(trgen):,} site-hours, '
            f'held-out {len(tegen):,} site-hours')
        trweather=make_training_weather(pairs,training_hours(trgen),fa,fb,oa,ob,base,batch_hours)
        cols=feature_cols(trweather.drop(columns=['wa_cloud','wa_temp','wa_humidity',
                         'wb_cloud','wb_temp','wb_rel_hum']))
        models=fit_quantiles(trweather,cols,pairs,max_rows,seed+fi,targets)
        del trweather
        gc.collect()
        q=generate_site_quantiles(tegen,sites,pairs,fa,fb,base,cols,models,targets,batch_hours)
        oofs.append(q)
        log(f'  fold {fi}: OOF quantile features {q.shape}')
        del models,q,trgen,tegen
        gc.collect()
    result=pd.concat(oofs,ignore_index=True)
    check_keys(result,['site_id','ts'],'weather OOF')
    if len(result)!=len(model_gen):
        raise ValueError('OOF weather features do not cover all training rows')
    return result,cols


def final_weather_models(model_gen,pairs,fa,fb,oa,ob,base,cols,targets,max_rows,seed,batch_hours):
    log('Weather final fitting: model_train days only')
    full=make_training_weather(pairs,training_hours(model_gen),fa,fb,oa,ob,base,batch_hours)
    if feature_cols(full.drop(columns=['wa_cloud','wa_temp','wa_humidity',
                              'wb_cloud','wb_temp','wb_rel_hum'])) != cols:
        raise ValueError('Weather feature layout differs between OOF and final fit')
    model=fit_quantiles(full,cols,pairs,max_rows,seed+2000,targets)
    del full
    gc.collect()
    return model


def read_prices(root):
    smp=pd.read_csv(root/'train'/'smp.csv',usecols=['ts','smp'])
    smp['ts']=kst(smp.ts)
    check_keys(smp,['ts'],'train SMP')
    if (smp.smp.isna()|(smp.smp<0)).any():
        raise ValueError('Bad SMP')
    return smp.set_index('ts').smp


def weighted_fit(base, sites, gen, aux, fa,fb, prices, targets):
    feats=base.FEATS+quantile_cols(targets)
    assert not aux.duplicated(['site_id','ts']).any()
    models={}
    for r in (10,17):
        d=base.features(gen,sites,fa,fb,None,r)
        if len(d)!=len(gen):
            raise ValueError(f'Round {r} original feature join changed row count')
        d=d.merge(aux,on=['site_id','ts'],how='left',validate='one_to_one')
        if d[quantile_cols(targets)].isna().any().any():
            raise ValueError('Missing OOF weather features in weighted fit')
        d=d.loc[d.elev>0]
        x=d[feats].astype('float32')
        y=(d.energy/d.cap).clip(0,1.2).astype('float32')
        k=np.where(d.ts.dt.hour.between(11,15).to_numpy(),3.,1.)
        prices_row=d.ts.map(prices).to_numpy(np.float64)
        sw=prices_row*k*d.cap.to_numpy(np.float64)
        if not np.isfinite(sw).all() or np.any(sw<0) or sw.mean()<=0:
            raise ValueError('Bad SMP*k*capacity sample weights')
        sw/=sw.mean()
        model=lgb.LGBMRegressor(objective='regression_l1',**BASE_PARAMS)
        log(f'Weighted-L1 round={r}: X={x.shape}, extra weather features={len(feats)-24}')
        model.fit(x,y,sample_weight=sw)
        log(f'  train normalized MAE={np.abs(model.predict(x)-y).mean():.6f}')
        models[r]=model
        del d,x,y,sw
        gc.collect()
    return models


def eval_predict(base,models,sites,gen,aux,fa,fb,targets):
    feats=base.FEATS+quantile_cols(targets)
    combined=None
    for r in (10,17):
        d=base.features(gen,sites,fa,fb,None,r)
        if len(d)!=len(gen):
            raise ValueError('Round forecast join altered rows')
        d=d.merge(aux,on=['site_id','ts'],how='left',validate='one_to_one')
        if d[quantile_cols(targets)].isna().any().any():
            raise ValueError('Missing quantile features during inference')
        pred=models[r].predict(d[feats].astype('float32'))
        kwh=np.maximum(np.where(d.elev.to_numpy()>0,pred,0.),0.)*d.cap.to_numpy()
        tmp=d[['site_id','ts']].copy()
        tmp[f'p{r}']=kwh
        combined=tmp if combined is None else combined.merge(tmp,on=['site_id','ts'],validate='one_to_one')
        del d,tmp,pred,kwh
        gc.collect()
    combined['pred_kwh']=.5*(combined.p10+combined.p17)
    return combined[['site_id','ts','pred_kwh']]


def metrics(root,actual,pred,prices):
    joined=actual.merge(pred,on=['site_id','ts'],how='left',validate='one_to_one')
    if joined.pred_kwh.isna().any():
        raise ValueError('Missing group prediction')
    e=(joined.pred_kwh-joined.energy).to_numpy(np.float64)
    w=joined.ts.map(prices).to_numpy(np.float64)*np.where(joined.ts.dt.hour.between(11,15),3.,1.)
    return dict(MAE_kwh=float(np.mean(np.abs(e))),RMSE_kwh=float(np.sqrt(np.mean(e**2))),
                weighted_MAE_kwh=float(np.dot(w,np.abs(e))/w.sum()))


def residual_matrix(actual, pred, sites, title):
    join=actual[['site_id','ts','energy']].merge(pred,on=['site_id','ts'],how='left',validate='one_to_one')
    if join.pred_kwh.isna().any():
        raise ValueError('Missing prediction for label')
    join['err']=join.pred_kwh-join.energy
    grid=join.pivot(index='ts',columns='site_id',values='err').reindex(columns=sites.site_id).sort_index()
    missing=int(grid.isna().sum().sum())
    log(f'{title}: {len(grid):,} hours; {missing:,} missing site-hours residual=0 (existing Greedy convention)')
    return grid.fillna(0.).to_numpy(np.float64),pd.DatetimeIndex(grid.index),missing


def group_score(opt,root,mat,times,original,optimized):
    w=opt.time_weights(root,times)
    initial=float(opt.penalty(mat,w,original))
    final=float(opt.penalty(mat,w,optimized))
    lb=float(opt.PENALTY_RATE*np.dot(w,np.abs(mat.sum(axis=1))))
    return dict(W_initial_krw=initial,W_optimized_krw=final,W_lower_bound_krw=lb,
                W_reduction_krw=initial-final,W_reduction_pct=100*(initial-final)/initial if initial else 0.,
                hours=len(times))


def run():
    p=argparse.ArgumentParser(description='Quantile A+B Round10/17 observed-weather distribution features')
    p.add_argument('--data-dir',type=Path,default=ROOT_DEFAULT)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--weather-folds',type=int,default=3)
    p.add_argument('--weather-max-rows',type=int,default=150000,
                   help='Max training station-pair-hour rows per weather target per fold (0=all)')
    p.add_argument('--batch-hours',type=int,default=240)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--max-iterations',type=int,default=0)
    p.add_argument('--targets',default=','.join(TARGETS),
                   help='Comma-separated weather targets: '+','.join(TARGETS))
    a=p.parse_args()
    targets=[v.strip() for v in a.targets.split(',')]
    if not targets or len(set(targets))!=len(targets) or any(x not in TARGETS for x in targets):
        p.error('--targets invalid; choose from '+','.join(TARGETS))
    if a.weather_folds<2 or a.batch_hours<1 or a.batch_size<1 or a.weather_max_rows<0:
        p.error('Invalid options')
    started=time.monotonic()
    root=a.data_dir.expanduser().resolve()
    base=load_py(root/'baseline.py','base_weather_quantile')
    optimizer=HERE/'baseline_optimizer.py'
    if not optimizer.exists():
        optimizer=root/'codes'/'baseline_optimizer.py'
    opt=load_py(optimizer,'opt_weather_quantile')
    out=root/'output'/'validation_602020'/'split602020_no_11_difference_2'
    out.mkdir(parents=True,exist_ok=True)
    log(f'OUTPUT={out}')
    sites,gen,fa,fb,oa,ob=load_data(root,base)
    train,group,hold,days=split_602020(gen,a.seed)
    days.to_csv(out/'split_days.csv',index=False)
    del gen
    pairs=make_site_pairs(sites)
    log(f'Station pairs={len(pairs)}, targets={targets}, quantiles={QUANTILES}, '
        f'forecast difference order={DIFF_ORDER}')
    aux_train,wx_cols=weather_oof(train,sites,pairs,fa,fb,oa,ob,base,targets,
                         a.weather_folds,a.weather_max_rows,a.seed,a.batch_hours)
    log(f'Weather OOF features complete: {aux_train.shape}')
    wx_models=final_weather_models(train,pairs,fa,fb,oa,ob,base,wx_cols,targets,
                                   a.weather_max_rows,a.seed,a.batch_hours)
    del oa,ob
    gc.collect()
    qextra=quantile_cols(targets)
    aux_train[qextra].describe().to_csv(out/'train_oof_quantile_summary.csv')
    base_models=weighted_fit(base,sites,train,aux_train,fa,fb,read_prices(root),targets)
    del train,aux_train
    gc.collect()
    prices=read_prices(root)
    results={}
    for label,actual in (('group_opt',group),('holdout',hold)):
        log(f'Predicting {label}, {len(actual):,} site-hours')
        aux=generate_site_quantiles(actual,sites,pairs,fa,fb,base,wx_cols,wx_models,
                                    targets,a.batch_hours)
        pred=eval_predict(base,base_models,sites,actual,aux,fa,fb,targets)
        pred.to_parquet(out/f'{label}_site_predictions.parquet',index=False)
        mat,times,missing=residual_matrix(actual,pred,sites,label.upper())
        results[label]=dict(matrix=mat,times=times,missing=missing,
                            metrics=metrics(root,actual,pred,prices))
        del aux,pred
        gc.collect()
    del wx_models,base_models,fa,fb
    g,h=results['group_opt'],results['holdout']
    log(f'HOLDOUT MODEL MAE={h["metrics"]["MAE_kwh"]:.5f}, '
        f'RMSE={h["metrics"]["RMSE_kwh"]:.5f}, '
        f'Weighted MAE={h["metrics"]["weighted_MAE_kwh"]:.5f}')
    startmask=opt.initial_mask(sites)
    opt.check_capacity(startmask,sites.capacity_kw.to_numpy(np.float64))
    gweights=opt.time_weights(root,g['times'])
    log(f'GROUP_OPT initial W={opt.penalty(g["matrix"],gweights,startmask):,.2f} KRW')
    finalmask,greedy,trace=opt.greedy_best_improvement(g['matrix'],gweights,sites,startmask,
                                             batch_size=a.batch_size,max_iterations=a.max_iterations)
    group_score_result=group_score(opt,root,g['matrix'],g['times'],startmask,finalmask)
    hold_score_result=group_score(opt,root,h['matrix'],h['times'],startmask,finalmask)
    pd.DataFrame({'site_id':sites.site_id,
                  'group':np.where(finalmask,'G1','G2')}).to_csv(out/'composition.csv',index=False)
    trace.to_csv(out/'greedy_trace.csv',index=False)
    summary=dict(experiment='no_11_difference_2',seed=a.seed,
      weather_input_difference_order=DIFF_ORDER,
      weather_difference_formulas=['F_t - F_(t-1)'] + (['F_(t+1) - 2*F_t + F_(t-1)'] if DIFF_ORDER>=2 else []),
      weather_model='LightGBM quantile regression, A+B both rounds',
      weather_targets=targets,quantiles=list(QUANTILES),
      station_pairs=len(pairs),weather_model_folds=a.weather_folds,
      weather_feature_input=wx_cols,weather_feature_outputs=qextra,
      weather_max_rows_per_target=a.weather_max_rows,
      generation_model='Weighted-L1 LightGBM; SMP*k*capacity; two rounds averaged',
      generation_input_features=list(base.FEATS)+qextra,
      model_train_days=int((days.split=='model_train').sum()),
      group_opt_days=int((days.split=='group_opt').sum()),
      holdout_days=int((days.split=='holdout').sum()),
      group_opt_missing_site_hours=g['missing'],holdout_missing_site_hours=h['missing'],
      holdout_model_metrics=h['metrics'],group_opt_model_metrics=g['metrics'],
      group_opt_penalty=group_score_result,holdout_penalty=hold_score_result,
      elapsed_seconds=time.monotonic()-started)
    with (out/'summary.json').open('w',encoding='utf-8') as fp:
        json.dump(summary,fp,ensure_ascii=False,indent=2,allow_nan=False)
    log('='*74)
    log('NO.11 + WEATHER DIFFERENCES ORDER 2 | RANDOM DAY 60:20:20')
    log(f'GROUP OPT     initial W       = {group_score_result["W_initial_krw"]:,.2f} KRW')
    log(f'GROUP OPT     optimized W     = {group_score_result["W_optimized_krw"]:,.2f} KRW')
    log(f'GROUP OPT     lower bound     = {group_score_result["W_lower_bound_krw"]:,.2f} KRW')
    log(f'HOLDOUT MODEL weighted MAE     = {h["metrics"]["weighted_MAE_kwh"]:.5f} kWh')
    log(f'HOLDOUT GROUP initial W       = {hold_score_result["W_initial_krw"]:,.2f} KRW')
    log(f'HOLDOUT GROUP optimized W     = {hold_score_result["W_optimized_krw"]:,.2f} KRW')
    log(f'HOLDOUT GROUP lower bound     = {hold_score_result["W_lower_bound_krw"]:,.2f} KRW')
    log(f'Results saved to {out}')


if __name__=='__main__':
    run()
