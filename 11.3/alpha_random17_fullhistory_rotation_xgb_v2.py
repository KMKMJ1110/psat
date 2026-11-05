#!/usr/bin/env python3
"""OIBC Alpha v2: Random Day 60/20/20 + ORIGINAL No.17 + full history
+ lat/lon rotation + SMP*k*capacity weighted-L1 XGBoost; Greedy unchanged.

Put in data/codes/. This script requires only:
  data/baseline.py, data/sites.csv,
  data/train/generation.parquet, data/train/smp.csv,
  data/train/observed_wa.parquet, data/train/observed_wb.parquet,
  data/output/preprocessed/train/forecast_wa.parquet, forecast_wb.parquet,
  data/codes/spatial_weather_validation.py, data/codes/baseline_optimizer.py.
It does NOT require spatial_weather_smp_k_loss.py or any saved No.17 cache.
It never uses test labels or uploads any contest data.

IMPORTANT: Unlike the original Random Split No.17, full-history training requires
new weather OOF corrections and quantile features for pre-2023 data. Hence this
is a new, independently trained combined experiment, not cached No.17 weights.
The original 60/20/20 split of days from 2023-06 onward remains unchanged.

Usage:
  cd /Users/mj/Desktop/PSAT_2/con/data
  python -u codes/alpha_random17_fullhistory_rotation_xgb_v2.py --check-only
  python -u codes/alpha_random17_fullhistory_rotation_xgb_v2.py

Output: output/alpha_random17_fullhistory_rotation_xgb_v2/
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'codes' else HERE
TZ = 'Asia/Seoul'
QUANTILES = (0.05, 0.25, 0.75, 0.95)
SITES_COLS = ['site_id', 'current_group', 'capacity_kw', 'lat', 'lon',
              'wa_station', 'wb_station']
ORIGINAL_NO17_SHA256 = '6e61a36ef44632fcfaffa3e64b64a06bb563afe4e7a9ab1100681aa1b30e3e63'
XGB_PARAMS = dict(objective='reg:absoluteerror', n_estimators=600,
                  learning_rate=0.06, max_depth=0, max_leaves=127,
                  grow_policy='lossguide', tree_method='hist',
                  min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                  random_state=42, n_jobs=-1)


def log(message):
    print('[ALPHA] ' + str(message), flush=True)


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required helper: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_kst(series):
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        return s.dt.tz_localize(TZ).dt.as_unit('ns')
    return s.dt.tz_convert(TZ).dt.as_unit('ns')


def delivery_day(series):
    return (series - pd.Timedelta(hours=1)).dt.normalize()


def quantile_names(col):
    return ([f'wx_{col}_q{int(q*100):02d}' for q in QUANTILES]
            + [f'wx_{col}_width50', f'wx_{col}_width90'])


def input_paths(root):
    return [
        root / 'baseline.py', root / 'sites.csv',
        root / 'train/generation.parquet', root / 'train/smp.csv',
        root / 'train/observed_wa.parquet', root / 'train/observed_wb.parquet',
        root / 'output/preprocessed/train/forecast_wa.parquet',
        root / 'output/preprocessed/train/forecast_wb.parquet',
        HERE / 'spatial_weather_validation.py',
        HERE / 'baseline_optimizer.py',
    ]


def check_paths(root):
    missing = [str(p) for p in input_paths(root) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required inputs missing:\n' + '\n'.join(missing))
    log('Required files are present. No spatial_weather_smp_k_loss.py dependency.')


def random_day_split(gen, train_from, seed):
    """Exact random-day 60/20/20 convention from No.17 (not random site rows)."""
    recent = gen.loc[gen.ts >= train_from].copy()
    historical = gen.loc[gen.ts < train_from].copy()
    days = pd.DatetimeIndex(delivery_day(recent.ts).drop_duplicates().sort_values())
    if len(days) < 5:
        raise ValueError('Insufficient recent training days')
    perm = np.random.default_rng(seed).permutation(len(days))
    n_fit = int(np.floor(.60 * len(days)))
    n_opt = int(np.floor(.20 * len(days)))
    fit_days = days[perm[:n_fit]]
    opt_days = days[perm[n_fit:n_fit+n_opt]]
    hold_days = days[perm[n_fit+n_opt:]]
    recent_day = delivery_day(recent.ts)
    fit = pd.concat([historical, recent.loc[recent_day.isin(fit_days)]], ignore_index=True)
    group = recent.loc[recent_day.isin(opt_days)].copy()
    holdout = recent.loc[recent_day.isin(hold_days)].copy()
    if not (len(fit) + len(group) + len(holdout) == len(gen)):
        raise AssertionError('Split row counts mismatch')
    if set(fit_days) & set(opt_days) or set(fit_days) & set(hold_days) or set(opt_days) & set(hold_days):
        raise AssertionError('Recent day splits overlap')
    split = pd.DataFrame({
        'day_kst': days.strftime('%Y-%m-%d'),
        'split': np.select([days.isin(fit_days), days.isin(opt_days)],
                           ['model_train', 'group_opt'], default='holdout')
    })
    # OOF cross-fitting weather features on ALL generation training dates,
    # including old history; never on group-opt or holdout dates.
    hist_days = pd.DatetimeIndex(delivery_day(historical.ts).drop_duplicates().sort_values())
    ordered_fit_days = pd.DatetimeIndex(list(hist_days) + list(fit_days))
    # Preserve original No.17 seeded OOF fold identity for recent fit days.
    # Historical extra days get new fold identities.
    fold_map = {day: i % 3 for i, day in enumerate(hist_days)}
    fold_map.update({day: i % 3 for i, day in enumerate(fit_days)})
    if len(fold_map) != len(ordered_fit_days):
        raise ValueError('Duplicate historical/model-fit delivery days')
    log(f'Split recent days={len(fit_days)}/{len(opt_days)}/{len(hold_days)} '
        f'(train/group/hold), historical days={len(hist_days)}')
    log(f'Generation rows fit={len(fit):,}, group={len(group):,}, hold={len(holdout):,}')
    return fit, group, holdout, split, fold_map


def weather_stage(root, out, base, graph, fold_map, max_weather_rows, seed):
    """No.17 median graph + 4 graph quantiles, refitted on full-history FIT only.

    For any forecast row belonging to a model-fit day, weather features are
    OOF against its fold. For group-opt/holdout, models fit all model-fit days.
    Both stages only use the forecasted weather at prediction time, never
    observed weather of the evaluation rows for weather-model training.
    """
    outputs = {}
    extras = []
    audit = []
    for prefix, cases in graph.CASES.items():
        station = prefix + '_station'
        original_cols = list(base.WA_COLS if prefix == 'wa' else base.WB_COLS)
        path = root / 'output/preprocessed/train' / f'forecast_{prefix}.parquet'
        existing = pq.read_schema(path).names
        flags = ['imputed_'+c for _, c in cases if 'imputed_'+c in existing]
        forecast = pd.read_parquet(path, columns=[station,'round','ts']+original_cols+flags)
        forecast['ts'] = to_kst(forecast.ts)
        forecast = forecast.loc[forecast['round'].isin((10,17))].reset_index(drop=True)
        keys = [station, 'round', 'ts']
        if forecast.duplicated(keys).any():
            raise ValueError(f'Duplicate forecast keys: {prefix}')
        corrected = forecast[keys+original_cols].copy()
        extra_names = [name for _,col in cases for name in quantile_names(col)]
        for name in extra_names:
            corrected[name] = np.float32(np.nan)
        extras.extend(extra_names)
        obs = graph.read_observed(root, prefix, [c for _,c in cases], type('Tz',(),{'to_kst':staticmethod(to_kst)}))
        obs['ts'] = to_kst(obs.ts)
        station_ids = sorted(forecast[station].dropna().unique())
        log(f'{prefix}: forecast rows={len(forecast):,}, stations={len(station_ids)}')
        for rnd in (10,17):
            row_ids = np.flatnonzero(forecast['round'].to_numpy() == rnd)
            part = forecast.iloc[row_ids].merge(obs, on=[station,'ts'], how='left',
                                                suffixes=('','_obs'), validate='many_to_one')
            if len(part) != len(row_ids):
                raise AssertionError('Station-hour join changed rows')
            days = delivery_day(part.ts)
            fold = days.map(fold_map).fillna(-1).to_numpy(dtype=int)
            is_fit = fold >= 0
            for variable,col in cases:
                frame = pd.DataFrame({
                    'station':part[station], 'ts':part.ts,
                    'raw':part[col].where(graph.valid(part[col], variable)),
                    'obs':part[col+'_obs'].where(graph.valid(part[col+'_obs'], variable))
                })
                frame['paired'] = frame.raw.notna() & frame.obs.notna()
                flag = 'imputed_'+col
                if flag in part:
                    frame['paired'] &= ~part[flag].fillna(False).astype(bool)
                raw = frame.raw.to_numpy(dtype=np.float64)
                medians = raw.copy()
                quantiles = np.full((len(part),4),np.nan, dtype=np.float32)
                for f in (0,1,2,-1):
                    train_mask = is_fit & (fold != f) if f >= 0 else is_fit
                    pred_mask = (fold == f) if f >= 0 else ~is_fit
                    # Non-fit rows can contain forecast-only hours; that's fine.
                    target = np.flatnonzero(pred_mask & np.isfinite(raw))
                    if not len(target):
                        continue
                    candidate = np.flatnonzero(train_mask & frame.paired.to_numpy())
                    if len(candidate) < 1000:
                        raise ValueError(f'Too few weather pairs {prefix}/{col}/round{rnd}/fold{f}')
                    weights, _, _ = graph.error_graph(frame, train_mask, station_ids)
                    _, x = graph.design(frame, station_ids, weights)
                    if len(candidate) > max_weather_rows:
                        rng = np.random.default_rng(seed+f+1)
                        candidate = np.sort(rng.choice(candidate, max_weather_rows, replace=False))
                    y = frame.obs.to_numpy(dtype=np.float64)[candidate] - raw[candidate]
                    median_model = lgb.LGBMRegressor(**graph.WEATHER_PARAMS)
                    median_model.fit(x.iloc[candidate], y, categorical_feature=['station_code'])
                    pred = raw[target] + median_model.predict(x.iloc[target])
                    medians[target] = pred if variable == 'temperature' else np.clip(pred,0,100)
                    del median_model
                    for j,q in enumerate(QUANTILES):
                        params = {**graph.WEATHER_PARAMS,'objective':'quantile','alpha':q}
                        mdl = lgb.LGBMRegressor(**params)
                        mdl.fit(x.iloc[candidate],y,categorical_feature=['station_code'])
                        val = raw[target] + mdl.predict(x.iloc[target])
                        quantiles[target,j] = val if variable == 'temperature' else np.clip(val,0,100)
                        del mdl
                    audit.append({'provider':prefix,'variable':variable,'round':rnd,
                                  'fold':f,'fit_weather_pairs':len(candidate),
                                  'target_forecasts':len(target)})
                    log(f'Weather {prefix}/{variable}/round{rnd}/fold{f}: '
                        f'fit={len(candidate):,}, predicted={len(target):,}')
                    del weights,x,y
                    gc.collect()
                lower = np.minimum(np.sort(quantiles[:,:2],axis=1),medians[:,None])
                upper = np.maximum(np.sort(quantiles[:,2:],axis=1),medians[:,None])
                ordered = np.column_stack([lower,upper])
                values = np.column_stack([ordered, ordered[:,2]-ordered[:,1],
                                          ordered[:,3]-ordered[:,0]]).astype('float32')
                corrected.loc[row_ids,col] = medians.astype('float32')
                for j,name in enumerate(quantile_names(col)):
                    corrected.loc[row_ids,name] = values[:,j]
                del frame,medians,quantiles,ordered,values
                gc.collect()
            del part
            gc.collect()
        file = out/f'weather_{prefix}_median_quantiles.parquet'
        corrected.to_parquet(file,index=False,compression='zstd')
        outputs[prefix] = file
        log(f'Saved {file.name}')
        del forecast,obs,corrected
        gc.collect()
    pd.DataFrame(audit).to_csv(out/'weather_crossfit_audit.csv',index=False)
    return outputs, extras


def load_corrected_weather(paths):
    corrected = {}
    for prefix,path in paths.items():
        d = pd.read_parquet(path)
        d['ts'] = to_kst(d.ts)
        corrected[prefix] = d
    return corrected


def train_weights(df, smp_by_ts):
    """Source No.17: (capacity * SMP * k), normalized to mean=1."""
    price = df.ts.map(smp_by_ts)
    if price.isna().any():
        # Entire Period requires historical SMP at every training timestamp.
        # Do not silently invent historical prices or substitute another loss.
        missing = df.loc[price.isna(), 'ts']
        raise ValueError(
            f'Missing SMP for {len(missing):,} training rows, '
            f'from {missing.min()} to {missing.max()}. '
            'Entire Period SMP*k*capacity Weighted-L1 cannot be reproduced '
            'without historical price coverage. No proxy prices were generated.')
    cap = df.cap.to_numpy(dtype=np.float64)
    smp = price.to_numpy(dtype=np.float64)
    k = np.where(df.ts.dt.hour.between(11,15),3.,1.)
    raw = cap * smp * k
    if not np.isfinite(raw).all() or (raw < 0).any() or raw.mean() <= 0:
        raise ValueError('Nonfinite/negative/zero training weights')
    return (raw/raw.mean()).astype('float32')


def make_matrix(base, original, sites, corrected, r, extras):
    # Let baseline features join forecast weather columns. Extra columns travel
    # through the merged weather data frames, but base.features only computes
    # solar/time/geo features from them. The extra columns are retained.
    fa, fb = corrected['wa'],corrected['wb']
    d = base.features(original, sites, fa, fb, None, r)
    if len(d) != len(original):
        raise ValueError(f'PV feature join changed row count (round={r})')
    if d.duplicated(['site_id','ts']).any():
        raise ValueError('Duplicate site-hour after feature join')
    # Spatial boundary diagonalization: preserve lat/lon and add these two.
    d['lat_lon_sum'] = (d.lat+d.lon).astype('float32')
    d['lat_lon_diff'] = (d.lat-d.lon).astype('float32')
    feats = list(base.FEATS) + list(extras) + ['lat_lon_sum','lat_lon_diff']
    missing = [c for c in feats if c not in d]
    if missing:
        raise ValueError(f'Missing required PV features: {missing}')
    return d,feats


def predict_sites(base, models, sites, corrected, actual, extras):
    # Prediction of group-opt/holdout rows uses only weather forecasts.
    merged = None
    for r in (10,17):
        d, feats = make_matrix(base, actual, sites, corrected, r, extras)
        x = d[feats].astype('float32')
        p = models[r].predict(x)
        kwh = np.maximum(np.where(d.elev.to_numpy()>0,p,0.),0.)*d.cap.to_numpy()
        one = d[['site_id','ts']].copy()
        one[f'r{r}'] = kwh
        merged = one if merged is None else merged.merge(one,on=['site_id','ts'],how='inner',validate='one_to_one')
        del d,feats,x,p,one
        gc.collect()
    if len(merged)!=len(actual):
        raise ValueError('Incomplete averaged two-round predictions')
    merged['pred_kwh'] = merged[['r10','r17']].mean(axis=1)
    if merged.pred_kwh.isna().any():
        raise ValueError('Non-finite predictions')
    return merged[['site_id','ts','pred_kwh']]


def metrics(root, actual, pred):
    frame = actual[['site_id','ts','energy']].merge(pred,on=['site_id','ts'],how='left',validate='one_to_one')
    if frame.pred_kwh.isna().any() or len(frame)!=len(actual):
        raise ValueError('Unmatched predictions')
    smp = pd.read_csv(root/'train/smp.csv',usecols=['ts','smp'])
    smp['ts'] = to_kst(smp.ts)
    frame = frame.merge(smp,on='ts',how='left',validate='many_to_one')
    if frame.smp.isna().any():
        raise ValueError('Missing SMP for holdout scoring')
    error = frame.pred_kwh.to_numpy(dtype=float)-frame.energy.to_numpy(dtype=float)
    weights = frame.smp.to_numpy(dtype=float)*np.where(frame.ts.dt.hour.between(11,15),3.,1.)
    return {'MAE_kwh':float(np.abs(error).mean()),
            'RMSE_kwh':float(np.sqrt(np.mean(error**2))),
            'weighted_MAE_kwh':float(np.dot(weights,np.abs(error))/weights.sum()),
            'n_observed_site_hours':int(len(frame))}


def residuals(actual, pred, site_ids):
    df = actual[['site_id','ts','energy']].merge(pred,on=['site_id','ts'],how='left',validate='one_to_one')
    if df.pred_kwh.isna().any():
        raise ValueError('Missing site predictions for residual matrix')
    df['error'] = df.pred_kwh-df.energy
    matrix = df.pivot(index='ts',columns='site_id',values='error')
    matrix = matrix.reindex(columns=list(site_ids)).sort_index()
    missing = int(matrix.isna().sum().sum())
    log(f'Residuals: {len(matrix):,} hours, {missing:,} missing site-hours filled with 0')
    return matrix.fillna(0).to_numpy(dtype=np.float64), pd.DatetimeIndex(matrix.index), missing


def group_scores(opt, root, matrix, times, initial, final):
    weights = opt.time_weights(root,times)
    old = float(opt.penalty(matrix,weights,initial))
    new = float(opt.penalty(matrix,weights,final))
    lower = float(opt.PENALTY_RATE*np.dot(weights,np.abs(matrix.sum(axis=1))))
    return {'W_initial_krw':old,'W_optimized_krw':new,'W_lower_bound_krw':lower,
            'improvable_cost':new-lower,'improved_percentage':100*(old-new)/(old-lower) if old>lower else None,
            'hours':int(len(times))}


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir',type=Path,default=ROOT)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max-weather-rows',type=int,default=50000)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--max-iterations',type=int,default=0)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    root=args.data_dir.expanduser().resolve()
    check_paths(root)
    if args.check_only:
        # Check the helper APIs without opening contest data.
        graph=load_module(HERE/'spatial_weather_validation.py','alpha_weather_graph')
        opt=load_module(HERE/'baseline_optimizer.py','alpha_greedy')
        assert callable(graph.error_graph) and callable(graph.design)
        assert callable(graph.read_observed) and callable(opt.greedy_best_improvement)
        try:
            import xgboost
            from packaging.version import Version
            if Version(xgboost.__version__) < Version('2.0'):
                raise RuntimeError('XGBoost objective reg:absoluteerror requires xgboost>=2.0')
            log(f'XGBoost {xgboost.__version__} supports reg:absoluteerror.')
        except ImportError:
            raise RuntimeError('Install xgboost>=2.0 (and packaging).')
        log('Helper APIs OK; ready to attempt full execution. NOT yet tested against data.')
        return
    if args.max_weather_rows<1000 or args.batch_size<1 or args.max_iterations<0:
        ap.error('Invalid weather/Greedy settings')
    if args.seed != 42:
        ap.error('Alpha uses the original No.17 seed=42; do not alter it')
    from packaging.version import Version
    import xgboost
    if Version(xgboost.__version__) < Version('2.0'):
        raise RuntimeError('xgboost>=2.0 is required for objective reg:absoluteerror')
    started=time.monotonic()
    out=root/'output/alpha_random17_fullhistory_rotation_xgb_v2'
    out.mkdir(parents=True,exist_ok=True)
    base=load_module(root/'baseline.py','alpha_baseline')
    graph=load_module(HERE/'spatial_weather_validation.py','alpha_weather_graph')
    opt=load_module(HERE/'baseline_optimizer.py','alpha_greedy')
    # No.17 uses the original validation's graph functions, but we re-fit all
    # weather models for Entire Period (the cached graph includes only recent data).
    sites=pd.read_csv(root/'sites.csv')[SITES_COLS]
    if len(sites)!=253 or sites.site_id.duplicated().any():
        raise ValueError('Expected 253 distinct sites')
    gen=pd.read_parquet(root/'train/generation.parquet',columns=['site_id','ts','energy'])
    gen['ts']=to_kst(gen.ts)
    if gen.duplicated(['site_id','ts']).any() or not np.isfinite(gen.energy).all():
        raise ValueError('Bad generation labels')
    fit,group_actual,hold_actual,split,fold_map=random_day_split(gen,base.TRAIN_FROM,args.seed)
    # Confirm that the current Random Day assignment reproduces No.17 exactly.
    original_split_path = root/'output/weather_error_graph/split_days.csv'
    if original_split_path.is_file():
        original_split = pd.read_csv(original_split_path)
        pd.testing.assert_frame_equal(split.reset_index(drop=True),
                                      original_split.reset_index(drop=True))
        log('Original No.17 random-day split verified against saved split_days.csv')
    else:
        log('Original split_days.csv absent; using exact seed-42 original split algorithm')
    split.to_csv(out/'split_days.csv',index=False)
    del gen
    gc.collect()
    # This is a time-agnostic RANDOM validation: both older historical days
    # and recent (60%) training days are eligible to fit the weather models.
    # 20% group and 20% independent holdout observed weather never enter fitting.
    weather_files, extras=weather_stage(root,out,base,graph,fold_map,args.max_weather_rows,args.seed)
    corrected=load_corrected_weather(weather_files)
    smp=pd.read_csv(root/'train/smp.csv',usecols=['ts','smp'])
    smp['ts']=to_kst(smp.ts)
    if smp.ts.duplicated().any():
        raise ValueError('Duplicate SMP timestamps')
    smp_by_ts=pd.to_numeric(smp.set_index('ts').smp,errors='coerce')
    models={}
    xgb_params=dict(XGB_PARAMS)
    for r in (10,17):
        d,feats=make_matrix(base,fit,sites,corrected,r,extras)
        daylight=d.elev>0
        d=d.loc[daylight].copy()
        x=d[feats].astype('float32')
        y=(d.energy/d.cap).clip(0,1.2).astype('float32')
        w=train_weights(d,smp_by_ts)
        log(f'XGBoost round={r}: training {x.shape}, features={len(feats)}, sample_weight mean={w.mean():.6f}')
        m=XGBRegressor(**xgb_params)
        m.fit(x,y,sample_weight=w)
        models[r]=m
        m.save_model(str(out/f'xgboost_round{r}.json'))
        del d,x,y,w,m
        gc.collect()
    # Evaluate with the exact same held-out (20%) delivery-days as No.17.
    actual=pd.concat([group_actual,hold_actual],ignore_index=True)
    pred=predict_sites(base,models,sites,corrected,actual,extras)
    del models,corrected,actual
    gc.collect()
    grp_days=set(delivery_day(group_actual.ts).unique())
    hold_days=set(delivery_day(hold_actual.ts).unique())
    p_days=delivery_day(pred.ts)
    pred_g=pred.loc[p_days.isin(grp_days)].copy()
    pred_h=pred.loc[p_days.isin(hold_days)].copy()
    if len(pred_g)!=len(group_actual) or len(pred_h)!=len(hold_actual):
        raise AssertionError('Prediction rows do not match group/holdout')
    pred_g.to_parquet(out/'group_opt_site_predictions.parquet',index=False)
    pred_h.to_parquet(out/'holdout_site_predictions.parquet',index=False)
    hold_metrics=metrics(root,hold_actual,pred_h)
    mat_g,time_g,missing_g=residuals(group_actual,pred_g,sites.site_id)
    mat_h,time_h,missing_h=residuals(hold_actual,pred_h,sites.site_id)
    initial=opt.initial_mask(sites)
    opt.check_capacity(initial,sites.capacity_kw.to_numpy(dtype=np.float64))
    weights_g=opt.time_weights(root,time_g)
    optimized,info,trace=opt.greedy_best_improvement(mat_g,weights_g,sites,initial,
                             batch_size=args.batch_size,max_iterations=args.max_iterations)
    gs=group_scores(opt,root,mat_g,time_g,initial,optimized)
    hs=group_scores(opt,root,mat_h,time_h,initial,optimized)
    pd.DataFrame({'site_id':sites.site_id,
                  'group':np.where(optimized,'G1','G2')}).to_csv(out/'composition.csv',index=False)
    trace.to_csv(out/'greedy_trace.csv',index=False)
    result={'experiment':'ALPHA_RandomSplit_No17_EntirePeriod_Rotation_XGBoost',
            'validation':'Random day 60/20/20 seed='+str(args.seed),
            'source_original_no17_sha256':ORIGINAL_NO17_SHA256,
            'train_window':'All available generation dates, with original 60/20/20 group/holdout days excluded; older history included in model fit only',
            'weather_strategy':'Re-fitted source No.17 station graph median correction + 4 quantiles, day 3-fold OOF, fit-only',
            'model':'XGBoost reg:absoluteerror weighted-L1, separate round 10/17, mean predictions',
            'loss_weights':'mean-normalized (SMP*k*capacity)',
            'rotation_features':['lat_lon_sum','lat_lon_diff'],
            'rotation_definition':{'lat_lon_sum':'lat+lon','lat_lon_diff':'lat-lon'},
            'features':list(base.FEATS)+list(extras)+['lat_lon_sum','lat_lon_diff'],
            'group_algorithm':'Original Greedy, no changes',
            'model_metrics_holdout':hold_metrics,
            'group_opt':gs,'group_evaluation_independent_holdout':hs,
            'missing_group_site_hours':missing_g,'missing_hold_site_hours':missing_h,
            'xgboost_params':xgb_params,
            'elapsed_seconds':time.monotonic()-started}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    log(f'HOLDOUT Weighted MAE = {hold_metrics["weighted_MAE_kwh"]:.5f}')
    log(f'HOLDOUT Initial W     = {hs["W_initial_krw"]:,.0f}')
    log(f'HOLDOUT Optimized W   = {hs["W_optimized_krw"]:,.0f}')
    log(f'HOLDOUT Lower Bound   = {hs["W_lower_bound_krw"]:,.0f}')
    log(f'Saved results: {out}')


if __name__ == '__main__':
    main()
