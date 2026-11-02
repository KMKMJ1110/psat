"""Error-graph weather quantiles + existing median correction + weighted-L1 PV.

Spyder: %run codes/spatial_weather_distribution_smp_k.py
Needs spatial_weather_validation.py, spatial_weather_smp_k_loss.py and original
baseline_validation.py / baseline_optimizer.py in codes/. Defaults match the
completed seed-42, three-fold, 50k-row median experiment.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
import importlib.util
import importlib.metadata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'codes' else HERE
QUANTILES = (.05, .25, .75, .95)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extras(col):
    return [f'wx_{col}_q{int(q*100):02d}' for q in QUANTILES] + [f'wx_{col}_width50', f'wx_{col}_width90']


def ordered_bounds(pred, median):
    """Keep the existing median fixed; order lower/upper quantiles around it."""
    lower = np.minimum(np.sort(pred[:, :2], axis=1), median[:, None])
    upper = np.maximum(np.sort(pred[:, 2:], axis=1), median[:, None])
    return np.column_stack([lower, upper])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-iterations', type=int, default=0)
    args = ap.parse_args(argv)
    if args.batch_size < 1 or args.max_iterations < 0:
        ap.error('Invalid Greedy settings')
    root = args.data_dir.expanduser().resolve()
    graph = load(HERE/'spatial_weather_validation.py', 'distribution_graph_helpers')
    weighted = load(HERE/'spatial_weather_smp_k_loss.py', 'distribution_weighted_helpers')
    v = load(HERE/'baseline_validation.py', 'distribution_validation')
    base = load(root/'baseline.py', 'distribution_baseline')
    stage = root/'output/weather_error_graph'
    meta_path = stage/'manifest.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('seed') != 42 or meta.get('folds') != 3 or meta.get('max_weather_rows') != 50000:
        raise ValueError('This comparison requires the original seed=42, folds=3, max_weather_rows=50000 median run.')
    for field, path in [('baseline_sha256', root/'baseline.py'),
                        ('validation_sha256', HERE/'baseline_validation.py'),
                        ('optimizer_sha256', HERE/'baseline_optimizer.py')]:
        if meta.get(field) != graph.sha(path):
            raise ValueError(f'Code differs from median experiment: {path}')
    # Require exact helper code that created the saved OOF medians.
    if meta.get('runner_sha256') != graph.sha(HERE/'spatial_weather_validation.py'):
        raise ValueError('spatial_weather_validation.py differs from the saved median run; restore that version before combining features.')
    split = pd.read_csv(stage/'split_days.csv')
    train_days = pd.DatetimeIndex(pd.to_datetime(split.loc[split.split=='model_train','day_kst']).dt.tz_localize('Asia/Seoul'))
    # Reproduce the original assignment order, not sorted training-day order.
    all_days = pd.DatetimeIndex(pd.to_datetime(split.day_kst).dt.tz_localize('Asia/Seoul'))
    order = np.random.default_rng(42).permutation(len(all_days))
    fit_days = all_days[order[:int(np.floor(.6*len(all_days)))]]
    if set(fit_days) != set(train_days):
        raise ValueError('Saved training days differ from seed-42 canonical split')
    fold_map = {d: i % 3 for i, d in enumerate(fit_days)}
    role_map = dict(zip(all_days, split.split))
    out = root/'output/validation_602020/split602020_weather_graph_distribution_l1_smp_k'
    out.mkdir(parents=True, exist_ok=True)
    qstage = out/'weather_quantiles'
    qstage.mkdir(exist_ok=True)
    metric_rows, audits, tables = [], [], {}
    for prefix, cases in graph.CASES.items():
        station = prefix+'_station'
        cols = list(base.WA_COLS if prefix=='wa' else base.WB_COLS)
        path = root/'output/preprocessed/train'/f'forecast_{prefix}.parquet'
        names = pq.read_schema(path).names
        flags = ['imputed_'+col for _, col in cases if 'imputed_'+col in names]
        forecast = pd.read_parquet(path, columns=[station,'round','ts']+cols+flags)
        forecast['ts'] = v.to_kst(forecast.ts)
        forecast = forecast.loc[(forecast.ts >= base.TRAIN_FROM) & forecast['round'].isin((10,17))].reset_index(drop=True)
        keys = [station,'round','ts']
        if forecast.duplicated(keys).any():
            raise ValueError('Duplicate input forecast keys')
        median = pd.read_parquet(stage/f'graph_forecast_{prefix}.parquet', columns=keys+[c for _,c in cases])
        median['ts'] = v.to_kst(median.ts)
        combined = forecast[keys].merge(median, on=keys, how='left', validate='one_to_one', indicator=True)
        if not combined['_merge'].eq('both').all():
            raise ValueError('Raw/median forecast key mismatch')
        obs = graph.read_observed(root, prefix, [c for _,c in cases], v)
        ids = sorted(forecast[station].unique())
        aux = forecast[keys].copy()
        for r in (10,17):
            row_ids = np.flatnonzero(forecast['round'].to_numpy()==r)
            part = forecast.iloc[row_ids].merge(obs, on=[station,'ts'], how='left', suffixes=('','_obs'), validate='many_to_one')
            day = graph.day_of(part.ts)
            fold = day.map(fold_map).fillna(-1).to_numpy(dtype=int)
            fit = fold >= 0
            roles = day.map(role_map).to_numpy()
            for variable,col in cases:
                label = f'{prefix}_{variable}_round{r}'
                frame = pd.DataFrame({'station':part[station], 'ts':part.ts,
                    'raw':part[col].where(graph.valid(part[col],variable)),
                    'obs':part[col+'_obs'].where(graph.valid(part[col+'_obs'],variable))})
                frame['paired'] = frame.raw.notna() & frame.obs.notna()
                if 'imputed_'+col in part:
                    frame['paired'] &= ~part['imputed_'+col].fillna(False).astype(bool)
                predictions = np.full((len(frame),4),np.nan)
                for f in (0,1,2,-1):
                    train = fit & (fold != f) if f>=0 else fit
                    target = fold==f if f>=0 else ~fit
                    if (train & target).any():
                        raise AssertionError('OOF overlap')
                    weight,_,_ = graph.error_graph(frame,train,ids)
                    _,x = graph.design(frame,ids,weight)
                    candidate = np.flatnonzero(train & frame.paired.to_numpy())
                    if len(candidate)<1000:
                        raise ValueError('Too few weather pairs')
                    rng = np.random.default_rng(42+f+1)
                    if len(candidate)>50000:
                        candidate = np.sort(rng.choice(candidate,50000,replace=False))
                    predict = np.flatnonzero(target & frame.raw.notna().to_numpy())
                    y = (frame.obs-frame.raw).to_numpy()[candidate]
                    graph.log(f'DISTRIBUTION {label} fold={f}: fit {len(candidate):,}, predict {len(predict):,}, quantiles={QUANTILES}')
                    for j,q in enumerate(QUANTILES):
                        params = {**graph.WEATHER_PARAMS, 'objective':'quantile','alpha':q}
                        model = lgb.LGBMRegressor(**params)
                        model.fit(x.iloc[candidate],y,categorical_feature=['station_code'])
                        if len(predict):
                            p = frame.raw.to_numpy()[predict]+model.predict(x.iloc[predict])
                            predictions[predict,j] = p if variable=='temperature' else np.clip(p,0,100)
                        if f==-1:
                            (qstage/f'{label}_q{int(q*100):02d}.txt').write_text(model.booster_.model_to_string(),encoding='utf-8')
                        del model
                    audits.append({'case':label,'fold':f,'fit_rows':len(candidate),'predict_rows':len(predict),
                                   'nonzero_edges':int((weight>0).sum())})
                    del x
                    gc.collect()
                center = combined.iloc[row_ids][col].to_numpy(dtype=float)
                predictions = ordered_bounds(predictions,center)
                values = np.column_stack([predictions,predictions[:,2]-predictions[:,1],predictions[:,3]-predictions[:,0]])
                for j,name in enumerate(extras(col)):
                    if name not in aux:
                        aux[name]=np.nan
                    aux.loc[row_ids,name]=values[:,j].astype('float32')
                for role in ('model_train','group_opt','holdout'):
                    ix = (roles==role) & frame.paired.to_numpy() & np.isfinite(predictions).all(axis=1) & np.isfinite(center)
                    if not ix.any():
                        continue
                    observed = frame.obs.to_numpy()[ix]
                    for j,q in enumerate(QUANTILES):
                        delta = observed-predictions[ix,j]
                        metric_rows.append({'case':label,'split':role,'metric':f'pinball_q{int(q*100):02d}',
                                            'value':float(np.maximum(q*delta,(q-1)*delta).mean()),'n':int(ix.sum())})
                    for a,b,nominal in ((1,2,50),(0,3,90)):
                        coverage = ((observed>=predictions[ix,a]) & (observed<=predictions[ix,b])).mean()
                        metric_rows.append({'case':label,'split':role,'metric':f'coverage_{nominal}',
                                            'value':float(coverage),'n':int(ix.sum())})
                    metric_rows.append({'case':label,'split':role,'metric':'median_MAE',
                                        'value':float(np.abs(observed-center[ix]).mean()),'n':int(ix.sum())})
                pd.DataFrame(metric_rows).to_csv(out/'weather_distribution_metrics.csv',index=False)
                pd.DataFrame(audits).to_csv(out/'quantile_crossfit_audit.csv',index=False)
        aux.to_parquet(qstage/f'quantile_features_{prefix}.parquet',index=False)
        tables[prefix] = aux
        del forecast,median,combined,obs,part
        gc.collect()

    original_features = base.features
    additional = [name for cases in graph.CASES.values() for _,col in cases for name in extras(col)]
    base.FEATS = list(base.FEATS)+additional
    def features(*pos, **kw):
        r = pos[5] if len(pos)>5 else kw['r']
        d = original_features(*pos,**kw)
        count = len(d)
        for prefix,aux in tables.items():
            sub = aux.loc[aux['round']==r].drop(columns='round')
            d = d.merge(sub,on=[prefix+'_station','ts'],how='left',validate='many_to_one')
        if len(d)!=count:
            raise ValueError('Quantile join changed site-hour row count')
        # Missing original weather can also leave quantiles missing; LightGBM handles NA.
        return d
    base.features = features

    # Reuse canonical validation and the tested sample-weight function.
    original_load = v.load_module
    v.load_module = lambda path,name: base if Path(path).resolve()==(root/'baseline.py').resolve() else original_load(path,name)
    v.RUN_NAMES['baseline'] = out.name
    original_split = v.split_602020
    def checked_split(gen,seed):
        result = original_split(gen,seed)
        pd.testing.assert_frame_equal(result[3],split)
        return result
    v.split_602020 = checked_split
    def load_corrected(data_root, baseline, minimum):
        frames=[]
        for prefix,cols in [('wa',base.WA_COLS),('wb',base.WB_COLS)]:
            d=pd.read_parquet(stage/f'graph_forecast_{prefix}.parquet',columns=[prefix+'_station','round','ts']+list(cols))
            d['ts']=v.to_kst(d.ts)
            frames.append(d.loc[d.ts>=minimum].copy())
        return tuple(frames)
    v.load_weather=load_corrected
    def fit_generation(baseline,sites,fit,fa,fb,variant):
        prices=pd.read_csv(root/'train/smp.csv',usecols=['ts','smp'])
        prices['ts']=v.to_kst(prices.ts)
        if prices.ts.duplicated().any():
            raise ValueError('Duplicate SMP timestamps')
        prices=pd.to_numeric(prices.set_index('ts').smp,errors='coerce')
        params={**v.BASELINE_PARAMS,'objective':'regression_l1'}
        models={};audit=[]
        for r in (10,17):
            d=base.features(fit,sites,fa,fb,None,r)
            if len(d)!=len(fit):
                raise ValueError('Generation join changed row count')
            d=d.loc[d.elev>0]
            x=d[base.FEATS].astype('float32')
            y=(d.energy/d.cap).clip(0,1.2).astype('float32')
            w,raw=weighted.training_weights(d,prices)
            graph.log(f'Generation weighted L1 round={r}: X={x.shape}; mean weight={w.mean():.6f}')
            model=lgb.LGBMRegressor(**params)
            model.fit(x,y,sample_weight=w)
            (out/f'generation_round{r}.txt').write_text(model.booster_.model_to_string(),encoding='utf-8')
            models[r]=model
            audit.append({'round':r,'fit_rows':len(d),'weight_mean':float(w.mean()),
                          'raw_weight_mean':float(raw.mean()),'zero_weights':int((w==0).sum())})
            del d,x,y,w,raw
            gc.collect()
        pd.DataFrame(audit).to_csv(out/'training_weight_audit.csv',index=False)
        return models,(10,17),params
    v.fit_models=fit_generation
    v.run('baseline',['--data-dir',str(root),'--seed','42','--batch-size',str(args.batch_size),
                             '--max-iterations',str(args.max_iterations)])
    summary_path = out/'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary.update({'experiment':'weather_graph_distribution_l1_smp_k',
                    'distribution_added':True,'weather_quantiles':list(QUANTILES),
                    'training_loss':'mean-normalized SMP*k*capacity weighted L1',
                    'sample_weight_definition':'SMP*k*capacity / mean(SMP*k*capacity)',
                    'weather_manifest':str(meta_path),
                    'weather_folds':3,'weather_max_rows_per_fold':50000,
                    'weather_quantile_parameters':{**graph.WEATHER_PARAMS,'objective':'quantile'},
                    'packages':{p:importlib.metadata.version(p) for p in ('numpy','pandas','lightgbm','pyarrow','scikit-learn')},
                    'source_weather_manifest_sha256':graph.sha(meta_path),
                    'median_preserved':True,'generation_features':base.FEATS,
                    'quantile_features':additional,'quantile_models_refitted':True,
                    'weather_source':'saved graph median correction + newly cross-fitted graph quantiles',
                    'weather_distribution_note':'conditional quantile estimates; nominal intervals are not calibrated coverage guarantees',
                    'distribution_runner_sha256':graph.sha(Path(__file__))})
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    # Build a comparison including the current experiment (the helper covers prior runs).
    rows=[]
    for name in ['split602020_baseline','split602020_weather_median_graph',
                 'split602020_weather_median_graph_l1_smp_k',out.name]:
        p=root/'output/validation_602020'/name
        if not (p/'summary.json').is_file():
            continue
        s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
        if s.get('seed')!=42:
            continue
        pd.testing.assert_frame_equal(pd.read_csv(p/'split_days.csv'),split)
        g=s['group_evaluation_independent_holdout'];i=g['W_initial_krw'];o=g['W_optimized_krw'];lo=g['W_lower_bound_krw']
        rows.append({'Name':name,'Weighted MAE':s['model_metrics_holdout']['weighted_MAE_kwh'],
                     'Initial Penalty':i,'Optimized Penalty':o,'Theoretical Lower Bound':lo,
                     'Improvable Cost':o-lo,'Improved Percentage':100*(i-o)/(i-lo) if i>lo else None})
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
    graph.log(f'DONE: {out}')


if __name__=='__main__':
    main()
