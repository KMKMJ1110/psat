"""Full-history OOF median/quantiles -> weather94 -> LGBM/CatBoost ensemble.
Copy this file and weather94_engine.py to data/codes. No submission creation.
Spyder: %run codes/validate_weather94_full_history.py --split random
"""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import weather94_engine as engine

QUANTILES = (.05, .25, .75, .95)


def extras(col):
    return [f'wx_{col}_q{int(q * 100):02d}' for q in QUANTILES] + [f'wx_{col}_width50', f'wx_{col}_width90']


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.parquet')
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def prepare(root, stage):
    """New cache only. Weather fit/graph construction never use evaluation-day obs."""
    import lightgbm as lgb
    base = engine.load(root / 'baseline.py', 'full94_base')
    v = engine.load(root / 'codes/baseline_validation.py', 'full94_val')
    graph = engine.load(root / 'codes/spatial_weather_validation.py', 'full94_graph')
    stage.mkdir(parents=True, exist_ok=True)
    files = [root / 'baseline.py', root / 'sites.csv', root / 'train/generation.parquet',
             root / 'codes/baseline_validation.py', root / 'codes/baseline_optimizer.py',
             root / 'codes/spatial_weather_validation.py', Path(__file__), Path(engine.__file__)]
    for p in ('wa', 'wb'):
        files += [root / 'train' / f'observed_{p}.parquet', root / 'output/preprocessed/train' / f'forecast_{p}.parquet']
    inputs = []
    for path in files:
        st = path.stat()
        inputs.append(dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                           sha256=engine.sha(path) if path.suffix == '.py' else None))
    manifest = dict(inputs=inputs, seed=42, folds=3, max_weather_rows=50000,
                    weather_params=graph.WEATHER_PARAMS, quantiles=list(QUANTILES),
                    old_history='all generation before baseline.TRAIN_FROM added to model_train only',
                    packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'lightgbm', 'pyarrow']})
    manifest_path = stage / 'cache_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Full-history weather cache inputs/code changed; choose NEW --output-dir')
    engine.save_json(manifest_path, manifest)
    gen = pd.read_parquet(root / 'train/generation.parquet', columns=['ts'])
    gen['ts'] = v.to_kst(gen.ts)
    minimum, maximum = gen.ts.min(), gen.ts.max()
    days = pd.DatetimeIndex(v.days_of(gen.loc[gen.ts >= base.TRAIN_FROM, 'ts']).drop_duplicates().sort_values())
    order = np.random.default_rng(42).permutation(len(days))
    nfit, nopt = int(np.floor(.6 * len(days))), int(np.floor(.2 * len(days)))
    recent_fit = days[order[:nfit]]
    recent_opt = days[order[nfit:nfit + nopt]]
    split = pd.DataFrame({'day_kst': days.strftime('%Y-%m-%d'),
                          'split': np.select([days.isin(recent_fit), days.isin(recent_opt)], ['model_train', 'group_opt'], default='holdout')})
    original_split = root / 'output/validation_602020/split602020_weather_graph_distribution_l1_smp_k/split_days.csv'
    if original_split.exists():
        pd.testing.assert_frame_equal(split, pd.read_csv(original_split))
    old_days = pd.DatetimeIndex(v.days_of(gen.loc[gen.ts < base.TRAIN_FROM, 'ts']).drop_duplicates().sort_values())
    all_fit_days = recent_fit.union(old_days).sort_values()
    eval_days = days[~days.isin(recent_fit)]
    if len(all_fit_days.intersection(eval_days)):
        raise ValueError('Historical and evaluation delivery days overlap')
    shuffled = all_fit_days[np.random.default_rng(42).permutation(len(all_fit_days))]
    fold_map = {d: i % 3 for i, d in enumerate(shuffled)}
    roles = dict(zip(pd.to_datetime(split.day_kst).dt.tz_localize('Asia/Seoul'), split.split))
    roles.update({d: 'model_train' for d in old_days})
    pd.DataFrame({'day_kst': all_fit_days.strftime('%Y-%m-%d'),
                  'fold': [fold_map[d] for d in all_fit_days]}).to_csv(stage / 'training_weather_folds.csv', index=False)
    del gen
    graph_dir, reference = stage / 'graph', stage / 'reference'
    graph_dir.mkdir(exist_ok=True)
    (reference / 'weather_quantiles').mkdir(parents=True, exist_ok=True)
    # Preserve mtimes: generation checkpoints fingerprint these files on resume.
    for path in [graph_dir / 'split_days.csv', reference / 'split_days.csv']:
        if path.exists():
            pd.testing.assert_frame_equal(split, pd.read_csv(path))
        else:
            split.to_csv(path, index=False)
    complete = stage / 'complete.json'
    expected = [graph_dir / f'graph_forecast_{p}.parquet' for p in ('wa', 'wb')]
    expected += [reference / 'weather_quantiles' / f'quantile_features_{p}.parquet' for p in ('wa', 'wb')]
    expected += [reference / 'summary.json', graph_dir / 'manifest.json']
    if complete.exists() and all(path.exists() for path in expected):
        print('FULL HISTORY WEATHER CACHE COMPLETE: REUSE', stage, flush=True)
        return graph_dir, reference
    audits = []
    for p, cases in graph.CASES.items():
        station = p + '_station'
        cols = list(base.WA_COLS if p == 'wa' else base.WB_COLS)
        weather_path = root / 'output/preprocessed/train' / f'forecast_{p}.parquet'
        schema = pq.read_schema(weather_path).names
        flags = ['imputed_' + col for _, col in cases if 'imputed_' + col in schema]
        observed = graph.read_observed(root, p, [col for _, col in cases], v)
        assembled_corrected, assembled_aux = [], []
        for r in (10, 17):
            forecast = pd.read_parquet(weather_path, columns=[station, 'round', 'ts'] + cols + flags,
                                       filters=[('round', '=', r)])
            forecast['ts'] = v.to_kst(forecast.ts)
            forecast = forecast.loc[forecast.ts.between(minimum, maximum)].reset_index(drop=True)
            engine.require_unique(forecast, [station, 'round', 'ts'], 'full-history forecast')
            if forecast.empty:
                raise ValueError('Empty forecast: ' + p + ' round ' + str(r))
            part = forecast.merge(observed, on=[station, 'ts'], how='left', suffixes=('', '_obs'), validate='many_to_one')
            day = graph.day_of(part.ts)
            folds = day.map(fold_map).fillna(-1).to_numpy(dtype=int)
            is_fit = folds >= 0
            ids = sorted(part[station].unique())
            corrected = forecast[[station, 'round', 'ts'] + cols].copy()
            aux = forecast[[station, 'round', 'ts']].copy()
            for variable, col in cases:
                label = f'{p}_{variable}_round{r}'
                casepath = stage / 'cases' / f'{label}.parquet'
                if casepath.exists():
                    cached = pd.read_parquet(casepath)
                    engine.require_unique(cached, [station, 'round', 'ts'], label)
                    joined = forecast[[station, 'round', 'ts']].merge(cached, how='left', on=[station, 'round', 'ts'], validate='one_to_one', indicator=True)
                    if len(cached) != len(forecast) or not joined['_merge'].eq('both').all():
                        raise ValueError('Case checkpoint keys differ: ' + label)
                    corrected[col] = joined['median'].to_numpy()
                    for name in extras(col):
                        aux[name] = joined[name].to_numpy()
                    print('RESUME WEATHER CASE', label, flush=True)
                    continue
                frame = pd.DataFrame({'station': part[station], 'ts': part.ts,
                                      'raw': part[col].where(graph.valid(part[col], variable)),
                                      'obs': part[col + '_obs'].where(graph.valid(part[col + '_obs'], variable))})
                frame['paired'] = frame.raw.notna() & frame.obs.notna()
                flag = 'imputed_' + col
                if flag in part:
                    frame['paired'] &= ~part[flag].fillna(False).astype(bool)
                # Match the previous median pipeline: retain raw values where raw
                # is outside correction bounds; quantiles there remain missing.
                median = part[col].to_numpy(dtype=float).copy()
                quantile = np.full((len(frame), 4), np.nan)
                for f in (0, 1, 2, -1):
                    train = is_fit & (folds != f) if f >= 0 else is_fit
                    target = folds == f if f >= 0 else ~is_fit
                    if np.any(train & target) or day.loc[train].isin(eval_days).any():
                        raise ValueError('Weather graph/training contains evaluation dates')
                    # This correlation graph, centering and model all use the same fit-only partition.
                    weight, corr, overlap = graph.error_graph(frame, train, ids)
                    own, x = graph.design(frame, ids, weight)
                    del own
                    candidate = np.flatnonzero(train & frame.paired.to_numpy())
                    if len(candidate) < 1000:
                        raise ValueError('Too few weather pairs: ' + label + ' fold ' + str(f))
                    if len(candidate) > 50000:
                        candidate = np.sort(np.random.default_rng(42 + f + 1).choice(candidate, 50000, replace=False))
                    target_ids = np.flatnonzero(target & frame.raw.notna().to_numpy())
                    y = (frame.obs - frame.raw).to_numpy()[candidate]
                    print('WEATHER', label, 'fold', f, 'fit', len(candidate), 'predict', len(target_ids), flush=True)
                    # Median and quantiles share rows/graph, and differ only by objective.
                    for j, q in enumerate((.5,) + QUANTILES):
                        params = dict(graph.WEATHER_PARAMS)
                        if j:
                            params.update(objective='quantile', alpha=q)
                        model = lgb.LGBMRegressor(**params)
                        model.fit(x.iloc[candidate], y, categorical_feature=['station_code'])
                        if len(target_ids):
                            values = frame.raw.to_numpy()[target_ids] + model.predict(x.iloc[target_ids])
                            if variable != 'temperature':
                                values = np.clip(values, 0, 100)
                            if j == 0:
                                median[target_ids] = values
                            else:
                                quantile[target_ids, j - 1] = values
                        del model
                    audit = dict(case=label, fold=f, fit_pairs=len(candidate), target_rows=len(target_ids),
                                 fit_days=int(day.loc[train].nunique()), target_days=int(day.loc[target].nunique()),
                                 evaluation_days_in_fit=0, edges=int((weight > 0).sum()))
                    audits.append(audit)
                    (stage / 'audits').mkdir(exist_ok=True)
                    engine.save_json(stage / 'audits' / f'{label}_fold{f}.json', audit)
                    if f == -1:
                        pd.DataFrame(weight, index=ids, columns=ids).to_csv(stage / 'audits' / f'{label}_weights.csv')
                    del x, weight, corr, overlap
                    gc.collect()
                lower = np.minimum(np.sort(quantile[:, :2], axis=1), median[:, None])
                upper = np.maximum(np.sort(quantile[:, 2:], axis=1), median[:, None])
                quantile = np.column_stack([lower, upper])
                values = np.column_stack([quantile, quantile[:, 2] - quantile[:, 1], quantile[:, 3] - quantile[:, 0]])
                cached = forecast[[station, 'round', 'ts']].copy()
                cached['median'] = median.astype('float32')
                corrected[col] = cached['median']
                for j, name in enumerate(extras(col)):
                    cached[name] = values[:, j].astype('float32')
                    aux[name] = cached[name]
                write_parquet(cached, casepath)
                print('SAVED WEATHER CASE', label, flush=True)
                del frame, quantile, median, values, cached
                gc.collect()
            assembled_corrected.append(corrected)
            assembled_aux.append(aux)
            del forecast, part, corrected, aux
            gc.collect()
        write_parquet(pd.concat(assembled_corrected, ignore_index=True), graph_dir / f'graph_forecast_{p}.parquet')
        write_parquet(pd.concat(assembled_aux, ignore_index=True), reference / 'weather_quantiles' / f'quantile_features_{p}.parquet')
        del observed, assembled_corrected, assembled_aux
        gc.collect()
    metadata = dict(seed=42, folds=3, max_weather_rows=50000,
                    baseline_sha256=engine.sha(root / 'baseline.py'),
                    validation_sha256=engine.sha(root / 'codes/baseline_validation.py'),
                    optimizer_sha256=engine.sha(root / 'codes/baseline_optimizer.py'))
    engine.save_json(graph_dir / 'manifest.json', metadata)
    features60 = list(base.FEATS) + [name for cases in graph.CASES.values() for _, col in cases for name in extras(col)]
    if len(features60) != 60:
        raise ValueError('Expected original baseline24 + weather36')
    params = {**v.BASELINE_PARAMS, 'objective': 'regression_l1'}
    engine.save_json(reference / 'summary.json', dict(seed=42, historical_training_added=True,
                     generation_features=features60, model_params=params))
    engine.save_json(complete, dict(complete=True, fit_days=len(all_fit_days), historical_days=len(old_days),
                     evaluation_days=len(eval_days), minimum=str(minimum), maximum=str(maximum)))
    return graph_dir, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['random'], default='random')
    parser.add_argument('--data-dir', type=Path, default=engine.DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-site', action='store_true')
    parser.add_argument('--no-time', action='store_true')
    parser.add_argument('--no-nat', action='store_true')
    parser.add_argument('--chunk-hours', type=int, default=168)
    args = parser.parse_args()
    # Fail before the long weather stage if either learner is missing.
    import lightgbm
    import catboost
    root = args.data_dir.resolve()
    name = 'random_full' + ('_notime' if args.no_time else '') + ('_nonat' if args.no_nat else '') + ('_nosite' if args.no_site else '')
    output = args.output_dir.resolve() if args.output_dir else root / 'output/validation_weather94_full_history' / name
    # Validate before creating or writing anything in the chosen destination.
    protected = [root / p for p in ['train', 'test', 'codes', 'output/preprocessed', 'output/weather_error_graph', 'output/validation_602020']]
    if output == root or any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Choose a separate full-history output directory')
    stage = output / 'weather_full_history'
    graph, reference = prepare(root, stage)
    engine.main(dict(graph=str(graph), reference=str(reference), output=str(output / 'generation'),
                     cache_manifest=str(stage / 'cache_manifest.json')))


if __name__ == '__main__':
    main()
