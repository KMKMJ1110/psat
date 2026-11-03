"""Validation only. Existing OOF weather66 + raw time19/national8 + categorical site1.
Spyder: %run codes/validate_weather94_ensemble.py --split random
No training before baseline.TRAIN_FROM: old weather OOF caches do not cover it.
No weather retraining, B-time shift, submission or automatic model replacement.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

A_COLS = ['wa_cloud', 'wa_rain', 'wa_humidity']
B_COLS = ['wb_cloud', 'wb_uv_idx', 'wb_total_liq']
NAT = ['a_nat_cloud_mean', 'a_nat_cloud_std', 'a_nat_rain_mean', 'a_cloud_dev',
       'b_nat_cloud_mean', 'b_nat_cloud_std', 'b_nat_uv_mean', 'b_cloud_dev']
TIME = [c + suffix for c in A_COLS + B_COLS for suffix in ('_prev', '_next', '_daymean')] + ['cloud_diff']
SOLAR = [p + '_' + name for p in ('wa', 'wb') for name in
         ('solar_heat_proxy', 'solar_cooling_proxy', 'cloud_solar_proxy')]
SITE = 'site_id_cat'
DEFAULT_ROOT = Path(r'C:/Users/gyumi/OneDrive/바탕 화면/data')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def require_unique(frame, keys, label):
    if frame.duplicated(keys).any():
        raise ValueError(label + ': duplicate keys ' + str(keys))


def raw_extras(frame, p, use_time, use_nat):
    """Only original forecasts: never aggregate/shift target-dependent OOF corrections."""
    station = p + '_station'
    cols = A_COLS if p == 'wa' else B_COLS
    keys = [station, 'round', 'ts']
    require_unique(frame, keys, p + ' raw forecast')
    frame = frame.sort_values(keys).copy()
    keep = keys.copy()
    if use_time:
        # Match teammate's row shifts, including existing gap/boundary semantics.
        group = frame.groupby([station, 'round'])
        for col in cols:
            for suffix, shift in (('_prev', 1), ('_next', -1)):
                frame[col + suffix] = group[col].shift(shift)
                keep.append(col + suffix)
        frame['_day'] = (frame.ts - pd.Timedelta(hours=1)).dt.normalize()
        for col in cols:
            frame[col + '_daymean'] = frame.groupby([station, 'round', '_day'])[col].transform('mean')
            keep.append(col + '_daymean')
        frame[p + '_raw_cloud_for_diff'] = frame[p + '_cloud']
        keep.append(p + '_raw_cloud_for_diff')
    if use_nat:
        cloud = p + '_cloud'
        prefix = 'a' if p == 'wa' else 'b'
        group = frame.groupby(['round', 'ts'])
        frame[prefix + '_nat_cloud_mean'] = group[cloud].transform('mean')
        frame[prefix + '_nat_cloud_std'] = group[cloud].transform('std')
        col, name = ('wa_rain', 'a_nat_rain_mean') if p == 'wa' else ('wb_uv_idx', 'b_nat_uv_mean')
        frame[name] = group[col].transform('mean')
        frame[prefix + '_cloud_dev'] = frame[cloud] - frame[prefix + '_nat_cloud_mean']
        keep += [c for c in NAT if c.startswith(prefix + '_')]
    return frame[keep]


def solar_features(d):
    z = np.maximum(d.cosz.to_numpy(dtype=float), 0)
    for p in ('wa', 'wb'):
        t = d[p + '_temp'].to_numpy(dtype=float)
        wind = d[p + '_wind_speed'].to_numpy(dtype=float) / (3.6 if p == 'wb' else 1.)
        cloud = d[p + '_cloud'].to_numpy(dtype=float)
        t = np.where((t >= -50) & (t <= 60), t, np.nan)
        wind = np.where((wind >= 0) & (wind <= 60), wind, np.nan)
        cloud = np.where((cloud >= 0) & (cloud <= 100), cloud, np.nan)
        d[p + '_solar_heat_proxy'] = (z * (t - 25.)).astype('float32')
        d[p + '_solar_cooling_proxy'] = (z * wind).astype('float32')
        d[p + '_cloud_solar_proxy'] = (z * (1 - cloud / 100.)).astype('float32')
    return d


def weights(d, prices):
    price = d.ts.map(prices).to_numpy(dtype=float)
    cap = d.cap.to_numpy(dtype=float)
    k = np.where(d.ts.dt.hour.between(11, 15), 3., 1.)
    if not np.isfinite(price).all() or (price < 0).any() or not np.isfinite(cap).all() or (cap <= 0).any():
        raise ValueError('Missing/invalid training SMP or capacity')
    w = price * k * cap
    if not np.isfinite(w).all() or w.mean() <= 0:
        raise ValueError('Invalid training weights')
    return w / w.mean()


def model_x(d, features, kind, levels):
    x = d[features].astype('float32')
    if SITE in features:
        if kind == 'lgbm':
            x[SITE] = pd.Categorical(x[SITE], categories=levels)
        else:
            x[SITE] = x[SITE].astype('int64').astype(str)
    numeric = [c for c in features if c != SITE]
    if np.isinf(x[numeric].to_numpy()).any():
        raise ValueError('Infinite numeric model feature')
    return x


def checked_prediction(path, actual):
    pred = pd.read_parquet(path)
    require_unique(pred, ['site_id', 'ts'], str(path))
    result = actual[['site_id', 'ts']].merge(pred, how='left', on=['site_id', 'ts'], validate='one_to_one')
    if len(pred) != len(actual) or not np.isfinite(result.pred_kwh).all():
        raise ValueError('Cached prediction keys/values differ: ' + str(path))
    return result


def main(weather_context=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', choices=['random'], default='random')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--no-site', action='store_true')
    ap.add_argument('--no-time', action='store_true')
    ap.add_argument('--no-nat', action='store_true')
    ap.add_argument('--chunk-hours', type=int, default=168)
    args = ap.parse_args()
    if args.chunk_hours < 1:
        ap.error('chunk-hours must be positive')
    # Check both dependencies before starting any expensive work.
    import lightgbm as lgb
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise ImportError('Install catboost into the Spyder Python environment before running this script.') from error
    root = args.data_dir.resolve()
    codes = root / 'codes'
    base = load(root / 'baseline.py', 'weather94_base')
    v = load(codes / 'baseline_validation.py', 'weather94_validation')
    opt = load(codes / 'baseline_optimizer.py', 'weather94_optimizer')
    reference = Path(weather_context['reference']) if weather_context else root / 'output/validation_602020/split602020_weather_graph_distribution_l1_smp_k'
    graph = Path(weather_context['graph']) if weather_context else root / 'output/weather_error_graph'
    source = json.loads((reference / 'summary.json').read_text(encoding='utf-8'))
    meta = json.loads((graph / 'manifest.json').read_text(encoding='utf-8'))
    for field, path in [('baseline_sha256', root / 'baseline.py'),
                        ('validation_sha256', codes / 'baseline_validation.py'),
                        ('optimizer_sha256', codes / 'baseline_optimizer.py')]:
        if meta.get(field) != sha(path):
            raise ValueError('Weather cache source code differs: ' + str(path))
    if (meta.get('seed'), meta.get('folds'), meta.get('max_weather_rows')) != (42, 3, 50000):
        raise ValueError('Expected original median weather cache: seed42 / 3fold / 50k rows')
    if source.get('seed') != 42 or source.get('historical_training_added', False) != bool(weather_context):
        raise ValueError('Reference split incompatible with cached OOF weather')
    features60 = source['generation_features']
    if len(features60) != 60 or features60[:len(base.FEATS)] != list(base.FEATS):
        raise ValueError('Existing baseline60 feature schema changed')
    params_lgb = source['model_params']
    if params_lgb.get('objective') != 'regression_l1' or params_lgb.get('n_estimators') != 600:
        raise ValueError('Expected existing weighted L1 / 600-tree parameters')
    params_cat = dict(loss_function='MAE', iterations=600, learning_rate=.06, depth=8,
                      random_seed=42, verbose=100, thread_count=-1, allow_writing_files=False)
    features = features60 + SOLAR + ([] if args.no_time else TIME) + ([] if args.no_nat else NAT) + ([] if args.no_site else [SITE])
    if len(set(features)) != len(features):
        raise ValueError('Overlapping feature names')
    name = 'random' + ('_notime' if args.no_time else '') + ('_nonat' if args.no_nat else '') + ('_nosite' if args.no_site else '')
    out = Path(weather_context['output']) if weather_context else (args.output_dir.resolve() if args.output_dir else root / 'output/validation_weather94_ensemble' / name)
    protected = [reference.resolve(), graph.resolve(), (root / 'train').resolve(), (root / 'test').resolve(), codes.resolve()]
    if out == root or any(out == p or p in out.parents or out in p.parents for p in protected):
        raise ValueError('Choose a separate output directory')
    out.mkdir(parents=True, exist_ok=True)
    files = [root / 'baseline.py', root / 'sites.csv', root / 'train/generation.parquet', root / 'train/smp.csv',
             codes / 'baseline_validation.py', codes / 'baseline_optimizer.py', reference / 'summary.json',
             reference / 'split_days.csv', graph / 'split_days.csv', graph / 'manifest.json']
    for p in ('wa', 'wb'):
        files += [graph / f'graph_forecast_{p}.parquet', reference / 'weather_quantiles' / f'quantile_features_{p}.parquet',
                  root / 'output/preprocessed/train' / f'forecast_{p}.parquet']
    if weather_context:
        files += [Path(weather_context['cache_manifest'])]
    signature = []
    for path in files:
        st = path.stat()
        signature.append(dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                              sha256=sha(path) if path.suffix in ('.py', '.json', '.csv') else None))
    manifest = dict(script_sha256=sha(__file__), inputs=signature, features=features,
                    lgbm_params=params_lgb, catboost_params=params_cat, historical_training_added=bool(weather_context),
                    packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'lightgbm', 'catboost', 'pyarrow']},
                    extra_feature_source='original forecasts, same round/time; no shifted OOF correction',
                    missing_residual_policy='existing helper: missing site-hours contribute zero residual, not true zero generation')
    checkpoint = out / 'manifest.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Code/input/config changed: choose a NEW --output-dir; existing predictions preserved')
    save_json(checkpoint, manifest)
    print('FEATURES', len(features), 'OUTPUT', out, flush=True)
    sites = pd.read_csv(root / 'sites.csv')
    require_unique(sites, ['site_id'], 'sites.csv')
    site_codes = {sid: i for i, sid in enumerate(sorted(sites.site_id))}
    gen = pd.read_parquet(root / 'train/generation.parquet', columns=['site_id', 'ts', 'energy'])
    gen['ts'] = v.to_kst(gen.ts)
    if not weather_context:
        gen = gen.loc[gen.ts >= base.TRAIN_FROM].copy()
    require_unique(gen, ['site_id', 'ts'], 'generation')
    fit, group_actual, hold_actual, split = v.split_602020(gen.loc[gen.ts >= base.TRAIN_FROM], 42)
    for path in [reference / 'split_days.csv', graph / 'split_days.csv']:
        pd.testing.assert_frame_equal(split, pd.read_csv(path))
    if weather_context:
        old = gen.loc[gen.ts < base.TRAIN_FROM].copy()
        fit = pd.concat([old, fit], ignore_index=True)
        print('HISTORICAL TRAINING ADDED', len(old), 'TOTAL FIT', len(fit), flush=True)
        del old
    weather_start = gen.ts.min() if weather_context else base.TRAIN_FROM
    split.to_csv(out / 'split_days.csv', index=False)
    actual = pd.concat([group_actual, hold_actual], ignore_index=True)
    require_unique(actual, ['site_id', 'ts'], 'evaluation labels')
    lookup = sites.set_index('site_id')
    elev, _, _ = base.solar(pd.DatetimeIndex(fit.ts), fit.site_id.map(lookup.lat).to_numpy(), fit.site_id.map(lookup.lon).to_numpy())
    fit = fit.loc[elev > 0].copy()
    levels = sorted(fit.site_id.map(site_codes).dropna().unique().tolist())
    prices = pd.read_csv(root / 'train/smp.csv', usecols=['ts', 'smp'])
    prices['ts'] = v.to_kst(prices.ts)
    require_unique(prices, ['ts'], 'SMP')
    prices = pd.to_numeric(prices.set_index('ts').smp, errors='coerce')
    times = pd.DatetimeIndex(actual.ts.unique()).sort_values()
    round_predictions = {}
    for r in (10, 17):
        paths = {kind: out / f'{kind}_round{r}_eval.parquet' for kind in ('lgbm', 'catboost')}
        pending = []
        for kind, path in paths.items():
            if path.exists():
                round_predictions[kind, r] = checked_prediction(path, actual)
                print('RESUME PREDICTIONS', kind, r, flush=True)
            else:
                pending.append(kind)
        if not pending:
            continue
        weather, aux, extra = {}, {}, {}
        for p, cols in [('wa', A_COLS), ('wb', B_COLS)]:
            filters = [('round', '=', r)]
            weather[p] = pd.read_parquet(graph / f'graph_forecast_{p}.parquet', filters=filters)
            aux[p] = pd.read_parquet(reference / 'weather_quantiles' / f'quantile_features_{p}.parquet', filters=filters)
            raw = pd.read_parquet(root / 'output/preprocessed/train' / f'forecast_{p}.parquet',
                                  columns=[p + '_station', 'round', 'ts'] + cols, filters=filters)
            raw['ts'] = v.to_kst(raw.ts)
            raw = raw.loc[raw.ts >= weather_start]
            extra[p] = raw_extras(raw, p, not args.no_time, not args.no_nat)
            del raw
            for frame in (weather[p], aux[p]):
                frame['ts'] = v.to_kst(frame.ts)
                require_unique(frame, [p + '_station', 'round', 'ts'], p)

        def design(labels):
            d = base.features(labels, sites, weather['wa'], weather['wb'], None, r)
            if len(d) != len(labels):
                raise ValueError('Weather merge changed label count')
            for p in ('wa', 'wb'):
                for table in (aux[p], extra[p]):
                    one = table.drop(columns='round')
                    d = d.merge(one, on=[p + '_station', 'ts'], how='left', validate='many_to_one')
            if not d[['site_id', 'ts']].reset_index(drop=True).equals(labels[['site_id', 'ts']].reset_index(drop=True)):
                raise ValueError('Feature merge changed label order')
            if not args.no_time:
                d['cloud_diff'] = d.wa_raw_cloud_for_diff - d.wb_raw_cloud_for_diff
            if not args.no_site:
                d[SITE] = d.site_id.map(site_codes)
                if d[SITE].isna().any():
                    raise ValueError('Unknown site ID outside sites.csv')
            return solar_features(d)

        for kind in pending:
            print('BUILD TRAIN', kind, r, flush=True)
            d = design(fit)
            x = model_x(d, features, kind, levels)
            y = (d.energy / d.cap).clip(0, 1.2).astype('float32')
            w = weights(d, prices)
            print('FIT', kind, r, x.shape, flush=True)
            model = lgb.LGBMRegressor(**params_lgb) if kind == 'lgbm' else CatBoostRegressor(**params_cat)
            categorical = {} if args.no_site else ({'categorical_feature': [SITE]} if kind == 'lgbm' else {'cat_features': [SITE]})
            model.fit(x, y, sample_weight=w, **categorical)
            save_json(out / f'{kind}_round{r}_fit_audit.json', dict(rows=len(x), features=len(features),
                       feature_names=features, weight_mean=float(w.mean()),
                       missing_fraction={c: float(d[c].isna().mean()) for c in features}))
            del d, x, y, w
            gc.collect()
            parts = []
            for offset in range(0, len(times), args.chunk_hours):
                chunk = actual.loc[actual.ts.isin(times[offset:offset + args.chunk_hours])]
                d = design(chunk)
                x = model_x(d, features, kind, levels)
                cf = model.predict(x)
                pred = np.maximum(np.where(d.elev.to_numpy() > 0, cf, 0), 0) * d.cap.to_numpy()
                p = d[['site_id', 'ts']].copy()
                p['pred_kwh'] = pred
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite predictions')
                parts.append(p)
                if offset % 1008 == 0:
                    print('PREDICT', kind, r, offset, '/', len(times), flush=True)
                del d, x
            result = pd.concat(parts, ignore_index=True)
            temporary = paths[kind].with_suffix('.tmp.parquet')
            result.to_parquet(temporary, index=False)
            temporary.replace(paths[kind])
            round_predictions[kind, r] = checked_prediction(paths[kind], actual)
            del model, result, parts
            gc.collect()
        del weather, aux, extra
        gc.collect()

    averages = {}
    for kind in ('lgbm', 'catboost'):
        p = actual[['site_id', 'ts']].copy()
        p['pred_kwh'] = (round_predictions[kind, 10].pred_kwh.to_numpy() + round_predictions[kind, 17].pred_kwh.to_numpy()) / 2
        p.to_parquet(out / f'{kind}_eval_predictions.parquet', index=False)
        averages[kind] = p.pred_kwh.to_numpy()
    del round_predictions, fit, gen
    rows = []
    initial = opt.initial_mask(sites)
    grid = np.arange(.85, 1.2001, .005)
    for alpha in [1., .75, .5, .25, 0.]:
        pred = actual[['site_id', 'ts']].copy()
        pred['pred_kwh'] = alpha * averages['lgbm'] + (1 - alpha) * averages['catboost']
        group_p = group_actual[['site_id', 'ts']].merge(pred, on=['site_id', 'ts'], validate='one_to_one')
        hold_p = hold_actual[['site_id', 'ts']].merge(pred, on=['site_id', 'ts'], validate='one_to_one')
        mg, tg, _ = v.residual_matrix(group_actual, group_p, sites.site_id, 'group_opt')
        mh, th, _ = v.residual_matrix(hold_actual, hold_p, sites.site_id, 'holdout')
        wg = opt.time_weights(root, tg)
        # Fit hourly scaling on group_opt only, matching teammate's search.
        pg = group_p.pivot(index='ts', columns='site_id', values='pred_kwh').reindex(index=tg, columns=sites.site_id).fillna(0).to_numpy()
        ph = hold_p.pivot(index='ts', columns='site_id', values='pred_kwh').reindex(index=th, columns=sites.site_id).fillna(0).to_numpy()
        # Keep the inherited zero-residual policy even if an explicit energy NA exists.
        valid_g = group_actual.pivot(index='ts', columns='site_id', values='energy').reindex(index=tg, columns=sites.site_id).notna().to_numpy()
        valid_h = hold_actual.pivot(index='ts', columns='site_id', values='energy').reindex(index=th, columns=sites.site_id).notna().to_numpy()
        pg = np.where(valid_g, pg, 0.)
        ph = np.where(valid_h, ph, 0.)
        hour_c = {}
        for hour in range(24):
            ix = tg.hour == hour
            losses = [float(np.sum(wg[ix] * np.abs((mg[ix] + (c - 1) * pg[ix]).sum(axis=1)))) for c in grid]
            hour_c[hour] = float(grid[int(np.argmin(losses))]) if ix.any() else 1.
        for correction in ['none', 'hourly']:
            tag = f'lgb{int(alpha * 100):03d}_{correction}'
            folder = out / tag
            folder.mkdir(exist_ok=True)
            cg = np.array([hour_c[h] for h in tg.hour])[:, None] if correction == 'hourly' else 1.
            ch = np.array([hour_c[h] for h in th.hour])[:, None] if correction == 'hourly' else 1.
            matrix_g, matrix_h = mg + (cg - 1) * pg, mh + (ch - 1) * ph
            optimized, meta_greedy, trace = opt.greedy_best_improvement(matrix_g, wg, sites, initial)
            score = v.score_group(opt, root, matrix_h, th, initial, optimized)
            hp = hold_p.copy()
            if correction == 'hourly':
                hp['pred_kwh'] *= hp.ts.dt.hour.map(hour_c)
            metrics = v.model_metrics(root, hold_actual, hp)
            i, o, lower = [score[k] for k in ['W_initial_krw', 'W_optimized_krw', 'W_lower_bound_krw']]
            row = dict(validation='random_fixed_evaluation_full_history' if weather_context else 'random_60_20_20_no_old_history', lgbm_weight=alpha,
                       catboost_weight=1 - alpha, correction=correction, feature_count=len(features),
                       weighted_MAE=metrics['weighted_MAE_kwh'], initial=i, optimized=o,
                       lower_bound=lower, improvable_cost=o - lower,
                       improved_percentage=100 * (i - o) / (i - lower) if i > lower else None)
            rows.append(row)
            trace.to_csv(folder / 'greedy_trace.csv', index=False)
            pd.DataFrame({'site_id': sites.site_id, 'group': np.where(optimized, 'G1', 'G2')}).to_csv(folder / 'composition.csv', index=False)
            save_json(folder / 'summary.json', dict(result=row, metrics=metrics, penalty=score,
                       hourly_scaling=hour_c if correction == 'hourly' else None))
            pd.DataFrame(rows).to_csv(out / 'result.csv', index=False, encoding='utf-8-sig')
            print('RESULT', json.dumps(row, ensure_ascii=False), flush=True)
    print('DONE', out / 'result.csv', flush=True)


if __name__ == '__main__':
    main()
