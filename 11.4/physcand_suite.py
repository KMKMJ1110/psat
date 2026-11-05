#!/usr/bin/env python3
"""Six independent, day-ahead safe No.28 feature experiments.

Requires the previously supplied codes/newcand_common.py and the existing
codes/baseline.py, baseline_validation.py, baseline_optimizer.py, nat_feat.py,
validation_timesplit.py and local contest data. It does not access the Internet.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import newcand_common as common

EXPERIMENTS = {
    1: (41, 'ceiling_only', 'WB ceiling raw feature; 1 feature'),
    2: (42, 'vpd', 'Magnus saturation vapour pressure deficit; 1 feature'),
    3: (43, 'clearsky_cloud', 'Clear-sky irradiance times cloud attenuation; 2 features'),
    4: (44, 'module_temperature', 'Faiman-like solar cell temperature proxy; 2 features'),
    5: (45, 'optical_transmission', 'EDA1 multivariate UV/cloud/humidity/ceiling irradiance proxy; 2 features'),
    6: (46, 'round_revision', 'Same valid-hour forecast revision F17-F10; 8 features'),
}
for _order, (_cand_id, _slug, _desc) in EXPERIMENTS.items():
    common.CANDIDATES[_cand_id] = (_slug, _desc)


def _log(msg):
    common.log('[PHYS]', msg)


def _float(df, name):
    return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype='float64')


def _gclear(df):
    """Haurwitz approximate horizontal clear-sky irradiance (W/m^2)."""
    c = np.maximum(_float(df, 'cosz'), 0.0)
    out = 1098.0 * c * np.exp(-0.059 / np.maximum(c, 1e-6))
    return np.where(c > 0.0, out, 0.0)


def _cloud_transmission(cloud_percent):
    """Empirical monotonic cloud cover attenuation; not observed transmittance."""
    q = np.clip(np.asarray(cloud_percent, dtype='float64') / 100.0, 0.0, 1.0)
    return 1.0 - 0.75 * np.power(q, 3.4)


def _attach_wb_ceiling(ctx):
    """Attach the unused WB forecast ceiling, from local preprocessed or raw forecasts."""
    if 'wb_ceiling' in ctx.fb.columns:
        return
    paths = [ctx.root / 'output' / 'preprocessed' / 'train' / 'forecast_wb.parquet',
             ctx.root / 'train' / 'forecast_wb.parquet']
    extra = None
    last_error = None
    for p in paths:
        if not p.exists():
            continue
        try:
            extra = pd.read_parquet(p, columns=['wb_station', 'round', 'ts', 'wb_ceiling'])
            _log(f'Loaded wb_ceiling from {p}')
            break
        except Exception as e:
            last_error = str(e)
    if extra is None:
        raise FileNotFoundError(f'wb_ceiling not available in local WB forecast files: {last_error}')
    extra['ts'] = ctx.val.to_kst(extra.ts)
    extra = extra.loc[extra.ts >= ctx.fb.ts.min()].copy()
    keys = ['wb_station', 'round', 'ts']
    # Accept byte-for-byte/equivalent duplicate forecast keys, not conflicting sources.
    extra = extra.drop_duplicates()
    if extra.duplicated(keys).any():
        raise ValueError('WB ceiling forecast has conflicting duplicate station/round/time keys')
    prev = len(ctx.fb)
    ctx.fb = ctx.fb.merge(extra, on=keys, how='left', validate='many_to_one', sort=False)
    if len(ctx.fb) != prev:
        raise AssertionError('WB ceiling enrichment unexpectedly altered forecast row count')
    coverage = ctx.fb.wb_ceiling.notna().mean()
    _log(f'WB ceiling joined: coverage {coverage:.2%}')
    if coverage == 0.0:
        raise ValueError('WB ceiling forecast joined zero non-null values; check timestamps/rounds')
    if coverage < 0.80:
        _log(f'WARNING: WB ceiling coverage {coverage:.1%} is low; check whether missing values mean clear sky')


def _append_round_revisions(df, station, variables):
    """Compute all F17-F10 differences for the SAME station + target timestamp.

    Forecast is made after the day-before 17:00 issue, so both rounds are known.
    The same revisions are supplied to the round-10 and round-17 PV models.
    """
    before = len(df)
    keys = [station, 'ts']
    rkeys = [station, 'ts', 'round']
    if df.duplicated(rkeys).any():
        raise ValueError(f'Duplicate forecast key: {rkeys}')
    basecols = keys + list(variables)
    early = df.loc[df['round'].eq(10), basecols]
    late = df.loc[df['round'].eq(17), basecols]
    j = late.merge(early, on=keys, suffixes=('_17', '_10'), validate='one_to_one', sort=False)
    out = j[keys].copy()
    names = []
    for col in variables:
        nm = 'revision_' + col
        out[nm] = pd.to_numeric(j[col+'_17'],errors='coerce').to_numpy() - pd.to_numeric(j[col+'_10'],errors='coerce').to_numpy()
        names.append(nm)
    result = df.merge(out, on=keys, how='left', validate='many_to_one', sort=False)
    if len(result) != before:
        raise AssertionError('Round revision enrichment altered forecast row count')
    _log(f'{station} revision pairs={len(j):,}; feature coverage={result[names].notna().mean().mean():.2%}')
    return result, names


def _build_extras(ctx, order):
    """Prepare forecast-table enrichment once, return per-row extra-feature function and names."""
    if order in (1, 5):
        _attach_wb_ceiling(ctx)
    if order == 1:
        return None, ['wb_ceiling'], {'description': 'Raw WB ceiling; original No28 feature list kept'}
    if order == 2:
        def make(df, r):
            t = np.clip(_float(df, 'wb_temp'), -50.0, 60.0)
            rh = np.clip(_float(df, 'wb_rel_hum'), 0.0, 100.0)
            es_kpa = 0.6112 * np.exp(17.62*t/(243.12+t))
            df['phys_vpd_kpa'] = (es_kpa * (1 - rh/100.0)).astype('float32')
            return df
        return make, ['phys_vpd_kpa'], {'description': 'VPD in kPa from Magnus saturation vapour pressure'}
    if order == 3:
        def make(df, r):
            g = _gclear(df)
            df['phys_clearcloud_wa'] = (g * _cloud_transmission(_float(df, 'wa_cloud'))).astype('float32')
            df['phys_clearcloud_wb'] = (g * _cloud_transmission(_float(df, 'wb_cloud'))).astype('float32')
            return df
        return make, ['phys_clearcloud_wa', 'phys_clearcloud_wb'], {
            'description':'Haurwitz G_clear * (1 - 0.75 * cloud_fraction^3.4) for WA and WB independently'}
    if order == 4:
        def make(df, r):
            g = _gclear(df)
            ta = _float(df, 'wa_temp')
            va = np.maximum(_float(df, 'wa_wind_speed'), 0.0)  # WA m/s
            tb = _float(df, 'wb_temp')
            vb = np.maximum(_float(df, 'wb_wind_speed')/3.6, 0.0)  # WB km/h -> m/s
            df['phys_tmodule_wa'] = (ta + g/(25.0 + 6.84*va)).astype('float32')
            df['phys_tmodule_wb'] = (tb + g/(25.0 + 6.84*vb)).astype('float32')
            return df
        return make, ['phys_tmodule_wa','phys_tmodule_wb'], {
            'description':'Faiman-like T_ambient + G_clear/(25 + 6.84*wind_m_per_s), separately WA/WB; coefficients are approximate, not PV-site-calibrated'}
    if order == 5:
        def make(df, r):
            # No future observed weather or generation is used. This is not a
            # physical radiation retrieval, only an interpretable optical proxy.
            g = _gclear(df)
            cloud = 0.5*(np.clip(_float(df,'wa_cloud'),0,100)+np.clip(_float(df,'wb_cloud'),0,100))
            uv = np.maximum(_float(df,'wb_uv_idx'),0)
            rh = 0.5*(np.clip(_float(df,'wa_humidity'),0,100)+np.clip(_float(df,'wb_rel_hum'),0,100))/100.0
            ceiling = np.maximum(_float(df,'wb_ceiling'),0)
            # UV index is dimensionless; use a bounded saturating UV signal.
            t_uv = uv/(uv+2.0)
            t_cloud = _cloud_transmission(cloud)
            t_rh = np.exp(-0.35*rh)
            # Positive ceiling is in WB source units; scale 1000 is a heuristic
            # and *not* a claimed physical cloud-base law.
            t_ceiling = 1.0 - 0.15*np.exp(-ceiling/1000.0)
            df['phys_optical_eda1'] = (g*t_uv*t_cloud*t_rh*t_ceiling).astype('float32')
            # A second scale-free transmissivity proxy helps avoid the model
            # mistaking nighttime low irradiance for poor sky transparency.
            df['phys_optical_transmission'] = (t_uv*t_cloud*t_rh*t_ceiling).astype('float32')
            return df
        return make, ['phys_optical_eda1','phys_optical_transmission'], {
            'description':'EDA1 UV-cloud-humidity-ceiling joint transmission; bounded, heuristic proxy (not a measured physical optical-depth model)'}
    if order == 6:
        avars = ['wa_cloud', 'wa_humidity', 'wa_rain']
        bvars = ['wb_cloud', 'wb_uv_idx', 'wb_rel_hum', 'wb_temp', 'wb_total_liq']
        ctx.fa, aa = _append_round_revisions(ctx.fa, 'wa_station', avars)
        ctx.fb, bb = _append_round_revisions(ctx.fb, 'wb_station', bvars)
        return None, aa+bb, {'description':'F17 - F10 by same site station and target ts, both known after 17:00; missing revisions retained as NaN'}
    raise ValueError(order)


def _parse(order, argv=None):
    p=argparse.ArgumentParser(description=f'Independent feature experiment {order}: {EXPERIMENTS[order][1]}')
    p.add_argument('--data-dir', default=str(Path(__file__).resolve().parent.parent))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--feature-parity', choices=['nat_feat','submission_safe'], default='nat_feat')
    p.add_argument('--fresh', action='store_true', help='Rename previous run directory before rerunning')
    return p.parse_args(argv)


def run(order, argv=None):
    if order not in EXPERIMENTS:
        raise ValueError(order)
    args=_parse(order,argv)
    cand,slug,_ = EXPERIMENTS[order]
    if args.fresh:
        import shutil, time
        folder=Path(args.data_dir).expanduser().resolve()/'output'/'newcand_no28'/f'{cand:02d}_{slug}'
        if folder.exists():
            replacement=folder.with_name(folder.name+'_backup_'+time.strftime('%Y%m%d_%H%M%S'))
            shutil.move(folder,replacement)
            _log(f'Backed up previous experiment to {replacement}')
    ctx=common.prepare_context(args,cand)
    summary=ctx.out/'summary.json'
    if summary.exists():
        _log(f'SKIP already complete: {summary}')
        return json.loads(summary.read_text(encoding='utf8'))
    extra,cols,notes=_build_extras(ctx,order)
    _log(f'Experiment {order}/6 [{slug}]: No28 51 + {len(cols)} features = {51+len(cols)}, seed={args.seed}')
    pg,ph=common.train_predict_four(ctx,'physcand_'+slug,extra=extra,extra_cols=cols)
    notes=dict(notes, feature_names=cols, reference='No28 51 features + weighted L1; four models; unchanged 60:20:20 seed42; Greedy on group 20%; independent holdout 20%')
    report=common.evaluate(ctx,pg,ph,notes=notes)
    _log(f'{slug}: holdout W={report["holdout_W_krw"]:,.2f}; WMAE={report["weighted_MAE"]:.5f}; delta={report["delta_vs_no28_krw"]:,.2f}')
    return report


def cli(order):
    try:
        run(order)
    except Exception:
        _log(f'Experiment {order} failed. Rerun the same command to resume checkpoints.')
        traceback.print_exc()
        raise
