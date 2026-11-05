#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No.28 RH residual REPLAY: physically dark hours forced to zero, GroupOpt multi-start Greedy.

No models are trained. Read only SAVED predictions (GroupOpt OOF, Holdout from full GroupOpt fit).
Original files are NOT modified; results in separate `rh_site_residual_night_safe_replay` folder.

Requires local codes/no28_rh_residual_models.py and codes/baseline_optimizer.py,
 plus previously finished output/newcand_no28/rh_site_residual_ridge_vs_lgb/.

From data root:
    /usr/bin/env python3 -u codes/no28_rh_night_safe_replay.py

Time convention: ts is KST RIGHT-CLOSED (ts-1 hour, ts] for each generation hour.
Night is full hour without visible sun: solar elevation at start, midpoint, and end
all < -0.833 deg. The threshold models geometric upper limb/refraction. Positive
predictions are forced to 0 ONLY for these fully-dark hours. Partial-light hours
are preserved to avoid accidentally removing sunrise/sunset generation.

No Holdout labels are used in model selection or grouping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from no28_rh_residual_models import (
    BASELINE_W, BASELINE_N, BASELINE_NH, KST, json_ready,
    store_json, store_csv, residual_matrix_for_split, matrix_scores,
)

VERSION = 'rh_site_residual_night_safe_replay_v1_20261104'
METHODS = ('ridge', 'lightgbm_q50')
CORRECTION_SOURCE_VERSION = 'no28_rh_site_residual_20261104_v1'
SUN_UP_ALT_DEG = -0.833
PARITY_TOLERANCE_KRW = 100.0  # <1.6 ppm of No28 W; saved predictions are float32


def log(*x):
    print('[RH-REPLAY]', *x, flush=True)


def kst(series):
    x = pd.to_datetime(series)
    if x.dt.tz is None:
        return x.dt.tz_localize(KST)
    return x.dt.tz_convert(KST)


def solar_altitude_noaa(ts, lat, lon):
    """Vectorized NOAA fractional-year approximate solar-altitude formula, degrees.

    `ts` must be tz-aware KST DatetimeIndex or Series, `lat`/`lon` aligned arrays.
    Geometric center of the solar disk, no artificial timezone assumption.
    Accurate enough for identifying unambiguously fully dark hour intervals in Korea.
    """
    t = pd.DatetimeIndex(ts)
    if t.tz is None:
        raise ValueError('Solar altitude requires timezone-aware timestamps')
    local = t.tz_convert(KST)
    latitude = np.asarray(lat, dtype=np.float64)
    longitude = np.asarray(lon, dtype=np.float64)
    if len(latitude) != len(t) or len(longitude) != len(t):
        raise ValueError('Coordinate/timestamp row misalignment')
    if not (np.isfinite(latitude).all() and np.isfinite(longitude).all()):
        raise ValueError('Nonfinite coordinates')
    if not ((latitude >= -90).all() and (latitude <= 90).all() and
            (longitude >= -180).all() and (longitude <= 180).all()):
        raise ValueError('Invalid latitude / longitude')
    doy = local.dayofyear.to_numpy(dtype=np.float64)
    hour = (local.hour.to_numpy(dtype=np.float64) +
            local.minute.to_numpy(dtype=np.float64)/60.0 +
            local.second.to_numpy(dtype=np.float64)/3600.0)
    gamma = 2.0*np.pi/365.0*(doy-1.0 + (hour-12.0)/24.0)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(gamma)
             -0.032077*np.sin(gamma) -0.014615*np.cos(2*gamma)
             -0.040849*np.sin(2*gamma))
    dec = (0.006918-0.399912*np.cos(gamma)+0.070257*np.sin(gamma)
           -0.006758*np.cos(2*gamma)+0.000907*np.sin(2*gamma)
           -0.002697*np.cos(3*gamma)+0.00148*np.sin(3*gamma))
    # KST has fixed UTC+9 and no DST. Solar time (minutes).
    true_solar_minutes = (hour*60.0+eqtime+4*longitude-540.0) % 1440.0
    hour_angle = np.deg2rad(true_solar_minutes/4.0 - 180.0)
    phi = np.deg2rad(latitude)
    cos_zen = (np.sin(phi)*np.sin(dec)
               +np.cos(phi)*np.cos(dec)*np.cos(hour_angle))
    return 90.0-np.rad2deg(np.arccos(np.clip(cos_zen,-1.0,1.0)))


def whole_hour_dark(ts,lat,lon, threshold=SUN_UP_ALT_DEG):
    """True only when complete right-closed hour lies below the solar horizon.

    Three samples catch sunrise/sunset crossings; conservative all-three rule.
    In the contest's Korean latitudes one hour cannot contain two crossings.
    """
    tt = pd.DatetimeIndex(ts)
    dark = np.ones(len(tt), dtype=bool)
    # Re-evaluating an entire vector 3 times avoids 3 huge simultaneous matrices.
    for minute in (-60,-30,0):
        z = solar_altitude_noaa(tt + pd.Timedelta(minutes=minute),lat,lon)
        dark &= z < threshold
        del z
    return dark


def load_site_metadata(root):
    sites = pd.read_csv(root/'sites.csv')
    required = ['site_id','capacity_kw','lat','lon']
    if not set(required).issubset(sites.columns):
        raise ValueError(f'sites.csv must have {required}; got {list(sites.columns)}')
    if len(sites)!=253 or not sites.site_id.is_unique:
        raise ValueError('Must have exactly 253 unique sites')
    if sites[required[1:]].isna().any().any():
        raise ValueError('Missing site metadata')
    if (sites.capacity_kw.to_numpy(dtype=float)<=0).any():
        raise ValueError('Nonpositive plant capacity')
    return sites


def read_previous_predictions(folder,method,split,sites,summary,weights_by_time):
    file = folder/method/f'{split}_site_predictions.parquet'
    if not file.exists():
        raise FileNotFoundError('Missing saved predictions; no retraining fallback: '+str(file))
    cols=['site_id','ts','rh_band','pred_kwh','new_pred_kwh',
          'baseline_error_kwh','corrected_error_kwh',
          'model_predicted_residual_cf','applied_correction_kwh']
    d=pd.read_parquet(file,columns=cols)
    if len(d)!=BASELINE_N[split]:
        raise ValueError(f'{method}/{split}: expected {BASELINE_N[split]}, got {len(d)}')
    d['ts']=kst(d.ts)
    if d.duplicated(['site_id','ts']).any():
        raise ValueError(f'{method}/{split}: duplicate site-hour keys')
    d=d.merge(sites[['site_id','capacity_kw','lat','lon']],
              how='left',on='site_id',sort=False,validate='many_to_one')
    if d[['capacity_kw','lat','lon']].isna().any().any():
        raise ValueError(f'{method}/{split}: unknown site ID')
    for col in cols[3:]:
        if not np.isfinite(pd.to_numeric(d[col],errors='coerce').to_numpy(float)).all():
            raise ValueError(f'{method}/{split}: invalid saved {col}')
    ts=pd.DatetimeIndex(d.ts.unique()).sort_values()
    if len(ts)!=BASELINE_NH[split]:
        raise ValueError(f'{method}/{split}: unexpected observed-hour count')
    wk=weights_by_time(ts)
    tmap=pd.Series(wk,index=ts)
    d['wk']=tmap.reindex(pd.DatetimeIndex(d.ts)).to_numpy(dtype=np.float64)
    if d.wk.isna().any() or (d.wk<0).any():
        raise ValueError('Missing or negative SMP*k')
    if d['rh_band'].isna().any():
        raise ValueError(f'{method}/{split}: missing saved RH bin')
    # Validate old prediction itself. It MUST be the same method result as in prior run.
    orig=d.pred_kwh.to_numpy(dtype=np.float64)
    old=d.new_pred_kwh.to_numpy(dtype=np.float64)
    e=d.baseline_error_kwh.to_numpy(dtype=np.float64)
    old_e=d.corrected_error_kwh.to_numpy(dtype=np.float64)
    if (orig<0).any() or (old<0).any():
        raise ValueError('Saved generation contains negatives')
    old_delta=old-orig
    if np.max(np.abs(old_e-(e+old_delta)))>0.004:
        raise ValueError('Saved original corrected error conflicts with saved forecasts')
    app=d.applied_correction_kwh.to_numpy(dtype=np.float64)
    if np.max(np.abs(app-old_delta))>0.004:
        raise ValueError('Saved applied correction is inconsistent')
    outside=~d.rh_band.isin(['RH_20_40','RH_40_60']).to_numpy()
    if np.max(np.abs(old_delta[outside]),initial=0)>0.004:
        raise ValueError('Prior method modified RH outside target bands')
    mat,t_check,w_check=residual_matrix_for_split(d,e,sites)
    if not ts.equals(t_check) or not np.allclose(wk,w_check,atol=1e-8,rtol=1e-9):
        raise ValueError('Unexpected time-weight alignment')
    # Reconstruct old W WITHOUT modifying source features or model predictions.
    return d,mat,ts,wk


def postprocess_dark(d):
    """Original daytime correction exactly reused; full-dark rows forced to 0 kWh."""
    base=d.pred_kwh.to_numpy(dtype=np.float64)
    old=d.new_pred_kwh.to_numpy(dtype=np.float64)
    e=d.baseline_error_kwh.to_numpy(dtype=np.float64)
    dark=whole_hour_dark(d.ts,d.lat.to_numpy(float),d.lon.to_numpy(float))
    # Important: preserving daytime predictions as originally fitted/read from cache.
    new=old.copy()
    new[dark]=0.0  # physical zero even if old No28 had tiny nighttime positives
    if np.any(new[dark]!=0.0): raise AssertionError('Night zero invariant failed')
    if not np.array_equal(new[~dark],old[~dark]):
        raise AssertionError('A daylight prediction changed unexpectedly')
    # Reuse the SAVED corrected signed errors exactly on all non-dark rows.
    # Reconstruct error only for the dark rows whose predictions are changed.
    olderr=d.corrected_error_kwh.to_numpy(dtype=np.float64).copy()
    err=olderr.copy()
    err[dark]=e[dark]-base[dark]
    return new,err,olderr,dark,{
       'site_hours_full_dark':int(dark.sum()),
       'site_hours_old_correction_nonzero_dark':int((dark & (abs(old-base)>1e-6)).sum()),
       'site_hours_old_prediction_positive_dark':int((dark & (old>1e-6)).sum()),
       'site_hours_base_prediction_positive_dark':int((dark & (base>1e-6)).sum()),
       'site_hours_saved_daylight_unchanged':int((~dark).sum()),
       'night_baseline_kwh_sum':float(base[dark].sum()),
       'night_old_prediction_kwh_sum':float(old[dark].sum()),
       'night_new_prediction_kwh_sum':float(new[dark].sum()),
       'night_original_correction_kwh_sum':float((old-base)[dark].sum()),
       'site_hours_changed_vs_original_corrected':int((abs(new-old)>1e-6).sum()),
       'site_hours_daylight_model_correction_active':int(((~dark)&(abs(old-base)>1e-6)).sum()),
    }


def compare_against_original_run(mat_old,wk,mask,summary,split,method):
    old_w,old_lb=matrix_scores(mat_old,wk,mask)
    want=float(summary[method][split]['W_fixed_original_mask_krw'])
    if abs(old_w-want)>PARITY_TOLERANCE_KRW:
        raise ValueError(f'{method}/{split}: prior-run fixed W parity FAILED: '
                         f'from saved forecasts={old_w:.5f}, logged={want:.5f}. '
                         'DO NOT CONTINUE with unverifiable cache.')
    target_lb=float(summary[method][split]['adjusted_TLB_krw'])
    if abs(old_lb-target_lb)>PARITY_TOLERANCE_KRW:
        raise ValueError(f'{method}/{split}: prior-run TLB parity FAILED: '
                         f'from saved forecasts={old_lb:.5f}, logged={target_lb:.5f}')
    log(method,split,f'old corrected W/LB parity passed (W={old_w:,.2f}, TLB={old_lb:,.2f})')
    return old_w,old_lb


def greedy_restarts(mat,wk,sites,base_mask,previous_mask,initial_mask,opt,method):
    """GroupOpt only. Evaluate original and warm-start masks; run native Greedy from all 3.

    BEST BY GROUPOPT W ONLY. Independent Holdout is never accessed here.
    """
    cap=sites.capacity_kw.to_numpy(dtype=float)
    starts=[('no28_original_optimized',base_mask),
            ('previous_run_optimized',previous_mask),
            ('original_initial',initial_mask)]
    seen=set()
    candidates=[]
    traces=[]
    for label,m in starts:
        mask=np.asarray(m,dtype=bool)
        opt.check_capacity(mask,cap)
        key=mask.tobytes()
        w_before,_=matrix_scores(mat,wk,mask)
        candidates.append(dict(name=label+'_unmodified',mask=mask.copy(),W=w_before))
        if key in seen:
            log(method,f'SKIP duplicate restart: {label}; unmodified mask still evaluated')
            continue
        seen.add(key)
        log(method,f'Original Greedy from {label}: start W={w_before:,.2f}')
        mask2,meta,trace=opt.greedy_best_improvement(mat,wk,sites,mask.copy())
        mask2=np.asarray(mask2,dtype=bool)
        opt.check_capacity(mask2,cap)
        w_after,_=matrix_scores(mat,wk,mask2)
        if w_after>w_before+0.01:
            raise ValueError(f'Greedy worsened its own starting partition {label}: '
                             f'{w_before:.7f} -> {w_after:.7f}')
        candidates.append(dict(name=label+'_greedy',mask=mask2,W=w_after))
        traces.append(dict(start=label,W_start=w_before,W_final=w_after,
                           greedy_delta=w_after-w_before,n_steps=len(trace)))
        log(method,f'Greedy {label} finished: {w_after:,.2f} (improved {w_before-w_after:,.2f})')
    candidates.sort(key=lambda q:(round(q['W'],4),len(q['name']),q['name']))
    best=candidates[0]
    if best['W']>matrix_scores(mat,wk,base_mask)[0]+0.01:
        raise ValueError('Best candidate worse than fixed No28 mask')
    return best, pd.DataFrame([dict(name=c['name'],groupopt_W_krw=c['W']) for c in candidates]),pd.DataFrame(traces)


def summarize_site_wmae(d,new_error):
    wk=d.wk.to_numpy(dtype=np.float64)
    e=d.baseline_error_kwh.to_numpy(dtype=np.float64)
    return float(np.dot(wk,abs(new_error))/wk.sum()),float(np.dot(wk,abs(e))/wk.sum())


def run(args):
    import baseline_optimizer as opt
    start=time.monotonic()
    root=Path(args.data_dir).expanduser().resolve()
    original=root/'output/newcand_no28/rh_site_residual_ridge_vs_lgb'
    output=root/'output/newcand_no28/rh_site_residual_night_safe_replay'
    if not original.exists():
        raise FileNotFoundError('Previous completed experiment not found: '+str(original))
    oldsumfile=original/'summary.json'
    if not oldsumfile.is_file():
        raise FileNotFoundError('Previous summary.json not found')
    oldsum=json.loads(oldsumfile.read_text(encoding='utf-8'))
    if oldsum.get('version')!=CORRECTION_SOURCE_VERSION:
        raise ValueError('Prior experiment version differs; cache reuse unsafe')
    sites=load_site_metadata(root)
    cap=sites.capacity_kw.to_numpy(dtype=float)
    maskpath=root/'output/newcand_no28/_shared/no28_group_seed42_nat_feat.npy'
    base_mask=np.load(maskpath).astype(bool)
    if len(base_mask)!=len(sites):raise ValueError('Base mask must match sites.csv order')
    opt.check_capacity(base_mask,cap)
    initial_mask=np.asarray(opt.initial_mask(sites),dtype=bool)
    opt.check_capacity(initial_mask,cap)

    # Verify cached model settings (metadata only; never load models or train).
    for method in METHODS:
        configfile=original/method/'cache_config.json'
        cfg=json.loads(configfile.read_text(encoding='utf-8'))
        if cfg.get('version')!=CORRECTION_SOURCE_VERSION or cfg.get('method')!=method:
            raise ValueError('Unexpected saved model config: '+str(configfile))
        if abs(float(cfg.get('shrinkage',-1))-0.75)>1e-9 or \
                abs(float(cfg.get('max_correction_cf',-1))-0.08)>1e-9:
            raise ValueError('Unexpected previous fitted correction settings')
        if cfg.get('inputs_tag')!=oldsum.get('groupopt_cache_hash'):
            raise ValueError('Source GroupOpt residual cache hash mismatch')
    output.mkdir(parents=True,exist_ok=True)
    allrows=[]; allnight=[]; allmasks=[]
    for method in METHODS:
        log('====',method,'====')
        path=original/method
        prevmask=np.load(path/'new_optimized_mask.npy').astype(bool)
        if len(prevmask)!=len(sites): raise ValueError('Previously saved mask size wrong')
        opt.check_capacity(prevmask,cap)
        d_by_split={}
        for split in ('groupopt','holdout'):
            d,basemat,times,wk=read_previous_predictions(original,method,split,sites,oldsum,
                                                          lambda t:opt.time_weights(root,t))
            bw,blb=matrix_scores(basemat,wk,base_mask)
            if abs(bw-BASELINE_W[split])>PARITY_TOLERANCE_KRW:
                raise ValueError(f'No28 baseline mismatch {method}/{split}: '
                                 f'{bw:.5f} expected {BASELINE_W[split]:.5f}')
            newpred,newerr,olderr,dark,nightstat=postprocess_dark(d)
            oldmat,told,wold=residual_matrix_for_split(d,olderr,sites)
            compare_against_original_run(oldmat,wk,base_mask,oldsum['results'],split,method)
            newmat,tnew,wnew=residual_matrix_for_split(d,newerr,sites)
            if not times.equals(tnew) or not np.allclose(wnew,wk,atol=1e-8,rtol=1e-9):
                raise ValueError('New residual weights/times changed')
            # Check model applied predictions are unchanged outside FULL dark hours.
            if not np.array_equal(newpred[~dark],d.new_pred_kwh.to_numpy(dtype=np.float64)[~dark]):
                raise ValueError('Daytime calibration changed; intended only night gate')
            wfix,lb=matrix_scores(newmat,wk,base_mask)
            wm,wb=summarize_site_wmae(d,newerr)
            log(method,split,f'No28 W={bw:,.2f}, night-safe fixed W={wfix:,.2f}, '
                                     f'night-safe TLB={lb:,.2f}, dark rows={dark.sum():,}')
            row=dict(method=method,split=split,baseline_W_krw=bw,
                     old_source_fixed_group_W_krw=matrix_scores(oldmat,wk,base_mask)[0],
                     night_safe_fixed_group_W_krw=wfix,
                     baseline_TLB_krw=blb,
                     original_corrected_TLB_krw=matrix_scores(oldmat,wk,base_mask)[1],
                     night_safe_TLB_krw=lb,
                     baseline_site_WMAE_kwh=wb,
                     night_safe_site_WMAE_kwh=wm,
                     full_dark_site_hours=nightstat['site_hours_full_dark'],
                     prevented_night_positive_site_hours=nightstat['site_hours_old_prediction_positive_dark'])
            d_by_split[split]=dict(df=d,mat=newmat,oldmat=oldmat,wk=wk,times=times,
                                   row=row,nightstat=nightstat,newpred=newpred,newerr=newerr,dark=dark)
            allnight.append(dict(method=method,split=split,**nightstat))

        # Important: grouping is chosen ONLY on GroupOpt. Holdout remains unseen.
        group=d_by_split['groupopt']
        best,candidates,trace=greedy_restarts(group['mat'],group['wk'],sites,base_mask,
                                              prevmask,initial_mask,opt,method)
        chosen=best['mask']
        opt.check_capacity(chosen,cap)
        selected_w=best['W']
        fixed_w=group['row']['night_safe_fixed_group_W_krw']
        if selected_w>fixed_w+0.01:
            raise AssertionError('Bad GroupOpt partition selected')
        md=output/method
        md.mkdir(parents=True,exist_ok=True)
        np.save(md/'chosen_groupopt_mask.npy',chosen)
        store_csv(md/'composition.csv',pd.DataFrame({'site_id':sites.site_id,
                    'group':np.where(chosen,'G1','G2')}))
        store_csv(md/'groupopt_multistart_candidates.csv',candidates)
        store_csv(md/'groupopt_multistart_traces.csv',trace)
        allmasks.append(dict(method=method,selected_start=best['name'],
                             groupopt_selected_W_krw=selected_w,
                             groupopt_fixed_baseline_group_W_krw=fixed_w,
                             changed_sites_vs_no28=int((chosen!=base_mask).sum()),
                             G1_capacity_kw=float(cap[chosen].sum()),
                             G2_capacity_kw=float(cap[~chosen].sum())))
        for split,obj in d_by_split.items():
            final_w,_=matrix_scores(obj['mat'],obj['wk'],chosen)
            r=dict(obj['row'],selected_group_W_krw=final_w,
                   delta_selected_vs_no28_W_krw=final_w-obj['row']['baseline_W_krw'],
                   selection_source='GroupOpt only',
                   chosen_mask_name=best['name'])
            allrows.append(r)
            log(method,split,f'FINAL selected-group W={final_w:,.2f}, '
                             f'delta vs No28 {final_w-obj["row"]["baseline_W_krw"]:+,.2f}')
            if not args.no_save_predictions:
                d=obj['df']
                save=d[['site_id','ts','rh_band','pred_kwh','baseline_error_kwh']].copy()
                save['night_full_dark']=obj['dark']
                save['original_corrected_pred_kwh']=d.new_pred_kwh.to_numpy(dtype=np.float32)
                save['night_safe_pred_kwh']=obj['newpred'].astype(np.float32)
                save['night_safe_signed_error_kwh']=obj['newerr'].astype(np.float32)
                save['correction_removed_at_night']=obj['dark'] & (
                    np.abs(d.new_pred_kwh.to_numpy(float)-d.pred_kwh.to_numpy(float))>1e-6)
                save.to_parquet(md/f'{split}_night_safe_predictions.parquet',index=False)
        store_json(md/'summary.json',dict(version=VERSION,model=method,selection=allmasks[-1],
                         groupopt=allrows[-2],holdout=allrows[-1],
                         groupopt_night=allnight[-2],holdout_night=allnight[-1]))

    scoretable=pd.DataFrame(allrows)
    store_csv(output/'00_groupopt_holdout_comparison.csv',scoretable)
    store_csv(output/'01_night_correction_audit.csv',pd.DataFrame(allnight))
    store_csv(output/'02_groupopt_mask_selection.csv',pd.DataFrame(allmasks))
    result=dict(version=VERSION,
                description='Cached predictions replay only; no training, no feature or target change',
                solar_definition='Fully dark only if elevation at start/mid/end of (ts-1h,ts] all below -0.833 degrees',
                night_policy='Force 0 generation at fully dark site-hours; retain OLD saved day predictions byte-for-byte',
                selection='Multi-start original Greedy on GroupOpt only: No28 group, previous optimized, initial; include unmodified masks',
                no28_baseline=BASELINE_W,source_original=CORRECTION_SOURCE_VERSION,
                provenance_groupopt_source_hash=oldsum['groupopt_cache_hash'],
                selected_groups=allmasks,results=allrows,
                night_audit=allnight,wall_seconds=float(time.monotonic()-start))
    store_json(output/'summary.json',result)
    log('DONE',output, 'elapsed_seconds=',round(time.monotonic()-start,1))
    return result


def selftest():
    # Korea local solar time: around local noon in summer is high; midnight far below.
    seq=pd.DatetimeIndex(['2026-06-21T00:00:00+09:00',
                          '2026-06-21T12:00:00+09:00',
                          '2026-12-21T00:00:00+09:00',
                          '2026-12-21T12:00:00+09:00'])
    x=solar_altitude_noaa(seq,np.full(4,37.5),np.full(4,127.0))
    assert x[0]<-15 and x[1]>60 and x[2]<-40 and 20<x[3]<40, x
    # KST right-closed sunrise/sunset conservatism: 00:00 full hour dark, 12:00 light.
    whole=whole_hour_dark(seq,np.full(4,37.5),np.full(4,127.0))
    assert whole.tolist()==[True,False,True,False],(x,whole)
    # Daylight unchanged, dark=0, including cases with nonzero baselines at night.
    d=pd.DataFrame({'ts':seq,'lat':37.5,'lon':127.0,
                    'pred_kwh':[0,50,1,40], 'new_pred_kwh':[3,60,2,44],
                    'baseline_error_kwh':[-1,10,0,2],
                    'corrected_error_kwh':[2,20,1,6]})
    new,err,olderr,dark,stats=postprocess_dark(d)
    assert np.allclose(new,[0,60,0,44]),new
    assert np.allclose(err,[-1,20,-1,6])  # only complete-dark hours corrected
    assert stats['site_hours_old_prediction_positive_dark']==2
    log('SELFTEST PASS:',{'solar_elevations_deg':x.tolist(),
                          'full_dark_flags':whole.tolist(),
                          'new_predictions_kwh':new.tolist()})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--no-save-predictions',action='store_true',help='Skip new large Parquet outputs')
    ap.add_argument('--selftest',action='store_true',help='Run solar/physical-night invariant tests, no data needed')
    args=ap.parse_args()
    if args.selftest: selftest(); return
    run(args)


if __name__=='__main__':
    main()
