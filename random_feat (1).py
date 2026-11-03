"""L1 LGBM + CatBoost (600트리, --full) 에 '피처 추가'.
위치: codes/random_feat.py
실행: python3 codes/random_feat.py --full

추가 피처 (전부 예보에서 만든 것 -> 평가 기간에도 똑같이 만들 수 있음)
  1시간 전 / 1시간 후 예보 : 구름이 들어오는 중인지 빠지는 중인지
  그날 하루 평균 예보       : 하루 날씨 분위기
  A - B 구름 차이           : 두 회사 예보가 다를수록 불확실
결과마다 '보정 없음' + '시간별 배율' 둘 다 출력
"""
import argparse

import numpy as np
import pandas as pd

import baseline_validation as val
import baseline_optimizer as opt
import validation_timesplit as vt

ROOT = val.DEFAULT_ROOT
opt.log = lambda message: None

MODEL_A = "l1_cap_smpk"
MODEL_B = "cat_l1_cap_smpk"
WEIGHTS_A = [1.0, 0.75, 0.5, 0.25, 0.0]
GRID = np.arange(0.85, 1.2001, 0.005)

A_COLS = ["wa_cloud", "wa_rain", "wa_humidity"]
B_COLS = ["wb_cloud", "wb_uv_idx", "wb_total_liq"]


def add_time_feats(f, station, cols):
    """관측소 + 회차별로 1시간 전/후 값, 그날 평균 붙이기"""
    f = f.sort_values([station, "round", "ts"]).copy()
    g = f.groupby([station, "round"])
    new_cols = []
    for c in cols:
        f[c + "_prev"] = g[c].shift(1)
        f[c + "_next"] = g[c].shift(-1)
        new_cols.append(c + "_prev")
        new_cols.append(c + "_next")

    f["_day"] = (f["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    g2 = f.groupby([station, "round", "_day"])
    for c in cols:
        f[c + "_daymean"] = g2[c].transform("mean")
        new_cols.append(c + "_daymean")
    f = f.drop(columns="_day")
    return f, new_cols


def get_predictions(variant, base, sites, model_fit, eval_actual, folder):
    path = folder / (variant + "_eval_pred.parquet")
    if path.exists():
        print(variant, ": 저장된 예측 불러옴")
        return pd.read_parquet(path)

    print(variant, ": 학습 시작")
    fa, fb = val.load_weather(ROOT, base, model_fit["ts"].min())
    fa, _ = add_time_feats(fa, "wa_station", A_COLS)
    fb, _ = add_time_feats(fb, "wb_station", B_COLS)
    models, rounds, params = vt.fit_models_weighted(base, sites, model_fit, fa, fb, variant)
    pred = val.predict_eval(base, models, rounds, sites, eval_actual, fa, fb)
    pred.to_parquet(path, index=False)
    print(variant, ": 예측 저장 ->", path)
    return pred


def to_mats(actual, pred, site_ids):
    df = actual[["site_id", "ts", "energy"]].merge(pred, on=["site_id", "ts"])
    P = df.pivot(index="ts", columns="site_id", values="pred_kwh").reindex(columns=list(site_ids)).sort_index()
    Y = df.pivot(index="ts", columns="site_id", values="energy").reindex(columns=list(site_ids)).sort_index()
    return P.fillna(0).to_numpy(), Y.fillna(0).to_numpy(), pd.DatetimeIndex(P.index)


def best_c(P, Y, w):
    ps = P.sum(axis=1)
    ys = Y.sum(axis=1)
    best, best_loss = 1.0, None
    for c in GRID:
        loss = np.sum(w * np.abs(c * ps - ys))
        if best_loss is None or loss < best_loss:
            best, best_loss = c, loss
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    base = val.load_module(ROOT / "baseline.py", "oibc_baseline_feat")

    # 피처 목록에 새 피처 추가 + A-B 구름 차이
    new_feats = []
    for c in A_COLS + B_COLS:
        new_feats = new_feats + [c + "_prev", c + "_next", c + "_daymean"]
    new_feats.append("cloud_diff")
    base.FEATS = list(base.FEATS) + new_feats

    original_features = base.features

    def features_plus(*a, **k):
        d = original_features(*a, **k)
        d["cloud_diff"] = d["wa_cloud"] - d["wb_cloud"]
        return d

    base.features = features_plus
    print("피처 수:", len(base.FEATS))

    name = "ens_feat_full" if args.full else "ens_feat"
    folder = ROOT / "output" / "random_ens" / name
    folder.mkdir(parents=True, exist_ok=True)
    print("저장 위치:", folder)

    sites = pd.read_csv(ROOT / "sites.csv")[[
        "site_id", "current_group", "capacity_kw", "lat", "lon",
        "wa_station", "wb_station"]]

    gen = pd.read_parquet(ROOT / "train" / "generation.parquet",
                          columns=["site_id", "ts", "energy"])
    gen["ts"] = val.to_kst(gen["ts"])
    baseline_gen = gen[gen["ts"] >= base.TRAIN_FROM].copy()
    model_fit, group_actual, hold_actual, day_split = val.split_602020(baseline_gen, val.SEED)
    if args.full:
        old = gen[gen["ts"] < base.TRAIN_FROM].copy()
        model_fit = pd.concat([old, model_fit], ignore_index=True)
        print("이전 데이터 추가:", len(old), "행")
    del gen, baseline_gen

    eval_actual = pd.concat([group_actual, hold_actual], ignore_index=True)

    pred_a = get_predictions(MODEL_A, base, sites, model_fit, eval_actual, folder)
    pred_b = get_predictions(MODEL_B, base, sites, model_fit, eval_actual, folder)
    del model_fit

    both = pred_a.merge(pred_b, on=["site_id", "ts"], suffixes=("_a", "_b"))
    init_mask = opt.initial_mask(sites)
    rows = []
    for wa in WEIGHTS_A:
        pred = both[["site_id", "ts"]].copy()
        pred["pred_kwh"] = wa * both["pred_kwh_a"] + (1 - wa) * both["pred_kwh_b"]
        label = "LGBM " + str(int(wa * 100)) + "% + Cat " + str(int((1 - wa) * 100)) + "%"

        pred_h = pred.merge(hold_actual[["site_id", "ts"]], on=["site_id", "ts"])
        m_h = val.model_metrics(ROOT, hold_actual, pred_h)

        Pg, Yg, tg = to_mats(group_actual, pred, sites.site_id)
        Ph, Yh, th = to_mats(hold_actual, pred, sites.site_id)
        wg = opt.time_weights(ROOT, tg)

        c_hour = {}
        for h in range(24):
            idx = (tg.hour == h)
            if idx.sum() == 0:
                c_hour[h] = 1.0
            else:
                c_hour[h] = best_c(Pg[idx], Yg[idx], wg[idx])

        for how in ["보정 없음", "시간별 배율"]:
            if how == "보정 없음":
                cg, ch = 1.0, 1.0
            else:
                cg = np.array([c_hour[h] for h in tg.hour])[:, None]
                ch = np.array([c_hour[h] for h in th.hour])[:, None]
            mat_g = cg * Pg - Yg
            mat_h = ch * Ph - Yh
            optimized, meta, trace = opt.greedy_best_improvement(mat_g, wg, sites, init_mask)
            score_h = val.score_group(opt, ROOT, mat_h, th, init_mask, optimized)

            row = {
                "blend": label,
                "fix": how,
                "weighted_MAE": round(m_h["weighted_MAE_kwh"], 5) if how == "보정 없음" else None,
                "initial": round(score_h["W_initial_krw"]),
                "optimized": round(score_h["W_optimized_krw"]),
                "lower_bound": round(score_h["W_lower_bound_krw"]),
            }
            rows.append(row)
            print(label, how, "| 초기", f"{row['initial']:,}", "| 최적", f"{row['optimized']:,}",
                  "| 하한", f"{row['lower_bound']:,}")

    table = pd.DataFrame(rows)
    table.to_csv(folder / "result.csv", index=False)
    print("=" * 70)
    print("HOLDOUT 결과  목표: 최적 67,243,635 보다 작게  (지금 최고 67,372,995)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
