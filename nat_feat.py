"""피처 추가(19개) + 전국 날씨 패턴 피처(8개).
위치: codes/nat_feat.py
실행:
  python3 codes/nat_feat.py --split random    # 팀 표용 (랜덤 seed 42, 2023-06 이전 데이터도 학습)
  python3 codes/nat_feat.py --split time      # 팀 제출 기준 (2024-06 이전 학습, 마지막 1년 짝수 주 최적화 / 홀수 주 평가)
  --no-nat 을 붙이면 전국 패턴 8개 없이 (피처 19개만)
전국 패턴 = 시각마다 전국 관측소 예보 요약 (같은 회차 예보에서 계산 -> 평가 기간에도 똑같이 만들 수 있음)
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


def add_national(fa, fb):
    """시각 + 회차마다 전국 관측소 요약을 붙이고, 내 관측소 - 전국 평균 차이도 계산"""
    na = fa.groupby(["round", "ts"]).agg(a_nat_cloud_mean=("wa_cloud", "mean"),
                                         a_nat_cloud_std=("wa_cloud", "std"),
                                         a_nat_rain_mean=("wa_rain", "mean")).reset_index()
    nb = fb.groupby(["round", "ts"]).agg(b_nat_cloud_mean=("wb_cloud", "mean"),
                                         b_nat_cloud_std=("wb_cloud", "std"),
                                         b_nat_uv_mean=("wb_uv_idx", "mean")).reset_index()
    fa = fa.merge(na, on=["round", "ts"], how="left")
    fb = fb.merge(nb, on=["round", "ts"], how="left")
    fa["a_cloud_dev"] = fa["wa_cloud"] - fa["a_nat_cloud_mean"]
    fb["b_cloud_dev"] = fb["wb_cloud"] - fb["b_nat_cloud_mean"]
    return fa, fb


NAT_FEATS = ["a_nat_cloud_mean", "a_nat_cloud_std", "a_nat_rain_mean", "a_cloud_dev",
             "b_nat_cloud_mean", "b_nat_cloud_std", "b_nat_uv_mean", "b_cloud_dev"]


def get_predictions(variant, base, sites, model_fit, eval_actual, folder):
    path = folder / (variant + "_eval_pred.parquet")
    if path.exists():
        print(variant, ": 저장된 예측 불러옴")
        return pd.read_parquet(path)

    print(variant, ": 학습 시작")
    fa, fb = val.load_weather(ROOT, base, model_fit["ts"].min())
    fa, _ = add_time_feats(fa, "wa_station", A_COLS)
    fb, _ = add_time_feats(fb, "wb_station", B_COLS)
    fa, fb = add_national(fa, fb)
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
    ap.add_argument("--split", default="random", choices=["random", "time"])
    ap.add_argument("--no-nat", action="store_true", help="전국 패턴 피처 빼기")
    args = ap.parse_args()

    base = val.load_module(ROOT / "baseline.py", "oibc_baseline_feat")

    # 피처 목록에 새 피처 추가 + A-B 구름 차이
    new_feats = []
    for c in A_COLS + B_COLS:
        new_feats = new_feats + [c + "_prev", c + "_next", c + "_daymean"]
    new_feats.append("cloud_diff")
    if not args.no_nat:
        new_feats = new_feats + NAT_FEATS
    base.FEATS = list(base.FEATS) + new_feats

    original_features = base.features

    def features_plus(*a, **k):
        d = original_features(*a, **k)
        d["cloud_diff"] = d["wa_cloud"] - d["wb_cloud"]
        return d

    base.features = features_plus
    print("피처 수:", len(base.FEATS))

    name = args.split
    if args.no_nat:
        name = name + "_nonat"
    folder = ROOT / "output" / "nat_feat" / name
    folder.mkdir(parents=True, exist_ok=True)
    print("저장 위치:", folder)

    sites = pd.read_csv(ROOT / "sites.csv")[[
        "site_id", "current_group", "capacity_kw", "lat", "lon",
        "wa_station", "wb_station"]]

    gen = pd.read_parquet(ROOT / "train" / "generation.parquet",
                          columns=["site_id", "ts", "energy"])
    gen["ts"] = val.to_kst(gen["ts"])
    if args.split == "random":
        baseline_gen = gen[gen["ts"] >= base.TRAIN_FROM].copy()
        model_fit, group_actual, hold_actual, day_split = val.split_602020(baseline_gen, val.SEED)
        old = gen[gen["ts"] < base.TRAIN_FROM].copy()
        model_fit = pd.concat([old, model_fit], ignore_index=True)
        print("이전 데이터 추가:", len(old), "행")
        del baseline_gen
    else:
        model_fit, valid, group_actual, hold_actual = vt.split_year(gen)
    del gen

    eval_actual = pd.concat([group_actual, hold_actual], ignore_index=True)
    print("학습 행:", len(model_fit), " 평가 행:", len(eval_actual))

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
        m_h = val.model_metrics(ROOT, hold_actual, pred_h)   # random: holdout / time: 홀수 주

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
    if args.split == "random":
        print("HOLDOUT 결과  비교: 피처 19개 50:50 보정 없음 64,967,844 / 25:75 시간별 배율 64,336,509")
    else:
        print("그룹 평가(홀수 주) 결과  비교: baseline 최적 113,298,226  (time_feat.py 결과와도 비교)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
