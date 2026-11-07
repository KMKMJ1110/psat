# 파일명: run_my_group_exp.py (프로젝트 루트 폴더에 저장 후 python run_my_group_exp.py 실행)
from pathlib import Path
import pandas as pd
import codes.baseline_validation as val
import codes.baseline_optimizer as opt

root = Path(".")
base = val.load_module(root / "baseline.py", "base")
sites = pd.read_csv(root / "sites.csv")[
    ["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]
]

# 1. 팀 표준 검증 분할 (Random 60/20/20, seed=42) 날짜 가져오기
gen = pd.read_parquet(root / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
gen["ts"] = val.to_kst(gen.ts)
_, group_actual, holdout_actual, day_split = val.split_602020(
    gen.loc[gen.ts >= base.TRAIN_FROM].copy(), seed=42
)

# 2. 1단계에서 합친 TabPFN 예측 파일 불러오기
pred = pd.read_parquet("output/site_tabpfn/random/tabpfn_eval_pred.parquet")
pred["ts"] = val.to_kst(pred["ts"])

# 만약 예측값 컬럼 이름이 pred_kwh가 아니라면 자동으로 맞춰줌
for col in ("pred", "kwh", "forecast_kwh", "y_pred"):
    if col in pred.columns and "pred_kwh" not in pred.columns:
        pred = pred.rename(columns={col: "pred_kwh"})

# group_opt와 holdout 구간의 예측값 분리 매칭
group_pred = group_actual[["site_id", "ts"]].merge(
    pred[["site_id", "ts", "pred_kwh"]], on=["site_id", "ts"], how="left"
)
holdout_pred = holdout_actual[["site_id", "ts"]].merge(
    pred[["site_id", "ts", "pred_kwh"]], on=["site_id", "ts"], how="left"
)

# [추가됨] 3. 모델 예측 정확도 (MAE, RMSE, Weighted MAE) 계산
group_model = val.model_metrics(root, group_actual, group_pred)
holdout_model = val.model_metrics(root, holdout_actual, holdout_pred)

# 4. group_opt(20%)와 holdout(20%) 오차 행렬 생성
group_matrix, group_times, _ = val.residual_matrix(
    group_actual, group_pred, sites.site_id, "GROUP_OPT"
)
hold_matrix, hold_times, _ = val.residual_matrix(
    holdout_actual, holdout_pred, sites.site_id, "HOLDOUT"
)

# 5. 수정된 baseline_optimizer로 그룹핑 실행 및 채점
initial = opt.initial_mask(sites)
group_weights = opt.time_weights(root, group_times)
optimized, meta, _ = opt.greedy_best_improvement(group_matrix, group_weights, sites, initial)

group_score = val.score_group(opt, root, group_matrix, group_times, initial, optimized)
hold_score = val.score_group(opt, root, hold_matrix, hold_times, initial, optimized)

# 6. 최종 결과 출력 (시트 기록용 전체 지표)
improvable_cost = hold_score["W_initial_krw"] - hold_score["W_lower_bound_krw"]
improved_cost = hold_score["W_initial_krw"] - hold_score["W_optimized_krw"]
improved_pct_of_lb = (improved_cost / improvable_cost * 100) if improvable_cost else 0.0
krw_per_hour = hold_score["W_optimized_krw"] / hold_score["hours"] if hold_score["hours"] else 0.0

print("\n" + "=" * 72)
print(f"GROUP OPT     initial W       = {group_score['W_initial_krw']:,.2f} KRW")
print(f"GROUP OPT     optimized W     = {group_score['W_optimized_krw']:,.2f} KRW")
print(f"GROUP OPT     lower bound     = {group_score['W_lower_bound_krw']:,.2f} KRW")
print("-" * 72)
print(f"HOLDOUT MODEL MAE             = {holdout_model['MAE_kwh']:.5f} kWh")
print(f"HOLDOUT MODEL RMSE            = {holdout_model['RMSE_kwh']:.5f} kWh")
print(f"HOLDOUT MODEL weighted MAE    = {holdout_model['weighted_MAE_kwh']:.5f} kWh")
print("-" * 72)
print(f"HOLDOUT GROUP initial W       = {hold_score['W_initial_krw']:,.2f} KRW")
print(f"HOLDOUT GROUP optimized W     = {hold_score['W_optimized_krw']:,.2f} KRW")
print(f"HOLDOUT GROUP lower bound     = {hold_score['W_lower_bound_krw']:,.2f} KRW")
print(f"HOLDOUT GROUP W reduction     = {hold_score['W_reduction_krw']:,.0f} KRW ({hold_score['W_reduction_pct']:.4f}%)")
print(f"HOLDOUT Improvable (Init-LB)  = {improvable_cost:,.0f} KRW (달성률: {improved_pct_of_lb:.2f}%)")
print(f"HOLDOUT Penalty per Hour      = {krw_per_hour:,.0f} 원/h (총 {hold_score['hours']}시간)")
print("=" * 72)