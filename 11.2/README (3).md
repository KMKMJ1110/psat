# OIBC Experiments — Results No.2–11, No.14

Results 시트의 `Random Day Split` 실험 번호와 원래 실행했던 Python 코드의 대응표입니다.

## 실험별 실행 파일

| Results No. | 실험 | 실행 파일 |
|---|---|---|
| 2 | Entire Period (전체 학습기간 사용) | `baseline_validation_entire_period.py` |
| 3 | MAE Loss | `baseline_validation_mae_loss.py` |
| 4 | Round 17 Only | `baseline_validation_only_round17.py` |
| 5 | Linear Regression | `baseline_validation_linear_regression.py` |
| 6 | MAE Loss + MILP (5분) | `baseline_validation_mae_loss_milp.py` |
| 7 | Posthoc — 관측소별 통계 보정 | `posthoc_stats.py` |
| 8 | Posthoc — 관측소 예보점 기반 보정 | `posthoc_observation_points.py` |
| 9 | Posthoc — Linear Regression 보정 | `posthoc_lr_model.py` |
| 10 | SMP × k 가중 Loss | `smp_k_loss.py` |
| 11 | SMP × k Loss + Weather Distribution Estimation | `weather_quantile_v1.py` |
| 14 | XGBoost Baseline | `xgb_baseline_602020.py` |

> **주의:** Results의 No.9(`posthoc_lr_model.py`)는 No.5(`baseline_validation_linear_regression.py`)와 서로 다른 실험입니다. No.11은 `weather_quantile_v1.py`이며, 후속 `spatial_weather_smp_k_loss.py`와 다른 코드입니다. No.14도 후속 Alpha 실험인 `alpha_random17_fullhistory_rotation_xgb_v2.py`와 구분합니다.

## 함께 필요한 코드·파일

- `baseline.py` — 프로젝트 데이터 루트(`data/`)의 공통 Feature 및 데이터 처리 코드.
- `baseline_validation.py` — 공통 Random Day Split 검증 실행기. 일부 실행 코드의 공통 기준 및 기본 예측 결과 생성용.
- `baseline_optimizer.py` — 그룹 배분 Greedy 및 정산금 페널티 평가.
- **No.7·8·9:** `posthoc_stats.py`가 공통 함수 제공자이며, No.8과 No.9를 실행할 때도 함께 보관해야 합니다. No.7~9는 기본 Baseline 검증 예측 결과를 사용하는 후처리 실험입니다.
- **No.11:** `weather_quantile_v1.py`는 자체 기상분포(분위수) 추정 코드를 포함합니다. 학습기간 기상 실측 데이터가 로컬에 있어야 합니다.
- 실행에 필요한 원본·전처리 데이터는 프로젝트 로컬 경로에 보관합니다. **대회 제공 데이터 파일은 GitHub에 업로드하지 않습니다.**

## 실행 예시

프로젝트 루트 `/Users/mj/Desktop/PSAT_2/con/data`에서 실행합니다.

```bash
python3 codes/baseline_validation.py
python3 codes/baseline_validation_entire_period.py
python3 codes/baseline_validation_mae_loss.py
python3 codes/baseline_validation_only_round17.py
python3 codes/baseline_validation_linear_regression.py
python3 codes/baseline_validation_mae_loss_milp.py
python3 codes/posthoc_stats.py
python3 codes/posthoc_observation_points.py
python3 codes/posthoc_lr_model.py
python3 codes/smp_k_loss.py
python3 codes/weather_quantile_v1.py
python3 codes/xgb_baseline_602020.py
```

코드마다 독립 실행하는 예시입니다. 실제 파일 위치와 실행 순서, MILP 솔버 등의 패키지 의존성은 로컬 코드 설정에 따릅니다. 코드를 GitHub의 날짜별 폴더에 보관하더라도 기존 실행 경로 및 `import` 경로는 별도로 확인해야 합니다.
