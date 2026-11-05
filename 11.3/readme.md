# OIBC Experiments — 11.3

2026-11-03 실험 보관용입니다. 아래 **No.는 Google Sheets `Results` → `Random Day Split`의 번호**이며, 코드 파일명에 포함된 번호와 다릅니다.

## Results 번호 ↔ 실행 코드

| Results No. | 실험 내용 | 원래 실행 코드 파일명 |
|---|---|---|
| **21** | Haurwitz + Faiman + PVWatts, 운량 미고려 | `physical_clear_v2.py` |
| **22** | Haurwitz + Kasten–Czeplak + Faiman + PVWatts, 운량 고려 | `physical_cloud.py` |
| **23** | No.17 + 전체 기간 + Rotation Features + XGBoost | `alpha_random17_fullhistory_rotation_xgb_v2.py` |
| **24** | Weather Distribution Estimation + `wa_dist`, `wb_dist` | `no_11_distance.py` |
| **25** | Weather Distribution Estimation + 1시간 예보 차분 | `no_11_difference_1.py` |
| **26** | Weather Distribution Estimation + 1·2시간 예보 차분 | `no_11_difference_2.py` |
| **31** | No.28 + 전국 운량 PCA 8개 (WA4/WB4) | `nat_pca.py` (원래 실행), `nat_pca_resume.py` (중단 후 이어서 실행) |

## 함께 보관할 파일

- `baseline.py` — 기본 발전량 Feature·모델.
- `baseline_optimizer.py` — Greedy 그룹 배정 및 페널티 평가.
- `baseline_validation.py`, `validation_timesplit.py` — 공통 검증/학습 로직을 참조하는 경우 필요.
- `smp_k_loss.py` — No.21·22 물리 Feature 실험의 기준 구현.
- `weather_quantile_v1.py` — No.24~26이 확장한 기상분포 예측 원형. **No.24~26의 실행 코드와는 별개**입니다.
- `nat_pca.py` / `nat_pca_resume.py` — No.31의 전체 실행 및 재시작 경로. 재시작에는 기존 출력 폴더의 학습·예측 캐시가 필요할 수 있습니다.

위 파일 중 일부는 원본 스크립트가 내부에서 직접 `import`하지 않고 독립적으로 구현한 **참고·재현 기준 파일**입니다. 정확한 실행 의존성은 해당 `.py`의 `import` 및 데이터 경로를 기준으로 확인해야 합니다.

> **주의:** 이 폴더는 코드 보관용입니다. 공모전의 발전량·기상·정산 데이터와 OOF 예측 Parquet 등은 GitHub에 업로드하지 마세요. 실제 실행은 기존 로컬 `data/` 구조에서 수행합니다.
