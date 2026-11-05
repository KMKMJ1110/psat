# OIBC 예측 실험 코드 (Results No.33~39, 41, 46)

이 폴더의 파일을 **Results 시트의 실험 번호(No.)**와 대응시킨 목록입니다. `newcand_01`, `rank15` 등 **파일명 속 번호는 Results의 No.가 아닙니다.**

## 1. 실험 번호와 실행 파일

| Results No. | 실험 내용 | 메인 실행 파일 |
|---|---|---|
| **33** | No.28 + 발전소별 전체·계절·시간대·오전/오후 평균 CF Feature | `newcand_01_site_response.py` |
| **34** | No.28 + 15-NN 유사 기상일 예측 앙상블 | `newcand_05_analog_ensemble.py` |
| **35** | No.28 + 그룹별 LightGBM 잔차 보정 | `newcand_06_bounded_local_global.py` |
| **36** | No.28 + 그룹 잔차 회귀 보정 (Total–Contrast 계열) | `newcand_08_total_contrast.py` |
| **37** | No.28 + Clear-Sky Proxy 대비 CF 타깃 변환 | `newcand_09_clearsky_target.py` |
| **38** | No.28 + 운고(Cloud Ceiling) Feature | `physcand_01_ceiling_only.py` |
| **39** | No.28 + 그룹별 조건부 잔차 분위수 LightGBM (q10/q50/q90) | `rank15_group_error_distribution.py` |
| **41** | No.28 + 동일 예측시각의 Round 17 − Round 10 기상 수정량 Feature | `physcand_06_round_revision.py` |
| **46** | No.28 + 상대습도 20~60% 조건부 발전소별 Ridge 잔차 보정·야간 안전 처리 | `no28_rh_night_safe_replay.py` |

※ No.40은 이 폴더의 대상에 포함되지 않습니다.

## 2. 추가로 필요한 공통 Python 파일

| 대상 | 함께 필요한 파일 | 역할 |
|---|---|---|
| **No.33~37** | `newcand_common.py`, `newcand_experiments.py` | 신규 후보 실험의 공통 기능·실행 로직 |
| **No.38, 41** | `physcand_suite.py` | 물리 Feature 후보의 공통 실험 엔진 |
| **No.39** | `rank11_15_common.py` | 그룹 잔차·분위수 실험의 공통 엔진 |
| **No.46** | `no28_rh_residual_models.py` | Ridge 보정 모델 생성·결과 캐시; `no28_rh_night_safe_replay.py`는 저장 결과의 재평가용 |
| **공통 기반** | `baseline.py`, `baseline_validation.py`, `baseline_optimizer.py` | 기본 예측, 데이터 분할·평가, Greedy 그룹 최적화 |

`newcand_common.py` 등 공통 모듈은 여러 실험이 공유하므로 **한 번만 보관하면 됩니다.** 실제 `import` 관계에 따라 위 목록 밖의 모듈이 필요할 수 있으며, 이 README는 파일의 내부 의존성을 전수 검증한 목록은 아닙니다.

## 3. 실행 시 주의사항

- 기존 로컬 프로젝트에서는 일반적으로 `data/codes/`에 Python 파일을 배치하고 `data/`에서 실행합니다. 예: `python3 codes/newcand_01_site_response.py`.
- **No.46의 Replay는 독립적인 최초 학습 스크립트가 아닙니다.** 기존 No.28 OOF 예측·잔차 및 Ridge 모델 결과 등이 없으면 먼저 관련 생성 스크립트를 실행해야 할 수 있습니다.
- No.28을 기반으로 하는 다른 후보도 로컬 `output/`의 **기존 OOF 예측·체크포인트**를 읽을 수 있습니다. Python 소스만 복사했다고 즉시 재현되는 것은 아닙니다.
- `sites.csv`, 학습 발전량·기상 Parquet, OOF 결과 파일 등 **대회 제공 데이터나 그 파생 데이터는 GitHub에 올리지 않습니다.** 로컬에서 보관합니다.

*정리 기준: Results 시트의 실험 정의 및 현재 업로드 대상으로 제시한 14개 파일. 개별 소스의 `import`와 캐시 경로는 추가 확인이 필요합니다.*
