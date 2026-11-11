# CAMS Total AOD 550 nm 검증

CAMS ADS에서 `total_aerosol_optical_depth_550nm` 운영 예보를 요청합니다. ECMWF 변수 번호는 `210207`입니다. 기존 황사 실험과 같은 위도 33–39°N, 경도 124–131.5°E 영역, 대상일 이틀 전 12 UTC 회차, +27~51시간 예보를 사용합니다. 예보 구간 양끝을 격자별로 평균한 뒤 공간 평균과 P90을 계산합니다. 같은 날 같은 회차의 직전·직후 한 시간 값까지 총 6개 피처를 저장합니다. 시각별 값은 모든 발전소에 공통으로 사용합니다. [ECMWF 변수 명세](https://confluence.ecmwf.int/spaces/CKB/pages/212454117/CAMS%2BGlobal%2Batmospheric%2Bcomposition%2Bforecast%2Bdata%2Bdocumentation)

## 실행

아래 두 경로에 기존 팀 프로젝트와 이 폴더를 지정한 뒤 PowerShell에서 실행합니다. 기존 CAMS 다운로드에 사용한 ADS API 설정과 이용 조건 동의, 기존 검증용 가상환경을 사용합니다.

```powershell
$data = 'C:\Users\82107\OneDrive\바탕 화면\data'
$kit = 'C:\Users\82107\OneDrive\바탕 화면\OIBC_2026_DATA\handover 2\01_participant_dataset\cams_total_aod'
$py = Join-Path $data '.venv\Scripts\python.exe'

& $py -B (Join-Path $kit 'build_cams_total_aod.py') --root $data
```

기본 기간은 발전 대상일 2021-01-19부터 2026-05-31까지입니다. 중간에 끊기면 같은 명령으로 저장된 GRIB 묶음을 재사용합니다. 이미 전체 GRIB가 있으면 `--local-only`로 가공할 수 있습니다. API 요청 변수와 회차만 확인하려면 `--dry-run`을 붙입니다. `--start`와 `--end`로 짧은 기간을 지정하면 날짜가 들어간 별도 피처 파일에 저장합니다.

피처와 출처 JSON이 생기면 아래 명령으로 검증합니다. 새 PowerShell 창에서도 실행할 수 있습니다.

```powershell
& 'C:\Users\82107\OneDrive\바탕 화면\data\.venv\Scripts\python.exe' -B 'C:\Users\82107\OneDrive\바탕 화면\OIBC_2026_DATA\handover 2\01_participant_dataset\cams_total_aod\validate_cams_total_aod.py' --root 'C:\Users\82107\OneDrive\바탕 화면\data' --w 0.5 0 0.5
```

팀 프로젝트의 기존 `codes/gfs_tree.py` 학습·채점 함수, `output/external/gfs_features.parquet`, `output/site_tabpfn/random_big/tabpfn_eval_pred.parquet`을 읽습니다. `gfs`, `total_aod2`, `total_aod6`에 같은 GFS 15개와 같은 트리 50%·TabPFN 50% 비율을 적용합니다. 피처 2개는 현재 평균과 P90, 피처 6개는 이 둘에 직전·직후 한 시간의 평균과 P90을 더한 구성입니다.

`cams_delay_exclusions.json`은 황사 실험에서 쓴 것과 동일합니다. 전체 기간의 제외 대상일 20일은 시간 행을 유지하고 Total AOD 값만 비웁니다. 검증에 해당 시각이 포함되면 CAMS 조합의 트리 예측을 GFS 단독 예측으로 대체합니다.

## 저장 위치

| 파일 | 팀 프로젝트 `data` 아래 위치 |
|---|---|
| Total AOD 원본 GRIB | `output/external/cams_raw_total_aod/cams_total_aod_<시작일>_<종료일>.grib` |
| 모델 입력 피처 | `output/external/cams_total_aod_features_final.parquet` |
| 출처·회차·제외 기록 | `output/external/cams_total_aod_features_final.provenance.json` |
| 검증 점수 | `output/feat_gfs_cams_total_aod/run_<실행시각>_<ID>/result.csv` |
| 독립 검증 점수 | 같은 실행 폴더의 `holdout.csv` |

다운로드 파일은 기존 Total AOD 캐시를 읽고, 같은 이름의 피처와 JSON이 있으면 덮어쓰기 전에 중단합니다. 검증을 다시 실행하면 새 실행 폴더에 모델별 예측과 점수표를 저장합니다. `result.csv`의 `group_opt_optimized`로 세 조합을 비교합니다. `holdout.csv`의 `optimized`로 선택 조합과 GFS 기준을 비교합니다. 정산 오차이므로 낮을수록 좋습니다.
