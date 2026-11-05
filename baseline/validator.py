"""2026 OIBC Big Data Challenge — 제출 파일 형식 검증 시스템.

    python validate_submission.py --composition composition.csv --forecast forecast.csv

제출 포털과 같은 형식 규칙(무효 조건 1~6)으로 두 파일을 검사합니다.
채점용 실적은 포함되어 있지 않으므로 **점수는 계산하지 않습니다.** 통과하면 포털에서도
형식 때문에 무효 처리되지 않습니다. 실패하면 무엇이 몇 건 잘못됐는지 모두 알려 줍니다.

    종료 코드  0 = 통과 · 1 = 무효(형식 위반) · 2 = 파일을 읽을 수 없음

`--data` 는 sites.csv · smp.csv 가 있는 디렉터리입니다(기본값: 이 스크립트가 있는 곳).
필요 패키지: pandas, numpy (.parquet 제출을 검사하려면 pyarrow).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

GROUPS = ("G1", "G2")
MIN_CAP = 20_000.0          # 집합별 최소 설비용량 (kW)
ATOL = 1e-9                 # 부동소수점 비교 허용오차
TZ = "Asia/Seoul"
TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
SHOW = 5                    # 오류마다 보여 줄 예시 개수


class Unreadable(Exception):
    """파일을 열 수 없거나 표 형식으로 읽을 수 없음."""


def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise Unreadable(f"파일이 없습니다: {path}")
    try:
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:   # 인코딩·구분자·손상 파일 등
        raise Unreadable(f"{path} 을(를) 읽을 수 없습니다 — {type(e).__name__}: {e}") from e


def _ex(items) -> str:
    items = list(items)
    head = ", ".join(str(x) for x in items[:SHOW])
    return head + (f" … 외 {len(items) - SHOW}건" if len(items) > SHOW else "")


def load_reference(data_dir: str):
    sites = pd.read_csv(os.path.join(data_dir, "sites.csv"))
    smp = pd.read_csv(os.path.join(data_dir, "smp.csv"))
    index = pd.DatetimeIndex(pd.to_datetime(smp.ts).dt.tz_convert(TZ))   # 채점 시간축 (8,760시각)
    return sites, index


# ------------------------------------------------------------------ composition.csv
def check_composition(path: str, sites: pd.DataFrame, errors: list, info: list):
    """무효 조건 1(site_id 253개 정확히 한 번씩) · 2(group 값) · 3(집합별 20,000 kW 이상)."""
    d = _read(path)
    miss_cols = {"site_id", "group"} - set(d.columns)
    if miss_cols:
        errors.append(f"[composition] 컬럼 누락: {sorted(miss_cols)} — 헤더는 정확히 site_id,group 이어야 합니다")
        return None

    sid = d.site_id.astype(str)          # 공백 등은 그대로 비교 (포털과 동일)
    want = set(sites.site_id)
    dup = sid[sid.duplicated(keep=False)].unique()
    if len(dup):
        errors.append(f"[composition] site_id 중복 {len(dup)}개: {_ex(sorted(dup))}")
    missing, unknown = sorted(want - set(sid)), sorted(set(sid) - want)
    if missing:
        errors.append(f"[composition] sites.csv 의 site_id 누락 {len(missing)}개: {_ex(missing)}")
    if unknown:
        errors.append(f"[composition] sites.csv 에 없는 site_id {len(unknown)}개: {_ex(unknown)}")

    grp = d.group.astype(str)
    bad = sorted(set(grp) - set(GROUPS))
    if bad:
        errors.append(f"[composition] group 값은 G1 · G2 만 허용 — 잘못된 값: {_ex(bad)}")

    if dup.size or missing or unknown or bad:
        return None
    z = pd.Series(grp.to_numpy(), index=sid.to_numpy())
    cap = sites.set_index("site_id").capacity_kw
    total = float(cap.sum())
    for g in GROUPS:
        members = z[z == g].index
        c = float(cap[members].sum())
        info.append(f"  {g}: {len(members):>3}개소  {c:>10,.1f} kW")
        if c < MIN_CAP - ATOL:
            errors.append(f"[composition] {g} 설비용량 {c:,.1f} kW < 최소 {MIN_CAP:,.0f} kW "
                          f"(한 집합은 {MIN_CAP:,.1f} ~ {total - MIN_CAP:,.1f} kW 범위여야 합니다)")
    return z


# ------------------------------------------------------------------ forecast.csv
def _parse_ts(raw: pd.Series, errors: list):
    """무효 조건 6: 모든 ts 에 타임존이 있어야 하며, 있으면 KST 로 변환한다(Z 표기도 유효)."""
    if pd.api.types.is_datetime64_any_dtype(raw):                     # parquet timestamp
        if raw.dt.tz is None:
            errors.append("[forecast] ts 에 타임존이 없습니다 — parquet 는 타임존이 붙은 timestamp 여야 합니다")
            return None
        return raw.dt.tz_convert(TZ)
    s = raw.astype(str).str.strip()
    naive = s[~s.str.contains(TZ_SUFFIX)]
    if len(naive):
        errors.append(f"[forecast] 타임존 표기가 없는 ts {len(naive):,}건: {_ex(naive.unique())} "
                      f"(예: 2025-06-01 02:00:00+09:00)")
        return None
    try:
        return pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_convert(TZ)   # +09:00 · Z 혼용도 허용
    except Exception as e:
        bad = []
        for v in s.unique():
            try:
                pd.to_datetime(v, format="ISO8601")
            except Exception:
                bad.append(v)
        errors.append(f"[forecast] 날짜로 해석할 수 없는 ts {len(bad):,}건: {_ex(bad) if bad else e}")
        return None


def check_forecast(path: str, index: pd.DatetimeIndex, errors: list):
    """무효 조건 4((group, ts) 17,520개 정확히) · 5(음수·NaN·무한대) · 6(타임존)."""
    d = _read(path)
    miss_cols = {"group", "ts", "forecast_kwh"} - set(d.columns)
    if miss_cols:
        errors.append(f"[forecast] 컬럼 누락: {sorted(miss_cols)} — 헤더는 정확히 group,ts,forecast_kwh 이어야 합니다")
        return None

    d = d[["group", "ts", "forecast_kwh"]].copy()
    d["group"] = d.group.astype(str)
    bad = sorted(set(d.group) - set(GROUPS))
    if bad:
        errors.append(f"[forecast] group 값은 G1 · G2 만 허용 — 잘못된 값: {_ex(bad)}")

    v = pd.to_numeric(d.forecast_kwh, errors="coerce")
    text = d.forecast_kwh[v.isna() & d.forecast_kwh.notna()]
    if len(text):
        errors.append(f"[forecast] 숫자가 아닌 forecast_kwh {len(text):,}건: {_ex(text.astype(str).unique())}")
    nan = int((v.isna() & d.forecast_kwh.isna()).sum())
    if nan:
        errors.append(f"[forecast] 비어 있거나 NaN 인 forecast_kwh {nan:,}건")
    inf = int(np.isinf(v).sum())
    if inf:
        errors.append(f"[forecast] 무한대 forecast_kwh {inf:,}건")
    neg = v[v < -ATOL]
    if len(neg):
        errors.append(f"[forecast] 음수 forecast_kwh {len(neg):,}건 (최솟값 {neg.min():,.3f})")
    d["forecast_kwh"] = v

    want = len(GROUPS) * len(index)
    if len(d) != want:
        errors.append(f"[forecast] 행 수 {len(d):,} ≠ {want:,} (= {len(GROUPS)}개 집합 × {len(index):,}시각, 헤더 제외)")
    ts = _parse_ts(d.ts, errors)
    if ts is None or bad:
        return None
    d["ts"] = ts

    dup = d[d.duplicated(["group", "ts"], keep=False)]
    if len(dup):
        errors.append(f"[forecast] (group, ts) 중복 {int(d.duplicated(['group', 'ts']).sum()):,}건: "
                      f"{_ex((dup.group + ' ' + dup.ts.astype(str)).unique())}")
    for g in GROUPS:
        got = pd.DatetimeIndex(d.loc[d.group == g, "ts"].unique())
        miss, extra = index.difference(got), got.difference(index)
        if len(miss):
            errors.append(f"[forecast] {g} 시각 누락 {len(miss):,}건: {_ex(miss.strftime('%Y-%m-%d %H:%M%z'))}")
        if len(extra):
            errors.append(f"[forecast] {g} 채점 시간축(smp.csv)에 없는 시각 {len(extra):,}건: "
                          f"{_ex(extra.strftime('%Y-%m-%d %H:%M%z'))}")
    return d


def advisories(comp: pd.Series, fcst: pd.DataFrame, sites: pd.DataFrame) -> list:
    """무효는 아니지만 실수일 가능성이 큰 것 (참고)."""
    out = []
    cap = sites.set_index("site_id").capacity_kw
    for g in GROUPS:
        limit = float(cap[comp[comp == g].index].sum())          # 1시간 최대 발전량 = 설비용량 kW × 1 h
        over = fcst[(fcst.group == g) & (fcst.forecast_kwh > limit + ATOL)]
        if len(over):
            out.append(f"[참고] {g} 예측 {len(over):,}시각이 집합 설비용량({limit:,.1f} kWh/h)을 넘습니다 — "
                       f"단위(kW · kWh · MWh)나 집합 배정을 확인하세요. 무효 사유는 아닙니다.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="2026 OIBC 제출 파일 형식 검증 (점수는 계산하지 않음)")
    ap.add_argument("--composition", required=True, help="composition.csv 또는 .parquet")
    ap.add_argument("--forecast", required=True, help="forecast.csv 또는 .parquet")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "OIBC2026_dataset"),
                    help="sites.csv · smp.csv 가 있는 디렉터리")
    a = ap.parse_args()

    try:
        sites, index = load_reference(a.data)
    except FileNotFoundError as e:
        print(f"[오류] 배포 데이터를 찾을 수 없습니다 — {e}. --data 로 sites.csv · smp.csv 위치를 지정하세요.")
        return 2

    errors, info = [], []
    try:
        comp = check_composition(a.composition, sites, errors, info)
        fcst = check_forecast(a.forecast, index, errors)
    except Unreadable as e:
        print(f"[오류] {e}")
        return 2

    if errors:
        print(f"[무효] 형식 위반 {len(errors)}건 — 이대로 제출하면 무효 처리됩니다.")
        for e in errors:
            print("  - " + e)
        return 1

    print(f"[통과] 제출 형식이 올바릅니다   채점 시각 {len(index):,}h ({index[0]} ~ {index[-1]})")
    print("\n".join(info))
    for m in advisories(comp, fcst, sites):
        print(m)
    print("점수는 제출 포털에서 확인할 수 있습니다 (이 도구는 형식만 검사합니다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
