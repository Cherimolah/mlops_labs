import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
REFERENCE_PATH = os.path.join(DATA_DIR, "reference.csv")
CURRENT_PATH = os.path.join(DATA_DIR, "train.csv")
REPORT_PATH = os.path.join(BASE, "drift_report.json")
RETRAIN_FLAG = os.path.join(BASE, "RETRAIN_REQUIRED")

NUMERIC_FEATURES = [
    "full_sq", "life_sq", "floor", "max_floor",
    "build_year", "num_room", "kitch_sq",
]
CATEGORICAL_FEATURES = ["sub_area"]
TARGET = "price_doc"

# Пороги принятия решения.
TARGET_PSI_THRESHOLD = 0.20
FEATURE_PSI_STRONG = 0.25
FEATURE_PSI_MODERATE = 0.10

EPS = 1e-6


def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI для числовой переменной. Бины строим по квантилям эталона."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Границы бинов — по квантилям эталонного распределения.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:  # вырожденный (константный) признак
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_perc = exp_counts / exp_counts.sum()
    act_perc = act_counts / act_counts.sum()

    return _psi_from_dist(exp_perc, act_perc)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI для категориального признака — по долям категорий."""
    categories = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    exp_perc = np.array([(expected == c).mean() for c in categories])
    act_perc = np.array([(actual == c).mean() for c in categories])
    return _psi_from_dist(exp_perc, act_perc)


def _psi_from_dist(exp_perc: np.ndarray, act_perc: np.ndarray) -> float:
    exp_perc = np.clip(exp_perc, EPS, None)
    act_perc = np.clip(act_perc, EPS, None)
    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def jensen_shannon_distance(expected: np.ndarray, actual: np.ndarray, bins: int = 30) -> float:
    """Jensen–Shannon distance (sqrt от дивергенции) для числовой переменной, [0, 1]."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)

    p, _ = np.histogram(expected, bins=edges, density=False)
    q, _ = np.histogram(actual, bins=edges, density=False)
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(math.sqrt(max(div, 0.0)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum(p * np.log(p / q)))


def psi_label(psi: float) -> str:
    if psi < FEATURE_PSI_MODERATE:
        return "stable"
    if psi < FEATURE_PSI_STRONG:
        return "moderate"
    return "significant"

def analyze(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    features_report = {}

    for col in NUMERIC_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        exp = pd.to_numeric(reference[col], errors="coerce").to_numpy(dtype=float)
        act = pd.to_numeric(current[col], errors="coerce").to_numpy(dtype=float)
        psi = psi_numeric(exp, act)
        ks_stat, ks_p = ks_2samp(
            exp[~np.isnan(exp)], act[~np.isnan(act)]
        ) if exp.size and act.size else (0.0, 1.0)
        features_report[col] = {
            "type": "numeric",
            "psi": round(psi, 4),
            "psi_label": psi_label(psi),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "ks_significant": bool(ks_p < 0.05),
        }

    for col in CATEGORICAL_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        psi = psi_categorical(reference[col].astype(str), current[col].astype(str))
        features_report[col] = {
            "type": "categorical",
            "psi": round(psi, 4),
            "psi_label": psi_label(psi),
        }

    target_report = {}
    if TARGET in reference.columns and TARGET in current.columns:
        exp_t = pd.to_numeric(reference[TARGET], errors="coerce").to_numpy(dtype=float)
        act_t = pd.to_numeric(current[TARGET], errors="coerce").to_numpy(dtype=float)
        target_psi = psi_numeric(exp_t, act_t)
        js = jensen_shannon_distance(exp_t, act_t)
        target_report = {
            "psi": round(target_psi, 4),
            "psi_label": psi_label(target_psi),
            "jensen_shannon_distance": round(js, 4),
        }
    else:
        target_psi = 0.0

    # --- Решение о переобучении ---
    psi_values = [v["psi"] for v in features_report.values()]
    n_strong = sum(p > FEATURE_PSI_STRONG for p in psi_values)
    n_moderate = sum(p > FEATURE_PSI_MODERATE for p in psi_values)

    reasons = []
    if target_psi > TARGET_PSI_THRESHOLD:
        reasons.append(f"PSI таргета {target_psi:.3f} > {TARGET_PSI_THRESHOLD}")
    if n_strong >= 1:
        strong_cols = [c for c, v in features_report.items() if v["psi"] > FEATURE_PSI_STRONG]
        reasons.append(f"сильный сдвиг признаков: {strong_cols}")
    if psi_values and n_moderate >= math.ceil(len(psi_values) / 2):
        reasons.append(
            f"{n_moderate} из {len(psi_values)} признаков со сдвигом PSI > {FEATURE_PSI_MODERATE}"
        )

    drift_detected = len(reasons) > 0

    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features": features_report,
        "target": target_report,
        "summary": {
            "mean_feature_psi": round(float(np.mean(psi_values)), 4) if psi_values else 0.0,
            "max_feature_psi": round(float(np.max(psi_values)), 4) if psi_values else 0.0,
            "features_significant_shift": n_strong,
            "features_moderate_shift": n_moderate,
        },
        "drift_detected": drift_detected,
        "reasons": reasons,
    }


def main() -> int:
    # Сбрасываем маркер с прошлого запуска.
    if os.path.exists(RETRAIN_FLAG):
        os.remove(RETRAIN_FLAG)

    if not os.path.exists(CURRENT_PATH):
        print(f"[drift] Нет свежих данных: {CURRENT_PATH}", file=sys.stderr)
        return 1

    if not os.path.exists(REFERENCE_PATH):
        # Первый запуск: эталона ещё нет — сравнивать не с чем.
        # Считаем, что нужно обучиться с нуля.
        report = {
            "drift_detected": True,
            "reasons": ["эталонный датасет reference.csv отсутствует — первичное обучение"],
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        open(RETRAIN_FLAG, "w").close()
        print("Эталон отсутствует -> требуется первичное обучение.")
        return 10

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    report = analyze(reference, current)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["drift_detected"]:
        print("\n*** ОБНАРУЖЕН СДВИГ ДАННЫХ ***")
        for r in report["reasons"]:
            print(f"  - {r}")
        open(RETRAIN_FLAG, "w").close()
        return 10

    print("\nСдвига данных не обнаружено — переобучение не требуется.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
