# utils/metrics.py
import json
import numpy as np
from typing import Sequence, Dict, Tuple, List

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _to_float_array(x: Sequence) -> np.ndarray:
    return np.asarray(list(x), dtype=float)

def _to_int_array(x: Sequence) -> np.ndarray:
    return np.asarray(list(x), dtype=int)

def _to_int_array_round(x: Sequence) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    return np.rint(a).astype(int)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    yt = _to_int_array_round(y_true)
    yp = _to_int_array_round(y_pred)
    return float((yt == yp).mean()) if len(yt) else 0.0

def adjacent_accuracy(y_true: Sequence, y_pred: Sequence, tol: int = 1) -> float:
    yt = _to_int_array_round(y_true)
    yp = _to_int_array_round(y_pred)
    return float((np.abs(yt - yp) <= tol).mean()) if len(yt) else 0.0

def mae(y_true: Sequence, y_pred: Sequence) -> float:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    return float(np.abs(yt - yp).mean()) if len(yt) else 0.0

def mse(y_true: Sequence, y_pred: Sequence) -> float:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    return float(((yt - yp) ** 2).mean()) if len(yt) else 0.0

def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


# ---------- Kappa ----------
def cohen_kappa(y_true: Sequence, y_pred: Sequence, labels: Sequence[int] = None) -> float:
    yt = _to_int_array(y_true)
    yp = _to_int_array(y_pred)
    if labels is None:
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    L = len(labels)
    if L == 0:
        return 0.0
    # confusion
    idx = {v: i for i, v in enumerate(labels)}
    M = np.zeros((L, L), dtype=float)
    for a, b in zip(yt, yp):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1.0
    N = M.sum()
    if N == 0:
        return 0.0
    Po = np.trace(M) / N
    rows = M.sum(axis=1)
    cols = M.sum(axis=0)
    Pe = float((rows @ cols) / (N * N))
    return float((Po - Pe) / (1.0 - Pe)) if (1.0 - Pe) != 0 else 0.0


def quadratic_weighted_kappa(y_true: Sequence, y_pred: Sequence, max_rating: int = 3) -> float:
    yt = _to_int_array_round(y_true)
    yp = _to_int_array_round(y_pred)
    n_cat = max_rating + 1

    mask = (yt >= 0) & (yt <= max_rating) & (yp >= 0) & (yp <= max_rating)
    yt = yt[mask]; yp = yp[mask]

    O = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(yt, yp):
        O[a, b] += 1.0
    N = O.sum()
    if N == 0:
        return 0.0

    I = np.arange(n_cat)
    W = ((I[:, None] - I[None, :]) ** 2) / (max_rating ** 2)

    rows = O.sum(axis=1)
    cols = O.sum(axis=0)
    E = np.outer(rows, cols) / N

    num = (W * O).sum()
    den = (W * E).sum()
    return float(1.0 - (num / den if den != 0 else 1.0))


def pearson_corr(y_true: Sequence, y_pred: Sequence) -> float:
    yt = _to_float_array(y_true); yp = _to_float_array(y_pred)
    if len(yt) < 2:
        return float("nan")
    if _HAS_SCIPY:
        r, _ = pearsonr(yt, yp); return float(r)
    yt = yt - yt.mean(); yp = yp - yp.mean()
    denom = (np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()))
    return float((yt * yp).sum() / denom) if denom != 0 else float("nan")

def spearman_corr(y_true: Sequence, y_pred: Sequence) -> float:
    yt = _to_float_array(y_true); yp = _to_float_array(y_pred)
    if len(yt) < 2:
        return float("nan")
    if _HAS_SCIPY:
        r, _ = spearmanr(yt, yp); return float(r)
    # fallback: rank + Pearson
    def rankdata(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks
    return pearson_corr(rankdata(yt), rankdata(yp))

def kendall_tau_corr(y_true: Sequence, y_pred: Sequence) -> float:
    yt = _to_float_array(y_true); yp = _to_float_array(y_pred)
    if len(yt) < 2:
        return float("nan")
    if _HAS_SCIPY:
        r, _ = kendalltau(yt, yp); return float(r)
    n = len(yt); conc = disc = 0
    for i in range(n):
        for j in range(i+1, n):
            a = np.sign(yt[i] - yt[j])
            b = np.sign(yp[i] - yp[j])
            if a == 0 or b == 0: 
                continue
            if a == b: conc += 1
            else:      disc += 1
    denom = conc + disc
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence[int]) -> List[List[int]]:
    idx = {v: i for i, v in enumerate(labels)}
    yt = _to_int_array_round(y_true)
    yp = _to_int_array_round(y_pred)
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(yt, yp):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    return M.tolist()


def evaluate_all(y_true: Sequence, y_pred: Sequence, *, max_rating: int = 3, include_confusion: bool = False) -> Dict[str, float]:
    labels = list(range(max_rating + 1))
    res: Dict[str, float] = {
        "Accuracy": accuracy(y_true, y_pred),
        "AdjAccuracy": adjacent_accuracy(y_true, y_pred, tol=1),
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "CohenKappa": cohen_kappa(y_true, y_pred, labels=labels),
        "QWK": quadratic_weighted_kappa(y_true, y_pred, max_rating=max_rating),
        "Pearson": pearson_corr(y_true, y_pred),
        "Spearman": spearman_corr(y_true, y_pred),
        "KendallTau": kendall_tau_corr(y_true, y_pred),
    }
    if include_confusion:
        res["_Confusion"] = confusion_matrix(y_true, y_pred, labels=labels)
    return res


def save_metrics(path: str, metrics: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

