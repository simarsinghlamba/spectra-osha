"""Metrics with bootstrap CIs, McNemar, paired bootstrap, BH, point-biserial correlation."""
import numpy as np
import scipy.sparse as sp
from scipy import stats as st
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

def macro_f1(y, p):    return f1_score(y, p, average="macro")
def weighted_f1(y, p): return f1_score(y, p, average="weighted")

def boot_ci(y, p, metric, n_boot=1000, seed=42):
    y, p = np.asarray(y), np.asarray(p)
    rng, n = np.random.default_rng(seed), len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            vals.append(metric(y[i], p[i]))
        except ValueError:
            pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def clf_report(model, split, y, p):
    lo, hi = boot_ci(y, p, macro_f1)
    return dict(model=model, split=split, n=len(y), macro_f1=macro_f1(y, p), macro_f1_lo=lo,
                macro_f1_hi=hi, weighted_f1=weighted_f1(y, p), accuracy=accuracy_score(y, p))

def bin_report(model, split, y, s):
    lo, hi = boot_ci(y, s, roc_auc_score)
    return dict(model=model, split=split, n=len(y), auroc=roc_auc_score(y, s), auroc_lo=lo,
                auroc_hi=hi, pr_auc=average_precision_score(y, s), prevalence=float(np.mean(y)))

def mcnemar_test(y, p1, p2):
    from statsmodels.stats.contingency_tables import mcnemar
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    c1, c2 = p1 == y, p2 == y
    table = [[int(np.sum(c1 & c2)), int(np.sum(c1 & ~c2))],
             [int(np.sum(~c1 & c2)), int(np.sum(~c1 & ~c2))]]
    r = mcnemar(table, exact=False, correction=True)
    return dict(only_A_right=table[0][1], only_B_right=table[1][0], stat=float(r.statistic), p=float(r.pvalue))

def paired_boot_diff(y, p1, p2, metric=macro_f1, n_boot=1000, seed=42):
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    rng, n = np.random.default_rng(seed), len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(metric(y[i], p1[i]) - metric(y[i], p2[i]))
    d = np.array(d)
    return dict(diff=float(metric(y, p1) - metric(y, p2)), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))

def bh(pvals):
    from statsmodels.stats.multitest import multipletests
    return multipletests(np.asarray(pvals), method="fdr_bh")[1]

def point_biserial(X, y):
    """Correlation of every feature (column of X) with a 0/1 target. Returns (r, p)."""
    X = sp.csr_matrix(X)
    y = np.asarray(y).astype(bool)
    n, n1 = len(y), int(y.sum()); n0 = n - n1
    m1 = np.asarray(X[y].mean(0)).ravel()
    m0 = np.asarray(X[~y].mean(0)).ravel()
    mu = np.asarray(X.mean(0)).ravel()
    ex2 = np.asarray(X.multiply(X).mean(0)).ravel()
    sd = np.sqrt(np.maximum(ex2 - mu ** 2, 1e-12))
    r = np.clip((m1 - m0) / sd * np.sqrt(n1 * n0) / n, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * st.t.sf(np.abs(t), n - 2)
    return r, p


def div_from_proba(P, classes, class_div, divs):
    """Sum 2-digit class probabilities into divisions ('other' ignored, then renormalised)."""
    P = np.asarray(P, dtype=float)
    M = np.zeros((len(classes), len(divs)))
    for i, c in enumerate(classes):
        if class_div.get(c) is not None:
            M[i, divs.index(class_div[c])] = 1.0
    Q = P @ M
    Q = Q / np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return Q.argmax(1), Q
