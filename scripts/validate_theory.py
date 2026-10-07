#!/usr/bin/env python3
"""Exact rational GSO and numerical checks of key identities in the revised
manuscript (main_review.tex).  Writes validate_theory.json.

Checks
  T1  Per-swap T-transform (Theorem 1) on exact LLL traces.
  T2  Variance-dissipation identity (Proposition 4) on exact LLL traces,
      with epsilon taken as the labelled displacement of Theorem 1.
  T3  Two-sided GSA extremality (Proposition 3): p* < p on the lower-gap set,
      p < p* on the decreasing upper-gap set; random feasible profiles.
  T4  Sublevel lemma: ||x||_inf > (d-1) R0  =>  x0 strictly majorized by x.
  T5  Exact counterexamples: admissible deep insertion that raises SS and V;
      insertion that improves every Schur-convex score but lengthens b_1.
  T6  q-ary and Goldstein-Mayer raw-profile calibration CV0 and alpha0.
  T7  Thermal expansion  D_phi = 2 alpha^2 D_V + O(alpha^3).
  T8  Fixed-shift entropy along exact LLL traces.
  T9  Weighted class (ii) and Lovasz-compatibility on random T-transforms.
  T10 Two-position update (Eq. two_position_update) vs explicit insertion.
  T11 Integral Gram-determinant separation for the variance count bound.
  T12 Variance selectors, degenerate fallback, zero-variance exit, and gap boundary.
  Logarithms are numerical evaluations, not exact rational values or proofs.
"""
import json
import math
import random
from fractions import Fraction as Fr

import numpy as np

random.seed(1)
np.random.seed(1)
OUT = {}


# ---------------------------------------------------------------- helpers
def gso(B):
    """Exact Gram-Schmidt (rows), returns r (squared norms) and mu."""
    d = len(B)
    Bs, r = [], []
    mu = [[Fr(0)] * d for _ in range(d)]
    for i in range(d):
        v = [Fr(x) for x in B[i]]
        for j in range(i):
            mu[i][j] = sum(Fr(a) * b for a, b in zip(B[i], Bs[j])) / r[j]
            v = [a - mu[i][j] * b for a, b in zip(v, Bs[j])]
        Bs.append(v)
        r.append(sum(a * a for a in v))
    return r, mu


def logp(r):
    return [0.5 * math.log(float(x)) for x in r]


def size_reduce(B):
    d = len(B)
    for i in range(1, d):
        for j in range(i - 1, -1, -1):
            _, mu = gso(B)
            q = round(mu[i][j])
            if q:
                B[i] = [a - q * b for a, b in zip(B[i], B[j])]
    return B


def lll_trace(B, delta=Fr(99, 100)):
    """Textbook exact LLL. Records profile before/after every swap and mu."""
    B = [list(row) for row in B]
    d = len(B)
    k = 1
    trace = []
    while k < d:
        size_reduce(B)
        r, mu = gso(B)
        if r[k] < (delta - mu[k][k - 1] ** 2) * r[k - 1]:
            before = logp(r)
            m = mu[k][k - 1]
            B[k - 1], B[k] = B[k], B[k - 1]
            r2, _ = gso(B)
            trace.append((k, before, logp(r2), float(m), r, r2))
            k = max(k - 1, 1)
        else:
            k += 1
    return B, trace


def V(p):
    p = np.asarray(p, float)
    return float(((p - p.mean()) ** 2).sum())


def majorized(y, x, tol=1e-12):
    """True if y is majorized by x (equal sums assumed)."""
    xs, ys = np.sort(x)[::-1], np.sort(y)[::-1]
    return bool(np.all(np.cumsum(ys)[:-1] <= np.cumsum(xs)[:-1] + tol)
                and abs(xs.sum() - ys.sum()) < 1e-9)


# ---------------------------------------------------- T1, T2, T8: LLL traces
n_swaps = n_nondeg = n_deg = 0
max_id_err = 0.0
t1_ok = True
ent_ok = True
for trial in range(40):
    d = random.choice([4, 5, 6])
    B = [[random.randint(-30, 30) for _ in range(d)] for _ in range(d)]
    if random.random() < 0.3:          # q-ary flavour, forces large gaps
        q = 97
        k0 = d // 2
        B = [[q if (i == j and i < k0) else 0 for j in range(d)] for i in range(k0)] + \
            [[random.randrange(q) for _ in range(k0)] + [1 if j == i else 0 for j in range(k0, d)]
             for i in range(k0, d)]
    from fractions import Fraction
    r0, _ = gso(B)
    if any(x == 0 for x in r0):
        continue
    _, tr = lll_trace(B)
    p0 = tr[0][1] if tr else None
    if p0 is None:
        continue
    C = -min(p0) + 1.0
    L = sum(p0)
    def H(p):
        u = (np.asarray(p) + C) / (L + len(p) * C)
        return float(-(u * np.log(u)).sum())
    for (k, a, b, m, r, r2) in tr:
        n_swaps += 1
        A, Bv = a[k - 1], a[k]
        A2, B2 = b[k - 1], b[k]
        eps = A - A2
        gap = A - Bv
        # T-transform: sum preserved, eps in (0, gap); degenerate: eps == gap
        if abs((A + Bv) - (A2 + B2)) > 1e-12:
            t1_ok = False
        others = all(abs(a[i] - b[i]) < 1e-14 for i in range(len(a)) if i not in (k - 1, k))
        if not others:
            t1_ok = False
        if m != 0:
            n_nondeg += 1
            if not (0 < eps < gap):
                t1_ok = False
        else:
            n_deg += 1
            if abs(eps - gap) > 1e-12:
                t1_ok = False
        drop = V(a) - V(b)
        pred = 2 * eps * (gap - eps)
        max_id_err = max(max_id_err, abs(drop - pred))
        if m != 0 and not H(b) > H(a) - 1e-15:
            ent_ok = False
        if min(b) < min(p0) - 1e-12:
            ent_ok = False
OUT["T1_per_swap_T_transform"] = dict(swaps=n_swaps, nondegenerate=n_nondeg,
                                      degenerate=n_deg, all_pass=t1_ok)
OUT["T2_dissipation_identity"] = dict(max_abs_error=max_id_err,
                                      all_pass=max_id_err < 1e-10)
OUT["T8_fixed_shift_entropy"] = dict(all_pass=ent_ok)

# ------------------------------------------------- T3: GSA two-sided extremality
c = 0.5 * math.log(1 / (0.99 - 0.25))
ok_low = ok_up = True
for _ in range(20000):
    d = random.randint(2, 30)
    L = random.uniform(-50, 50)
    ps = L / d + c * (d + 1 - 2 * np.arange(1, d + 1)) / 2
    gaps_hi = c + np.random.exponential(1.0, d - 1) * (np.random.rand(d - 1) < 0.7)
    p = np.concatenate([[0.0], -np.cumsum(gaps_hi)])
    p += (L - p.sum()) / d
    if not majorized(ps, p):
        ok_low = False
    gaps_lo = c * np.random.rand(d - 1)
    p = np.concatenate([[0.0], -np.cumsum(gaps_lo)])
    p += (L - p.sum()) / d
    if not majorized(p, ps):
        ok_up = False
OUT["T3_gsa_two_sided"] = dict(c_delta=c, lower_gap_set_pstar_majorized=ok_low,
                               upper_gap_decreasing_p_majorized=ok_up, samples=20000)

# ------------------------------------------------------------ T4: sublevel lemma
ok = True
for _ in range(20000):
    d = random.randint(2, 25)
    x0 = np.random.randn(d); x0 -= x0.mean()
    R0 = np.abs(x0).max()
    x = np.random.randn(d) * random.choice([0.1, 1, 10, 100])
    x -= x.mean()
    if np.abs(x).max() > (d - 1) * R0 * (1 + 1e-9):
        xs, x0s = np.sort(x)[::-1], np.sort(x0)[::-1]
        if not np.all(np.cumsum(xs)[:-1] > np.cumsum(x0s)[:-1]):
            ok = False
# tightness: x0 = (R0, -R0/(d-1), ...) shape and x = e1 extreme
OUT["T4_sublevel_lemma"] = dict(all_pass=ok, samples=20000)

# ------------------------------------------------------------ T5: counterexamples
def insert(B, k, j):
    B = [list(b) for b in B]
    v = B.pop(k)
    B.insert(j, v)
    return B

B = [[4, 0, 0], [0, 2, 0], [0, 1, 3]]
r, mu = gso(B)
B2 = insert(B, 2, 0)
r2, _ = gso(B2)
P1 = sum(Fr(x) ** 2 for x in B[2])
ex1 = dict(r=[str(x) for x in r], r_after=[str(x) for x in r2],
           mu32=str(mu[2][1]), P1=str(P1), admissible=bool(P1 < Fr(99, 100) * r[0]),
           SS_before=str(sum(r)), SS_after=str(sum(r2)),
           V_before=V(logp(r)), V_after=V(logp(r2)))
B = [[1, 0, 0], [0, 10, 0], [0, 1, 1]]
r, mu = gso(B)
r2, _ = gso(insert(B, 2, 0))
ex2 = dict(r=[str(x) for x in r], r_after=[str(x) for x in r2],
           mu32=str(mu[2][1]),
           new_majorized_by_old=majorized(logp(r2), logp(r)),
           not_permutation=sorted(map(str, r)) != sorted(map(str, r2)),
           b1_sq_before=str(r[0]), b1_sq_after=str(r2[0]))
OUT["T5_counterexamples"] = dict(admissible_but_SS_and_V_increase=ex1,
                                 majorizing_but_b1_longer=ex2)

# ------------------------------------------------- T6: calibration of alpha0
def alpha0(logr, gamma=2.0, floor=0.4):
    m = np.mean(logr); s = np.std(logr)
    cv = float("inf") if m == 0 else s / abs(m)
    return cv, max(floor, (2 / (1 + cv)) ** gamma)

cal = {}
for d in [40, 80, 120, 160]:
    q = 1009
    k = d // 2
    logr = np.array([2 * math.log(q)] * k + [0.0] * (d - k))
    cal[f"qary_d{d}"] = dict(zip(["CV0", "alpha0"], alpha0(logr)))
    # Goldstein-Mayer in fplll HNF form [I_{d-1}, h; 0, q] with log2 q = 10 d.
    # r_i = (1+S_i)/(1+S_{i-1}), S_i = sum_{j<=i} h_j^2,  r_d = q^2/(1+S_{d-1}).
    lnq = 10 * d * math.log(2)
    rng = np.random.default_rng(d)
    # h_j uniform in [0,q): work in log-space, h_j = q * U_j
    U = rng.random(d - 1)
    lnS = [None]
    logr = []
    # ln(1+S_i) ~ ln S_i since S_i is astronomically large
    cum = np.log(np.cumsum(U ** 2)) + 2 * lnq
    logr.append(cum[0])
    logr += list(np.diff(cum))
    logr.append(2 * lnq - cum[-1])
    logr = np.array(logr)
    cal[f"gm_fplll_d{d}"] = dict(zip(["CV0", "alpha0"], alpha0(logr)),
                                 sum_check=float(logr.sum() - 2 * lnq))
    # Same lattice in lower-block form [q, 0; x, I]
    logr = np.array([2 * lnq] + [0.0] * (d - 1))
    cal[f"gm_lowerblock_d{d}"] = dict(zip(["CV0", "alpha0"], alpha0(logr)))
    # Gaussian N(0,25) rounded, raw profile
    Bg = np.rint(rng.normal(0, 5, (d, d)))
    Rg = np.linalg.qr(Bg.T)[1]
    logr = 2 * np.log(np.abs(np.diag(Rg)))
    cal[f"gaussian_d{d}"] = dict(zip(["CV0", "alpha0"], alpha0(logr)))
    # scale check: B -> 2B
    cal[f"gaussian_d{d}_scaled_x2"] = dict(zip(["CV0", "alpha0"],
                                              alpha0(logr + 2 * math.log(2))))
OUT["T6_calibration"] = cal

# ---------------------------------------------------- T7: thermal expansion
p = np.random.randn(10); p2 = p.copy()
p2[3], p2[4] = p[3] - 0.3, p[4] + 0.3
DV = (p ** 2).sum() - (p2 ** 2).sum()
rows = []
for a in [1e-1, 1e-2, 1e-3]:
    Dphi = np.exp(2 * a * p).sum() - np.exp(2 * a * p2).sum()
    rows.append(dict(alpha=a, ratio=Dphi / (2 * a * a * DV)))
OUT["T7_thermal_limit"] = rows

# ------------------------------------- T9: class (ii) weighted compatibility
ok = True
for _ in range(20000):
    d = random.randint(2, 12)
    w = np.sort(np.random.rand(d) + 0.01)[::-1]
    w = w + np.linspace(1e-3, 0, d)       # strictly decreasing
    p = np.random.randn(d) * 3
    k = random.randint(1, d - 1)
    if p[k - 1] <= p[k]:
        continue
    eps = random.uniform(0, p[k - 1] - p[k])
    if eps <= 0:
        continue
    q = p.copy(); q[k - 1] -= eps; q[k] += eps
    for psi in (lambda x: 2 * x, lambda x: np.exp(1.3 * x)):
        if not (w * psi(p)).sum() > (w * psi(q)).sum():
            ok = False
OUT["T9_weighted_class"] = dict(all_pass=ok)

# --------------------------------- T10: two-position update vs explicit insertion
ok = True
maxerr = 0.0
for _ in range(200):
    d = random.randint(3, 7)
    B = [[random.randint(-9, 9) for _ in range(d)] for _ in range(d)]
    r, mu = gso(B)
    if any(x == 0 for x in r):
        continue
    for alpha in (0.4, 1.0, 2.5):
        f = lambda x: float(x) ** alpha
        F = sum(f(x) for x in r)
        for k in range(1, d):
            D = 0.0
            Pn = r[k]
            for j in range(k - 1, -1, -1):
                Pc = Pn + mu[k][j] ** 2 * r[j]
                D = D + f(r[j]) + f(Pn) - f(Pc) - f(r[j] * Pn / Pc)
                r2, _ = gso(insert(B, k, j))
                Dex = F - sum(f(x) for x in r2)
                maxerr = max(maxerr, abs(D - Dex) / max(1.0, abs(F)))
                Pn = Pc
OUT["T10_two_position_update"] = dict(max_rel_error=maxerr, all_pass=maxerr < 1e-9)


# -------------------------------- T11: integer separation; exhaustive small inputs
from decimal import Decimal, localcontext
from itertools import product


def decimal_profile(r):
    return [(Decimal(x.numerator) / Decimal(x.denominator)).ln() / 2 for x in r]


def decimal_variance(r):
    if len(set(r)) == 1:
        return Decimal(0)
    p = decimal_profile(r)
    mean = sum(p) / len(p)
    return sum((x - mean) ** 2 for x in p)


checked_profiles = checked_pairs = 0
with localcontext() as ctx:
    ctx.prec = 80
    bases = []
    for a, b, c, d in product(range(-2, 3), repeat=4):
        if a*d != b*c:
            bases.append([[a, b], [c, d]])
    exhaustive_count = len(bases)
    rng = random.Random(713)
    for _ in range(100):
        # Triangular integral bases with guaranteed nonzero determinant.
        d = rng.randrange(3, 7)
        bases.append([[rng.randrange(1, 12) if i == j else
                       rng.randrange(-7, 8) if j < i else 0
                       for j in range(d)] for i in range(d)])
    for B in bases:
        r, _ = gso(B)
        d = len(r)
        R = max(2, *(math.ceil(x) for x in r))
        D = [Fr(1)]
        for x in r:
            D.append(D[-1] * x)
        assert all(x.denominator == 1 for x in D)
        assert all(1 <= x <= R**i for i, x in enumerate(D))
        v = decimal_variance(r)
        for i in range(d):
            for j in range(i):
                a, b = D[i+1]*D[j], D[i]*D[j+1]
                assert a/b == r[i]/r[j]
                if a == b:
                    continue
                assert max(a, b) <= R**(2*d)
                assert abs(a-b)/max(a,b) >= Fr(1, R**(2*d))
                log_ratio = (Decimal(a.numerator) / Decimal(b.numerator)).ln()
                assert abs(log_ratio) >= Decimal(1) / Decimal(R**(2*d))
                checked_pairs += 1
        if v:
            assert v >= Decimal(1) / Decimal(8*R**(4*d))
        checked_profiles += 1
OUT["T11_integral_variance_separation"] = dict(
    all_pass=True, exhaustive_2x2=exhaustive_count,
    profiles=checked_profiles, unequal_pairs=checked_pairs, log_precision_digits=80)

# -------------------------------- T12: complete small variance-descent trajectories

def variance_run(B, depth_priority=False):
    B = [row[:] for row in B]
    r0, _ = gso(B)
    d = len(B)
    v0 = decimal_variance(r0)
    mean0 = sum(decimal_profile(r0)) / d
    log_R = max(Decimal(2).ln(), 2*mean0 + 2*v0.sqrt())
    log_floor = -Decimal(8).ln() - 4*d*log_R
    rho = Decimal('0.01')
    M = A = 0
    for _ in range(1000):
        size_reduce(B)
        r, mu = gso(B)
        v = decimal_variance(r)
        assert v <= v0 + Decimal('1e-65')
        if v:
            assert v.ln() >= log_floor
        best = None
        for k in range(1, d):
            for j in range(k-1, -1, -1):
                B1 = insert(B, k, j)
                rr, _ = gso(B1)
                drop = v - decimal_variance(rr)
                if drop > rho*v:
                    priority = drop / (k-j) if depth_priority else drop
                    if best is None or priority > best[0]:
                        best = priority, B1
        if best:
            B = best[1]
            M += 1
            continue
        bad = [k for k in range(1, d) if
               r[k] < (Fr(99,100)-mu[k][k-1]**2)*r[k-1]]
        if bad:
            k = bad[0]
            B[k-1], B[k] = B[k], B[k-1]
            A += 1
            continue
        break
    else:
        raise AssertionError('small validation trajectory did not terminate')
    if v0:
        primary_bound = 1+math.ceil((v0.ln()-log_floor)/(-(1-rho).ln()))
        width = Decimal(d*(d+1))*v0.sqrt()
        assert M <= primary_bound
        assert A <= (M+1)*math.ceil(width/(Decimal(100)/99).ln()*2)
    else:
        assert M == A == 0
    return M, A, decimal_variance(gso(B)[0])

with localcontext() as ctx:
    ctx.prec = 80
    # [[2,0],[1,1]] gives r=(4,1) -> (2,2), so the final-zero case is real.
    assert variance_run([[2,0],[1,1]]) == (1,0,Decimal(0))
    assert variance_run([[4,0],[0,1]])[:2] == (0,1)
    assert variance_run([[1,0],[0,1]]) == (0,0,Decimal(0))
    r, mu = gso([[2,0],[1,1]])
    # At delta=1/2, the c_delta gap is exactly at Lovasz equality.
    assert r[1] == (Fr(1,2)-mu[1][0]**2)*r[0]
    total_moves = 0
    for B in bases[::12]:
        for priority in (False, True):
            M, A, _ = variance_run(B, priority)
            total_moves += M+A
    assert alpha0(np.zeros(3))[1] == 0.4
OUT["T12_variance_trajectories_and_boundaries"] = dict(
    all_pass=True, runs=2*len(bases[::12])+3, moves=total_moves,
    zero_variance_exit=True, degenerate_fallback=True, gap_equality_is_not_violation=True)

# Fail visibly if any diagnostic failed; numerical checks complement the proofs.
for key in ("T1_per_swap_T_transform", "T2_dissipation_identity", "T4_sublevel_lemma",
            "T8_fixed_shift_entropy", "T9_weighted_class", "T10_two_position_update"):
    assert OUT[key]["all_pass"], key
assert ok_low and ok_up
assert ex1["admissible"] and Fr(ex1["SS_after"]) > Fr(ex1["SS_before"])
assert ex1["V_after"] > ex1["V_before"]
assert ex2["new_majorized_by_old"] and ex2["not_permutation"]

with open("validate_theory.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
