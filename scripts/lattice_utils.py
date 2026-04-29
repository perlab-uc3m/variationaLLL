# lattice_utils.py -- shared utilities for lattice reduction experiments
# Javier Blanco-Romero (@fj-blanco)

import numpy as np
from fpylll import IntegerMatrix, GSO, LLL

# ---------------------------------------------------------------------------
# Lattice generators
# ---------------------------------------------------------------------------


def _is_nonsingular(B):
    """Return True when B has full rank."""
    return np.linalg.matrix_rank(B) == B.shape[0]


def gen_lattice_uniform(d, rng=10, max_attempts=500):
    """Random integer lattice with entries ~ Uniform({-rng, ..., rng})."""
    for _ in range(max_attempts):
        B = np.random.randint(-rng, rng + 1, (d, d))
        if _is_nonsingular(B):
            return B
    raise ValueError(f"Failed to generate non-singular lattice at d={d}")


def gen_lattice_gaussian(d, sigma=5.0, max_attempts=500):
    """Random integer lattice with entries ~ round(Normal(0, sigma))."""
    for _ in range(max_attempts):
        B = np.round(np.random.randn(d, d) * sigma).astype(int)
        if _is_nonsingular(B):
            return B
    raise ValueError(f"Failed to generate non-singular Gaussian lattice at d={d}")


def gen_lattice_sparse(d, density=0.5, rng=10, max_attempts=500):
    """Random sparse integer lattice (diagonal forced nonzero)."""
    for _ in range(max_attempts):
        B = np.random.randint(-rng, rng + 1, (d, d))
        mask = np.random.rand(d, d) > density
        B[mask] = 0
        np.fill_diagonal(B, np.random.randint(1, rng + 1, d))
        if _is_nonsingular(B):
            return B
    raise ValueError(f"Failed to generate non-singular sparse lattice at d={d}")


def gen_lattice_qary(d, q=101):
    """Half-rank q-ary lattice: [q*I_k, 0; A, I_{d-k}]. det = q^k."""
    k = d // 2
    B = np.zeros((d, d), dtype=int)
    np.fill_diagonal(B[:k, :k], q)
    B[k:, :k] = np.random.randint(0, q, (d - k, k))
    np.fill_diagonal(B[k:, k:], 1)
    return B


GENERATORS = {
    "uniform": gen_lattice_uniform,
    "gaussian": gen_lattice_gaussian,
    "sparse": gen_lattice_sparse,
    "qary": gen_lattice_qary,
}


# ---------------------------------------------------------------------------
# fpylll wrappers
# ---------------------------------------------------------------------------


def np_to_fpylll(B_np):
    """Convert a NumPy integer matrix to an fpylll IntegerMatrix."""
    return IntegerMatrix.from_matrix(B_np.tolist())


def get_gso_profile(B_fpy):
    """Return the GSO log-norm profile: p_i = log ||b*_i||."""
    M = GSO.Mat(B_fpy)
    M.update_gso()
    d = B_fpy.nrows
    return np.array([0.5 * np.log(max(M.get_r(i, i), 1e-30)) for i in range(d)])


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_profiles(d, n, delta=0.99, gen_func=gen_lattice_uniform, **gen_kwargs):
    """
    Generate n lattice bases of dimension d, run LLL-delta, return
    list of (pre_profile, post_profile) pairs.
    """
    pairs = []
    for _ in range(n):
        try:
            B_np = gen_func(d, **gen_kwargs)
            A_pre = np_to_fpylll(B_np)
            pre = get_gso_profile(A_pre)
            A_post = np_to_fpylll(B_np)
            LLL.reduction(A_post, delta=delta)
            post = get_gso_profile(A_post)
            pairs.append((pre, post))
        except Exception:
            continue
    return pairs


# ---------------------------------------------------------------------------
# Profile normalization and interpolation
# ---------------------------------------------------------------------------


def normalize_profile(p):
    """Zero-mean, unit-variance normalization."""
    mu = np.mean(p)
    sigma = np.std(p)
    if sigma < 1e-12:
        return p - mu
    return (p - mu) / sigma


def interpolate_profile(p, n_pts=100):
    """Interpolate profile to n_pts equispaced points on [0, 1]."""
    d = len(p)
    x_orig = np.linspace(0, 1, d)
    x_new = np.linspace(0, 1, n_pts)
    return np.interp(x_new, x_orig, p)


# ---------------------------------------------------------------------------
# Invariant functionals
# ---------------------------------------------------------------------------


def profile_mass(p):
    """Sum of absolute values: Sigma |p_i|."""
    return np.sum(np.abs(p))


def profile_entropy(p):
    """Shannon entropy of |p| / sum|p|."""
    a = np.abs(p)
    s = np.sum(a)
    if s < 1e-30:
        return 0.0
    q = a / s
    q = q[q > 1e-30]
    return -np.sum(q * np.log(q))


def profile_adjacency(p):
    """Adjacent inner product: Sigma p_i * p_{i+1}."""
    return np.sum(p[:-1] * p[1:])


def relative_change(val_pre, val_post):
    """Relative change |post - pre| / |pre|, guarded against zero."""
    denom = abs(val_pre)
    if denom < 1e-30:
        return 0.0
    return abs(val_post - val_pre) / denom
