/*
 * deep_lll.h — LLL selectors: adjacent-swap and deep-insertion.
 *
 * Adjacent selectors:
 *   LLL          : fplll's sequential LLL (via LLLReduction::lll), row-move and depth-counted
 *   Greedy-Var   : maximise Δ(Σ p²)  among all Lovász violations
 *   Greedy-H     : maximise ΔH       among all Lovász violations
 *   Toda-Greedy  : maximise −ΔC      (curvature decrease)
 *   GSA-Target   : maximise residual decrease w.r.t. OLS line
 *
 * Deep-insertion selectors:
 *   Deep-Var     : maximise the centered log-energy drop ΔV
 *   SS-GG        : maximise Δ(Σ r_i), scored by the closed-form
 *                  recurrence S ← S + μ²_{k,j} r_j (r_j/D_j − 1)
 *                  during the global insertion scan (O(d²)/step).
 *   Init-Adaptive: power-law α from initial CV, maximise Δ(Σ r^α)
 *   Batch-Deep   : sequential deepest-valid insertion
 *   Deficit-Greedy: maximise deficit decrease
 *   fplll-LLL    : baseline fplll lll_reduction wrapper
 *
 * Architecture
 * ------------
 * All GSO computation, size reduction, and row operations are
 * delegated to fplll's MatGSO<Z_NR<mpz_t>, FT> and
 * LLLReduction<Z_NR<mpz_t>, FT>.  Novel work:
 *   - scoring functions (score_adj_*, cascade_r_new, score_deep_*)
 *   - greedy scan / selection loop
 * remain here.
 *
 * Precision dispatch (per algorithm call, based on largest entry):
 *   max_bits ≤ 26  →  FP_NR<double>,   GSO_DEFAULT
 *   max_bits > 26  →  FP_NR<mpfr_t>,   GSO_INT_GRAM   ← exact Gram + MPFR
 * Dispatch happens in the non-templated public wrapper; the templated
 * *_impl function holds the hot path.
 *
 * GSO invariant exploited throughout:
 *   After LLLReduction::size_reduction(kappa_min, d), the GSO is
 *   fully valid (gso_valid_cols[k] ≥ k+1) for every row k.
 *   We therefore call read_gso() immediately after, without any
 *   separate update_gso() call.
 *
 * Size reduction and why the greedy algorithms use one pass per step:
 *   "Iterative size reduction" refers to the babai loop inside fplll's
 *   LLLReduction::size_reduction: for each row k it applies
 *   b_k ← b_k − round(µ[k][j])·b_j for all j < k, recomputes GSO,
 *   and repeats until all |µ[k][j]| ≤ η.  Standard sequential LLL
 *   needs this loop because it moves the pointer backward after a
 *   swap, re-visiting partially-processed rows; the accumulated
 *   floating-point error on those rows can be large.  The greedy
 *   algorithms differ: each iteration starts from scratch (full scan
 *   of all violations on a freshly computed, fully size-reduced basis),
 *   so one call to size_reduction() per iteration is sufficient — the
 *   basis is clean before every score computation, and the single swap
 *   only invalidates the two swapped rows, which size_reduction()
 *   fixes on the next call.  For Goldstein-Mayer lattices with
 *   O(10d)-bit entries fplll's babai handles the multi-pass innerloop
 *   transparently through the GSO_INT_GRAM exact-Gram arithmetic path.
 *
 * MPFR precision is selected in each benchmark thread before the GSO
 * objects are constructed.
 */
#ifndef DEEP_LLL_H
#define DEEP_LLL_H

#include <fplll/fplll.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <vector>

using namespace fplll;

/* ================================================================== */
/*  Result types                                                       */
/* ================================================================== */

typedef struct {
    int    n_ops;            /* swap / insertion count              */
    int    n_equiv_swaps;    /* cumulative depth for deep moves     */
    int    n_fallback_swaps; /* subthreshold adjacent LLL swaps      */
    int    hit_op_limit;     /* one if the implementation cap fired */
    int    lll_reduced;      /* one if the final basis passes fplll  */
    double delta0;           /* root Hermite factor                 */
    double final_var;        /* log-norm profile variance           */
    double kappa_f;          /* final maximum relative score drop   */
    double elapsed_sec;      /* wall-clock time                     */
    double alpha_used;       /* thermal exponent actually used      */
} deep_result_t;

typedef struct {
    deep_result_t base;
    double        cv0;
} deep_alpha_result_t;

/* ================================================================== */
/*  Basis helpers                                                      */
/* ================================================================== */

static ZZ_mat<mpz_t> copy_basis(const ZZ_mat<mpz_t> &src) {
    int           d = src.get_rows(), m = src.get_cols();
    ZZ_mat<mpz_t> dst(d, m);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < m; j++) dst[i][j] = src[i][j];
    return dst;
}

/* Largest bit-size of any entry in B (used for precision dispatch). */
static int compute_max_bits(const ZZ_mat<mpz_t> &B, int d, int m) {
    int mb = 0;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < m; j++) {
            int b = (int)mpz_sizeinbase(B[i][j].get_data(), 2);
            if (b > mb) mb = b;
        }
    return mb;
}

/* ================================================================== */
/*  GSO → double arrays                                               */
/*                                                                     */
/*  read_gso<FT>: low-level copy from a MatGSO already in valid state. */
/*  gso_to_doubles: high-level one-shot GSO computation (dispatches   */
/*  to double or MPFR based on entry size, builds a temporary MatGSO, */
/*  and returns r[0..d-1] + µ[0..d*d-1]).                             */
/* ================================================================== */

template <class FT>
static void read_gso(MatGSOInterface<Z_NR<mpz_t>, FT> &M, int d, double *mu,
                     double *r) {
    FT tmp;
    memset(mu, 0, (size_t)d * d * sizeof(double));
    for (int i = 0; i < d; i++) {
        M.get_r(tmp, i, i);
        r[i] = tmp.get_d();
        for (int j = 0; j < i; j++) {
            M.get_mu(tmp, i, j);
            mu[i * d + j] = tmp.get_d();
        }
    }
}

template <class FT>
static void read_log_diagonal(MatGSOInterface<Z_NR<mpz_t>, FT> &M, int d,
                              double *log_r) {
    FT value, log_value;
    for (int i = 0; i < d; i++) {
        M.get_r(value, i, i);
        log_value.log(value);
        log_r[i] = log_value.get_d();
    }
}

template <class FT> static long double fp_to_long_double(FT &value) {
    return (long double)value.get_d();
}

static long double fp_to_long_double(FP_NR<mpfr_t> &value) {
    return mpfr_get_ld(value.get_data(), MPFR_RNDN);
}

template <class FT>
static void read_gso_wide(MatGSOInterface<Z_NR<mpz_t>, FT> &M, int d, long double *mu,
                          long double *r) {
    FT value;
    memset(mu, 0, (size_t)d * d * sizeof(long double));
    for (int i = 0; i < d; i++) {
        M.get_r(value, i, i);
        r[i] = fp_to_long_double(value);
        for (int j = 0; j < i; j++) {
            M.get_mu(value, i, j);
            mu[i * d + j] = fp_to_long_double(value);
        }
    }
}

/*
 * One-shot GSO computation: build a temporary MatGSO, compute full GSO,
 * fill mu[0..d*d-1] and r[0..d-1].  Precision-dispatches automatically.
 * m is the number of columns (usually equals d for square bases).
 */
static void gso_to_doubles(ZZ_mat<mpz_t> &B, int d, int m, double *mu, double *r) {
    int           mb = compute_max_bits(B, d, m);
    ZZ_mat<mpz_t> eu, eut;
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> M(B, eu, eut, GSO_INT_GRAM);
        M.update_gso();
        read_gso(M, d, mu, r);
    } else {
        MatGSO<Z_NR<mpz_t>, FP_NR<double>> M(B, eu, eut, GSO_DEFAULT);
        M.update_gso();
        read_gso(M, d, mu, r);
    }
}

/* ================================================================== */
/*  Metrics                                                            */
/* ================================================================== */

static double compute_delta0(const ZZ_mat<mpz_t> &B, int d, int m, const double *r) {
    /* Compute log(||b_0||²) exactly via mpz+mpfr to avoid double
     * overflow on lattices with very large entries (e.g. GM d≥80). */
    mpz_t b0sq, tmp;
    mpz_init(b0sq);
    mpz_init(tmp);
    for (int c = 0; c < m; c++) {
        B[0][c].get_mpz(tmp);
        mpz_addmul(b0sq, tmp, tmp);
    }
    mpfr_t fb;
    mpfr_init2(fb, 128);
    mpfr_set_z(fb, b0sq, MPFR_RNDN);
    mpfr_log(fb, fb, MPFR_RNDN);
    double log_b0 = 0.5 * mpfr_get_d(fb, MPFR_RNDN);
    mpfr_clear(fb);
    mpz_clear(tmp);
    mpz_clear(b0sq);

    double log_vol = 0.0;
    for (int i = 0; i < d; i++) log_vol += 0.5 * log(fmax(r[i], 1e-30));
    return exp((log_b0 - log_vol / d) / d);
}

static double compute_delta0(const ZZ_mat<mpz_t> &B, int d, int m,
                             const long double *r) {
    mpz_t b0sq, tmp;
    mpz_init(b0sq);
    mpz_init(tmp);
    for (int c = 0; c < m; c++) {
        B[0][c].get_mpz(tmp);
        mpz_addmul(b0sq, tmp, tmp);
    }
    mpfr_t fb;
    mpfr_init2(fb, 128);
    mpfr_set_z(fb, b0sq, MPFR_RNDN);
    mpfr_log(fb, fb, MPFR_RNDN);
    long double log_b0 = 0.5L * mpfr_get_ld(fb, MPFR_RNDN);
    mpfr_clear(fb);
    mpz_clear(tmp);
    mpz_clear(b0sq);

    long double log_vol = 0.0L;
    for (int i = 0; i < d; i++) log_vol += 0.5L * logl(fmaxl(r[i], LDBL_MIN));
    return (double)expl((log_b0 - log_vol / d) / d);
}

static double profile_variance(const double *r, int d) {
    double mean = 0.0;
    for (int i = 0; i < d; i++) mean += 0.5 * log(fmax(r[i], 1e-30));
    mean /= d;
    double var = 0.0;
    for (int i = 0; i < d; i++) {
        double p = 0.5 * log(fmax(r[i], 1e-30)) - mean;
        var += p * p;
    }
    return var / d;
}

static double profile_variance(const long double *r, int d) {
    long double mean = 0.0L;
    for (int i = 0; i < d; i++) mean += 0.5L * logl(fmaxl(r[i], LDBL_MIN));
    mean /= d;
    long double var = 0.0L;
    for (int i = 0; i < d; i++) {
        long double p = 0.5L * logl(fmaxl(r[i], LDBL_MIN)) - mean;
        var += p * p;
    }
    return (double)(var / d);
}

/* ================================================================== */
/*  Adjacent-swap scoring helpers                                     */
/* ================================================================== */

/*
 * Predicted GSO norms after swapping rows k-1 and k.
 * These are the EXACT new values (no approximation) for the two
 * affected positions, assuming the rest of the profile is unchanged.
 * Derivation: Yagasaki-Yasuda cascade, depth-1 case.
 */
static inline void adj_new_r(const double *r, const double *mu, int d, int k,
                             double *r_new_km1, double *r_new_k) {
    double mkk1 = mu[k * d + (k - 1)];
    *r_new_km1  = r[k] + mkk1 * mkk1 * r[k - 1];
    *r_new_k    = r[k - 1] * r[k] / fmax(*r_new_km1, 1e-30);
}

/* f(x) = -|x|·log(|x|) for Shannon-entropy decomposition */
static inline double f_entropy(double x) {
    double ax = fabs(x);
    return (ax < 1e-30) ? 0.0 : -ax * log(ax);
}

/*
 * Greedy-Var score: decrease in Σp_i² after swapping rows k-1, k.
 * p_i = 0.5·log(r_i) are the log-GSO norms ("profile" values).
 * A positive score means the swap reduces the profile variance.
 */
static double score_adj_var(const double *r, const double *mu, int d, int k) {
    double rn_km1, rn_k;
    adj_new_r(r, mu, d, k, &rn_km1, &rn_k);
    double po_km1 = 0.5 * log(fmax(r[k - 1], 1e-30));
    double po_k   = 0.5 * log(fmax(r[k], 1e-30));
    double pn_km1 = 0.5 * log(fmax(rn_km1, 1e-30));
    double pn_k   = 0.5 * log(fmax(rn_k, 1e-30));
    return (po_km1 * po_km1 + po_k * po_k) - (pn_km1 * pn_km1 + pn_k * pn_k);
}

/*
 * Greedy-H score: ΔH (entropy increase) after swapping rows k-1, k.
 * H = -Σ q_i·log(q_i), q_i = |p_i|/Σ|p_i|.
 * sum_abs = Σ_i|p_i| must be pre-computed and passed in.
 */
static double score_adj_h(const double *r, const double *mu, int d, int k,
                          double sum_abs) {
    if (sum_abs < 1e-30) return 0.0;
    double rn_km1, rn_k;
    adj_new_r(r, mu, d, k, &rn_km1, &rn_k);
    double po_km1 = 0.5 * log(fmax(r[k - 1], 1e-30));
    double po_k   = 0.5 * log(fmax(r[k], 1e-30));
    double pn_km1 = 0.5 * log(fmax(rn_km1, 1e-30));
    double pn_k   = 0.5 * log(fmax(rn_k, 1e-30));
    return (f_entropy(pn_km1) + f_entropy(pn_k) - f_entropy(po_km1) - f_entropy(po_k)) /
           sum_abs;
}

/* ================================================================== */
/*  Toda-Greedy score: decrease in total curvature                    */
/*  C = Σ_{j=1}^{d-2} (p_{j-1} - 2p_j + p_{j+1})²                  */
/* ================================================================== */

static double score_adj_toda(const double *r, const double *mu, int d, int k) {
    double rn_km1, rn_k;
    adj_new_r(r, mu, d, k, &rn_km1, &rn_k);

    double p_km1_o = 0.5 * log(fmax(r[k - 1], 1e-30));
    double p_k_o   = 0.5 * log(fmax(r[k], 1e-30));
    double p_km1_n = 0.5 * log(fmax(rn_km1, 1e-30));
    double p_k_n   = 0.5 * log(fmax(rn_k, 1e-30));

    double p_km3 = (k >= 3) ? 0.5 * log(fmax(r[k - 3], 1e-30)) : 0.0;
    double p_km2 = (k >= 2) ? 0.5 * log(fmax(r[k - 2], 1e-30)) : 0.0;
    double p_kp1 = (k + 1 < d) ? 0.5 * log(fmax(r[k + 1], 1e-30)) : 0.0;
    double p_kp2 = (k + 2 < d) ? 0.5 * log(fmax(r[k + 2], 1e-30)) : 0.0;

    double old_c = 0.0, new_c = 0.0, d_o, d_n;
    if (k >= 3) {
        d_o = p_km3 - 2.0 * p_km2 + p_km1_o;
        d_n = p_km3 - 2.0 * p_km2 + p_km1_n;
        old_c += d_o * d_o;
        new_c += d_n * d_n;
    }
    if (k >= 2 && k <= d - 1) {
        d_o = p_km2 - 2.0 * p_km1_o + p_k_o;
        d_n = p_km2 - 2.0 * p_km1_n + p_k_n;
        old_c += d_o * d_o;
        new_c += d_n * d_n;
    }
    if (k <= d - 2) {
        d_o = p_km1_o - 2.0 * p_k_o + p_kp1;
        d_n = p_km1_n - 2.0 * p_k_n + p_kp1;
        old_c += d_o * d_o;
        new_c += d_n * d_n;
    }
    if (k <= d - 3) {
        d_o = p_k_o - 2.0 * p_kp1 + p_kp2;
        d_n = p_k_n - 2.0 * p_kp1 + p_kp2;
        old_c += d_o * d_o;
        new_c += d_n * d_n;
    }
    return old_c - new_c;
}

/* ================================================================== */
/*  GSA-Target score: decrease in squared residual from OLS line      */
/* ================================================================== */

static void compute_ols_line(const double *r, int d, double *slope_out,
                             double *intercept_out) {
    double ix_bar = (d - 1) / 2.0;
    double sum_p = 0.0, sum_ip = 0.0, Sxx = 0.0;
    for (int i = 0; i < d; i++) {
        double pi = 0.5 * log(fmax(r[i], 1e-30));
        sum_p += pi;
        sum_ip += (i - ix_bar) * pi;
        Sxx += (i - ix_bar) * (i - ix_bar);
    }
    *slope_out     = (Sxx > 1e-30) ? sum_ip / Sxx : 0.0;
    *intercept_out = sum_p / d - (*slope_out) * ix_bar;
}

static double score_adj_gsa(const double *r, const double *mu, int d, int k,
                            double slope, double intercept) {
    double rn_km1, rn_k;
    adj_new_r(r, mu, d, k, &rn_km1, &rn_k);
    double p_km1_o = 0.5 * log(fmax(r[k - 1], 1e-30)),
           p_k_o   = 0.5 * log(fmax(r[k], 1e-30));
    double p_km1_n = 0.5 * log(fmax(rn_km1, 1e-30)),
           p_k_n   = 0.5 * log(fmax(rn_k, 1e-30));
    double fit_km1 = intercept + slope * (k - 1), fit_k = intercept + slope * k;
    double eo = p_km1_o - fit_km1, eo2 = p_k_o - fit_k;
    double en = p_km1_n - fit_km1, en2 = p_k_n - fit_k;
    return (eo * eo + eo2 * eo2) - (en * en + en2 * en2);
}

/* ================================================================== */
/*  Deep-insertion GSO cascade                                        */
/*                                                                     */
/*  cascade_r_new: predicted r-values after inserting b_k at          */
/*  position j (Yamaguchi-Yasuda recursion, NuTMic 2017).             */
/*  span = k-j+1 entries written to r_new[0..span-1].                 */
/* ================================================================== */

static void cascade_r_new(const double *mu, const double *r, int d, int k, int j,
                          double *r_new, double *P) {
    int span    = k - j + 1;
    P[span - 1] = r[k];
    for (int idx = span - 2; idx >= 0; idx--) {
        int    ell   = j + idx;
        double mu_kl = mu[k * d + ell];
        P[idx]       = P[idx + 1] + mu_kl * mu_kl * r[ell];
    }
    r_new[0] = P[0];
    for (int idx = 1; idx < span; idx++) {
        int ell    = j + idx;
        r_new[idx] = r[ell - 1] * P[idx] / fmax(P[idx - 1], 1e-30);
    }
}

static inline double positive_norm(double x) {
    return fmax(x, std::numeric_limits<double>::min());
}

static inline double variance_term(double r) {
    double p = 0.5 * log(positive_norm(r));
    return p * p;
}

/* Extend the score from insertion point j+1 to j.  Only the two new
 * positions differ, so the update is constant-time. */
static inline double extend_variance_drop(double drop, double rj, double P_next,
                                          double P_cur) {
    double shifted = rj * (P_next / positive_norm(P_cur));
    double old_gap = 0.5 * (log(positive_norm(rj)) - log(positive_norm(P_next)));
    double new_gap = 0.5 * (log(positive_norm(P_cur)) - log(positive_norm(shifted)));
    return drop + 0.5 * (old_gap * old_gap - new_gap * new_gap);
}

static inline double power_term(double r, double alpha) {
    return pow(positive_norm(r), alpha);
}

static inline double extend_power_drop(double drop, double rj, double P_next,
                                       double P_cur, double alpha) {
    double shifted = rj * (P_next / positive_norm(P_cur));
    return drop + power_term(rj, alpha) + power_term(P_next, alpha) -
           power_term(P_cur, alpha) - power_term(shifted, alpha);
}

static inline long double positive_norm(long double x) {
    return fmaxl(x, LDBL_MIN);
}

static inline long double variance_term(long double r) {
    long double p = 0.5L * logl(positive_norm(r));
    return p * p;
}

static inline long double extend_variance_drop(long double drop, long double rj,
                                               long double P_next, long double P_cur) {
    long double shifted = rj * (P_next / positive_norm(P_cur));
    long double old_gap =
        0.5L * (logl(positive_norm(rj)) - logl(positive_norm(P_next)));
    long double new_gap =
        0.5L * (logl(positive_norm(P_cur)) - logl(positive_norm(shifted)));
    return drop + 0.5L * (old_gap * old_gap - new_gap * new_gap);
}

static inline long double power_term(long double r, double alpha) {
    return powl(positive_norm(r), (long double)alpha);
}

static long double centered_log_energy(const long double *r, int d) {
    long double mean = 0.0L;
    for (int i = 0; i < d; i++) mean += 0.5L * logl(positive_norm(r[i]));
    mean /= d;

    long double energy = 0.0L;
    for (int i = 0; i < d; i++) {
        long double p = 0.5L * logl(positive_norm(r[i])) - mean;
        energy += p * p;
    }
    return energy;
}

static long double power_energy(const long double *r, int d, double alpha) {
    long double energy = 0.0L;
    if (alpha == 1.0) {
        for (int i = 0; i < d; i++) energy += positive_norm(r[i]);
        return energy;
    }
    for (int i = 0; i < d; i++) energy += power_term(r[i], alpha);
    return energy;
}

static double relative_score_residual(long double best_drop, long double energy) {
    if (best_drop <= 0.0L) return 0.0;
    if (energy <= 0.0L) return std::numeric_limits<double>::infinity();
    return (double)(best_drop / energy);
}

static double max_relative_variance_drop(const long double *mu, const long double *r,
                                         int d) {
    long double best = 0.0L;
    for (int k = 1; k < d; k++) {
        long double P_next = r[k];
        long double drop   = 0.0L;
        for (int j = k - 1; j >= 0; j--) {
            long double mkj = mu[k * d + j];
            long double Pc  = P_next + mkj * mkj * r[j];
            drop            = extend_variance_drop(drop, r[j], P_next, Pc);
            if (drop > best) best = drop;
            P_next = Pc;
        }
    }
    return relative_score_residual(best, centered_log_energy(r, d));
}

static double max_relative_ss_drop(const long double *mu, const long double *r, int d) {
    long double best = 0.0L;
    for (int k = 1; k < d; k++) {
        long double Pc   = r[k];
        long double drop = 0.0L;
        for (int j = k - 1; j >= 0; j--) {
            long double mkj  = mu[k * d + j];
            long double term = mkj * mkj * r[j];
            Pc += term;
            drop += term * (r[j] / positive_norm(Pc) - 1.0L);
            if (drop > best) best = drop;
        }
    }
    return relative_score_residual(best, power_energy(r, d, 1.0));
}

static inline long double extend_power_drop(long double drop, long double rj,
                                            long double P_next, long double P_cur,
                                            double alpha) {
    long double shifted = rj * (P_next / positive_norm(P_cur));
    return drop + power_term(rj, alpha) + power_term(P_next, alpha) -
           power_term(P_cur, alpha) - power_term(shifted, alpha);
}

static double max_relative_power_drop(const long double *mu, const long double *r,
                                      int d, double alpha) {
    long double best = 0.0L;
    for (int k = 1; k < d; k++) {
        long double P_next = r[k];
        long double drop   = 0.0L;
        for (int j = k - 1; j >= 0; j--) {
            long double mkj = mu[k * d + j];
            long double Pc  = P_next + mkj * mkj * r[j];
            drop            = extend_power_drop(drop, r[j], P_next, Pc, alpha);
            if (drop > best) best = drop;
            P_next = Pc;
        }
    }
    return relative_score_residual(best, power_energy(r, d, alpha));
}

/* Deep-Var: ΔV over [j..k], equal to Δ(Σp²) at fixed log-volume. */
static double score_deep_var(const double *mu, const double *r, int d, int k, int j,
                             double *wr, double *wP) {
    int span = k - j + 1;
    cascade_r_new(mu, r, d, k, j, wr, wP);
    double old_sq = 0.0, new_sq = 0.0;
    for (int idx = 0; idx < span; idx++) {
        double po = 0.5 * log(fmax(r[j + idx], 1e-30));
        double pn = 0.5 * log(fmax(wr[idx], 1e-30));
        old_sq += po * po;
        new_sq += pn * pn;
    }
    return old_sq - new_sq;
}

/* Deficit-Greedy: decrease in total Lovász proxy-deficit. */
static double score_deep_deficit(const double *mu, const double *r, int d, int k, int j,
                                 double delta, double *wr, double *wP) {
    int    span = k - j + 1;
    double thr  = delta - 0.25;
    cascade_r_new(mu, r, d, k, j, wr, wP);
    double old_def = 0.0, new_def = 0.0;
    if (j > 0) {
        old_def += fmax(0.0, thr * r[j - 1] - r[j]);
        new_def += fmax(0.0, thr * r[j - 1] - wr[0]);
    }
    for (int idx = 0; idx < span - 1; idx++) {
        old_def += fmax(0.0, thr * r[j + idx] - r[j + idx + 1]);
        new_def += fmax(0.0, thr * wr[idx] - wr[idx + 1]);
    }
    if (k + 1 < d) {
        old_def += fmax(0.0, thr * r[k] - r[k + 1]);
        new_def += fmax(0.0, thr * wr[span - 1] - r[k + 1]);
    }
    return old_def - new_def;
}

/* ================================================================== */
/*  Precision dispatch macro                                           */
/*                                                                     */
/*  DISPATCH_IMPL(fn, B, d, m, ...extra args...)                      */
/*  Computes max_bits from B, sets MPFR precision if needed, then      */
/*  calls fn##_impl<FT>(B, d, m, gso_flags, ...extra args...).        */
/*                                                                     */
/*  We don't use a macro in the end — each wrapper is 4 lines.        */
/* ================================================================== */

/* ================================================================== */
/*  Standard LLL (fplll LLLReduction, row-move and depth-counted)                   */
/* ================================================================== */

/* fplll's n_swaps counts batched row moves, not adjacent transpositions.
 * Observe the virtual GSO row operation to record its actual total depth. */
template <class ZT, class FT>
class DepthCountingGSO : public MatGSO<ZT, FT> {
  public:
    using MatGSO<ZT, FT>::MatGSO;
    int equivalent_swaps = 0;
    void move_row(int old_r, int new_r) override {
        equivalent_swaps += std::abs(old_r - new_r);
        MatGSO<ZT, FT>::move_row(old_r, new_r);
    }
};

template <class FT>
static deep_result_t run_lll_standard_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                           int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    long double *mu = (long double *)calloc((size_t)d * d, sizeof(long double));
    long double *r  = (long double *)calloc((size_t)d, sizeof(long double));

    ZZ_mat<mpz_t>                 eu, eut;
    DepthCountingGSO<Z_NR<mpz_t>, FT> M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    /* n_swaps is the row-move count; the GSO observer measures depth. */
    if (!L.lll(0, 0, d)) fprintf(stderr, "LLL warning: status %d\n", L.status);

    res.n_ops         = L.n_swaps;
    res.n_equiv_swaps = M.equivalent_swaps;

    /* Ensure full GSO validity for final metric read. */
    M.update_gso();
    read_gso_wide(M, d, mu, r);
    res.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    res.delta0      = compute_delta0(B, d, m, r);
    res.final_var   = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_lll_standard(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_lll_standard_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_lll_standard_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  Greedy-Var adjacent LLL                                           */
/*  Each iteration: full size-reduction of all rows, then pick the    */
/*  Lovász violation with maximum Δ(Σp²).                             */
/* ================================================================== */

template <class FT>
static deep_result_t run_lll_greedy_var_impl(ZZ_mat<mpz_t> &B, int d, int m,
                                             double delta, int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops = 500000;
    double *mu      = (double *)calloc((size_t)d * d, sizeof(double));
    double *r       = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    /* Initial LLL: needed only when entries are too large for double GSO
     * (e.g. Goldstein-Mayer d≥80). For moderate entries the deep selector
     * starts from the raw size-reduced basis, preserving full discriminative
     * power of the scoring rule. */
    if (gso_flags == GSO_INT_GRAM) L.lll(0, 0, d);

    int sr_from = 0; /* first violated row from last swap; 0 = full pass */
    while (res.n_ops < max_ops) {
        /* Size-reduce all rows ≥ sr_from; updates GSO incrementally.
         * After this call gso_valid_cols[k] ≥ k+1 for k ≥ sr_from.
         * Rows 0..sr_from-1 were already valid from the previous iteration. */
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        int    best_k = -1;
        double best   = -1e300;
        for (int k = 1; k < d; k++) {
            double mkk1 = mu[k * d + (k - 1)];
            double lhs  = r[k] + mkk1 * mkk1 * r[k - 1];
            if (lhs < delta * r[k - 1]) {
                double sc = score_adj_var(r, mu, d, k);
                if (sc > best) {
                    best   = sc;
                    best_k = k;
                }
            }
        }
        if (best_k < 0) break;

        /* move_row invalidates GSO from new_r onwards (row_swap does not) */
        M.move_row(best_k, best_k - 1);
        res.n_ops++;
        res.n_equiv_swaps++;
        /* Only rows best_k-1 onwards need re-size-reducing next iteration. */
        sr_from = best_k - 1;
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_lll_greedy_var(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_lll_greedy_var_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_lll_greedy_var_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  Greedy-H adjacent LLL                                             */
/*  Each iteration: full size-reduction, then pick the violation with  */
/*  maximum ΔH (entropy increase, normalised by Σ|p_i|).              */
/* ================================================================== */

template <class FT>
static deep_result_t run_lll_greedy_h_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                           int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops = 500000;
    double *mu      = (double *)calloc((size_t)d * d, sizeof(double));
    double *r       = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        double sum_abs = 0.0;
        for (int i = 0; i < d; i++) sum_abs += fabs(0.5 * log(fmax(r[i], 1e-30)));

        int    best_k = -1;
        double best   = -1e300;
        for (int k = 1; k < d; k++) {
            double mkk1 = mu[k * d + (k - 1)];
            double lhs  = r[k] + mkk1 * mkk1 * r[k - 1];
            if (lhs < delta * r[k - 1]) {
                double sc = score_adj_h(r, mu, d, k, sum_abs);
                if (sc > best) {
                    best   = sc;
                    best_k = k;
                }
            }
        }
        if (best_k < 0) break;

        M.move_row(best_k, best_k - 1);
        res.n_ops++;
        res.n_equiv_swaps++;
        sr_from = best_k - 1;
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_lll_greedy_h(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_lll_greedy_h_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_lll_greedy_h_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  Toda-Greedy adjacent LLL                                          */
/* ================================================================== */

template <class FT>
static deep_result_t run_lll_toda_greedy_impl(ZZ_mat<mpz_t> &B, int d, int m,
                                              double delta, int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops = 500000;
    double *mu      = (double *)calloc((size_t)d * d, sizeof(double));
    double *r       = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        int    best_k = -1;
        double best   = -1e300;
        for (int k = 1; k < d; k++) {
            double mkk1 = mu[k * d + (k - 1)];
            double lhs  = r[k] + mkk1 * mkk1 * r[k - 1];
            if (lhs < delta * r[k - 1]) {
                double sc = score_adj_toda(r, mu, d, k);
                if (sc > best) {
                    best   = sc;
                    best_k = k;
                }
            }
        }
        if (best_k < 0) break;

        M.move_row(best_k, best_k - 1);
        res.n_ops++;
        res.n_equiv_swaps++;
        sr_from = best_k - 1;
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_lll_toda_greedy(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_lll_toda_greedy_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_lll_toda_greedy_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  GSA-Target adjacent LLL                                           */
/* ================================================================== */

template <class FT>
static deep_result_t run_lll_gsa_target_impl(ZZ_mat<mpz_t> &B, int d, int m,
                                             double delta, int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops = 500000;
    double *mu      = (double *)calloc((size_t)d * d, sizeof(double));
    double *r       = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        double slope, intercept;
        compute_ols_line(r, d, &slope, &intercept);

        int    best_k = -1;
        double best   = -1e300;
        for (int k = 1; k < d; k++) {
            double mkk1 = mu[k * d + (k - 1)];
            double lhs  = r[k] + mkk1 * mkk1 * r[k - 1];
            if (lhs < delta * r[k - 1]) {
                double sc = score_adj_gsa(r, mu, d, k, slope, intercept);
                if (sc > best) {
                    best   = sc;
                    best_k = k;
                }
            }
        }
        if (best_k < 0) break;

        M.move_row(best_k, best_k - 1);
        res.n_ops++;
        res.n_equiv_swaps++;
        sr_from = best_k - 1;
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_lll_gsa_target(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_lll_gsa_target_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_lll_gsa_target_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  Deep-Var                                                           */
/* ================================================================== */

template <class FT>
static deep_result_t run_deep_var_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                       int gso_flags, long double drop_fraction) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int           max_ops = 500000;
    long double  *mu      = (long double *)calloc((size_t)d * d, sizeof(long double));
    long double  *r       = (long double *)calloc((size_t)d, sizeof(long double));
    ZZ_mat<mpz_t> eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso_wide(M, d, mu, r);

        int         best_k = -1, best_j = -1;
        int         fallback_k = -1;
        long double best       = 0.0L;
        long double min_drop   = drop_fraction * centered_log_energy(r, d);
        for (int k = 1; k < d; k++) {
            long double P_next = r[k];
            long double drop   = 0.0L;
            for (int j = k - 1; j >= 0; j--) {
                long double mkj = mu[k * d + j];
                long double Pc  = P_next + mkj * mkj * r[j];
                drop            = extend_variance_drop(drop, r[j], P_next, Pc);
                if (j == k - 1 && Pc < delta * r[j] && fallback_k < 0) {
                    fallback_k = k;
                }
                if (drop > min_drop && drop > best) {
                    best   = drop;
                    best_k = k;
                    best_j = j;
                }
                P_next = Pc;
            }
        }
        if (best_k < 0) {
            if (fallback_k < 0) break;
            best_k = fallback_k;
            best_j = fallback_k - 1;
            res.n_fallback_swaps++;
        }

        /* move_row(old, new): move row best_k to position best_j.    */
        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }
    if (res.n_ops >= max_ops) res.hit_op_limit = 1;

    L.size_reduction(0, d);
    read_gso_wide(M, d, mu, r);
    res.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    res.delta0      = compute_delta0(B, d, m, r);
    res.final_var   = profile_variance(r, d);
    res.kappa_f     = max_relative_variance_drop(mu, r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_deep_var(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                  long double drop_fraction) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_var_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM,
                                                drop_fraction);
    }
    return run_deep_var_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, drop_fraction);
}

/* ================================================================== */
/*  SS-GG                                                              */
/*                                                                    */
/*  ΔSS is evaluated by the closed-form recurrence                    */
/*      S ← S + μ²_{k,j} r_j (r_j / D_j − 1),                         */
/*  with D_k = r_k, D_j = D_{j+1} + μ²_{k,j} r_j, accumulated in the  */
/*  descending-j candidate scan (Pc ≡ D_j). The score is exact and   */
/*  costs O(1) per candidate. Reference: Yasuda & Yamaguchi, Eq. 5,   */
/*  as implemented in Bhattacherjee et al. 2025.                     */
/* ================================================================== */

template <class FT>
static deep_result_t run_deep_ssgg_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                        int gso_flags, long double drop_fraction) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int          max_ops = 500000;
    long double *mu      = (long double *)calloc((size_t)d * d, sizeof(long double));
    long double *r       = (long double *)calloc((size_t)d, sizeof(long double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso_wide(M, d, mu, r);

        int         best_k = -1, best_j = -1;
        int         fallback_k = -1;
        long double best       = 0.0L;
        long double min_drop   = drop_fraction * power_energy(r, d, 1.0);
        for (int k = 1; k < d; k++) {
            long double Pc = r[k];
            long double S  = 0.0L;
            for (int j = k - 1; j >= 0; j--) {
                long double mkj  = mu[k * d + j];
                long double term = mkj * mkj * r[j];
                Pc += term;
                /* Pc now equals D_j; add the j-th ΔSS contribution. */
                S += term * (r[j] / positive_norm(Pc) - 1.0L);
                if (j == k - 1 && Pc < delta * r[j] && fallback_k < 0) {
                    fallback_k = k;
                }
                if (S > min_drop && S > best) {
                    best   = S;
                    best_k = k;
                    best_j = j;
                }
            }
        }
        if (best_k < 0) {
            if (fallback_k < 0) break;
            best_k = fallback_k;
            best_j = fallback_k - 1;
            res.n_fallback_swaps++;
        }

        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }
    if (res.n_ops >= max_ops) res.hit_op_limit = 1;

    L.size_reduction(0, d);
    read_gso_wide(M, d, mu, r);
    res.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    res.delta0      = compute_delta0(B, d, m, r);
    res.final_var   = profile_variance(r, d);
    res.kappa_f     = max_relative_ss_drop(mu, r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_deep_ssgg(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                   long double drop_fraction) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_ssgg_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM,
                                                 drop_fraction);
    }
    return run_deep_ssgg_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT,
                                             drop_fraction);
}

/* ================================================================== */
/*  Schur-K: non-separable symmetric Schur-convex selector            */
/*                                                                    */
/*  Score: phi(p) = sum of the top K entries of p sorted descending,  */
/*  with p_i = 1/2 log r_i and K = ceil(d/2). Symmetric, Schur-convex */
/*  (partial sum of sorted tuple), but non-separable: sits outside    */
/*  both branches of Prop. "canonical" in the paper. Goal is to probe */
/*  whether a globally profile-aware score beats the thermal family.  */
/*                                                                    */
/*  Per-candidate cost is O(d log d): reconstruct r' via the cascade  */
/*  on the changed window [j..k], sort, sum top-K. Total O(d^3 log d) */
/*  per iteration; enough for a prototype at d <= 80.                 */
/* ================================================================== */

template <class FT>
static deep_result_t run_deep_schurK_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                          int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops = 500000;
    int     K       = (d + 1) / 2;
    double *mu      = (double *)calloc((size_t)d * d, sizeof(double));
    double *r       = (double *)calloc((size_t)d, sizeof(double));
    double *P       = (double *)calloc((size_t)d, sizeof(double));
    double *rp      = (double *)calloc((size_t)d, sizeof(double));
    double *p_cur   = (double *)calloc((size_t)d, sizeof(double));
    double *p_new   = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    if (gso_flags == GSO_INT_GRAM) L.lll(0, 0, d);

    auto topK_sum_desc = [&](double *vec) {
        std::sort(vec, vec + d, std::greater<double>());
        double s = 0.0;
        for (int i = 0; i < K; i++) s += vec[i];
        return s;
    };

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        for (int i = 0; i < d; i++) p_cur[i] = 0.5 * log(fmax(r[i], 1e-30));
        memcpy(p_new, p_cur, sizeof(double) * (size_t)d);
        double phi_cur = topK_sum_desc(p_new);

        int    best_k = -1, best_j = -1;
        double best = 0.0;
        for (int k = 1; k < d; k++) {
            double Pc = r[k];
            P[k]      = Pc;
            for (int j = k - 1; j >= 0; j--) {
                double mkj = mu[k * d + j];
                Pc += mkj * mkj * r[j];
                P[j] = Pc;
                if (Pc < delta * r[j]) {
                    /* Cascade r' on window [j..k]. */
                    rp[j] = P[j];
                    for (int l = j + 1; l <= k; l++) {
                        rp[l] = r[l - 1] * P[l] / fmax(P[l - 1], 1e-30);
                    }
                    /* Build p_new: copy p_cur, overwrite window. */
                    memcpy(p_new, p_cur, sizeof(double) * (size_t)d);
                    for (int l = j; l <= k; l++) {
                        p_new[l] = 0.5 * log(fmax(rp[l], 1e-30));
                    }
                    double phi_new = topK_sum_desc(p_new);
                    double dphi    = phi_cur - phi_new; /* maximize descent */
                    if (dphi > best) {
                        best   = dphi;
                        best_k = k;
                        best_j = j;
                    }
                }
            }
        }
        if (best_k < 0) break;

        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }

    if (res.n_ops >= max_ops) res.hit_op_limit = 1;
    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    res.delta0      = compute_delta0(B, d, m, r);
    res.final_var   = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    free(P);
    free(rp);
    free(p_cur);
    free(p_new);
    return res;
}

static deep_result_t run_deep_schurK(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_schurK_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_deep_schurK_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  Init-Adaptive                                                      */
/*  alpha = (2/(1+CV_0))^gamma, fixed from the initial profile.      */
/* ================================================================== */

typedef struct {
    int         k;
    int         j;
    long double drop;
    long double potential_drop;
} thermal_candidate_t;

typedef enum {
    THERMAL_EXACT,
    THERMAL_SHALLOW_BAND,
    THERMAL_POTENTIAL_BAND
} thermal_tie_mode_t;

/* If positive, Thermal runs use this exponent instead of the calibration. */
static double g_thermal_force_alpha = -1.0;
/* If nonzero, Thermal uses precomputed powers: one power per candidate. */
static int g_thermal_fastpow = 0;

template <class FT>
static deep_result_t run_deep_inita_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                         int gso_flags, double gamma_p,
                                         long double        drop_fraction,
                                         thermal_tie_mode_t tie_mode) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int          max_ops     = 500000;
    double       alpha_floor = 0.4;
    double       alpha       = 1.0;
    long double *mu = (long double *)calloc((size_t)d * d, sizeof(long double));
    long double *r  = (long double *)calloc((size_t)d, sizeof(long double));
    std::vector<long double> ra((size_t)d);

    ZZ_mat<mpz_t>                    eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>          M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT>    L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);
    std::vector<thermal_candidate_t> candidates;
    if (tie_mode != THERMAL_EXACT) candidates.reserve((size_t)d * (d - 1) / 2);

    /* Compute alpha from the raw GSO profile.  Use the selected GSO
       precision here as well, because a double GSO can overflow on a
       raw Goldstein-Mayer basis. */
    {
        std::vector<double> log_r_raw(d);
        M.update_gso();
        read_log_diagonal(M, d, log_r_raw.data());
        double mean_logr_raw = 0.0;
        for (int i = 0; i < d; i++) mean_logr_raw += log_r_raw[i];
        mean_logr_raw /= d;
        double var_logr_raw = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = log_r_raw[i] - mean_logr_raw;
            var_logr_raw += diff * diff;
        }
        var_logr_raw /= d;
        double cv0 = mean_logr_raw == 0.0
                         ? std::numeric_limits<double>::infinity()
                         : sqrt(var_logr_raw) / fabs(mean_logr_raw);
        alpha      = fmax(alpha_floor, pow(2.0 / (1.0 + cv0), gamma_p));
        if (g_thermal_force_alpha > 0.0) alpha = g_thermal_force_alpha;
    }
    res.alpha_used = alpha;

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso_wide(M, d, mu, r);

        int         best_k = -1, best_j = -1;
        int         fallback_k    = -1;
        long double best          = 0.0L;
        long double fallback_best = 0.0L;
        long double min_drop      = drop_fraction * power_energy(r, d, alpha);
        candidates.clear();
        if (g_thermal_fastpow)
            for (int i = 0; i < d; i++) ra[i] = power_term(r[i], alpha);
        for (int k = 1; k < d; k++) {
            long double P_next         = r[k];
            long double P_next_a       = g_thermal_fastpow ? ra[k] : 0.0L;
            long double drop           = 0.0L;
            long double potential_drop = 0.0L;
            for (int j = k - 1; j >= 0; j--) {
                long double mkj = mu[k * d + j];
                long double Pc  = P_next + mkj * mkj * r[j];
                if (g_thermal_fastpow) {
                    long double Pc_a = power_term(Pc, alpha);
                    drop += ra[j] + P_next_a - Pc_a - ra[j] * (P_next_a / Pc_a);
                    P_next_a = Pc_a;
                } else
                    drop = extend_power_drop(drop, r[j], P_next, Pc, alpha);
                potential_drop += 0.5L * logl(positive_norm(r[j]) / positive_norm(Pc));
                if (j == k - 1 && Pc < delta * r[j]) {
                    long double fallback_priority =
                        tie_mode == THERMAL_POTENTIAL_BAND ? potential_drop : drop;
                    if (fallback_k < 0 || (tie_mode != THERMAL_EXACT &&
                                           fallback_priority > fallback_best)) {
                        fallback_k    = k;
                        fallback_best = fallback_priority;
                    }
                }
                if (drop > min_drop) {
                    if (tie_mode != THERMAL_EXACT)
                        candidates.push_back({k, j, drop, potential_drop});
                    if (drop > best) {
                        best   = drop;
                        best_k = k;
                        best_j = j;
                    }
                }
                P_next = Pc;
            }
        }
        if (tie_mode != THERMAL_EXACT && best_k >= 0) {
            const long double band_floor = best - min_drop;
            int               band_k = -1, band_j = -1;
            int               band_depth = d + 1;
            long double band_potential = -std::numeric_limits<long double>::infinity();
            for (const thermal_candidate_t &candidate : candidates) {
                int  depth  = candidate.k - candidate.j;
                bool better = tie_mode == THERMAL_SHALLOW_BAND
                                  ? depth < band_depth
                                  : candidate.potential_drop > band_potential;
                if (candidate.drop >= band_floor && better) {
                    band_k         = candidate.k;
                    band_j         = candidate.j;
                    band_depth     = depth;
                    band_potential = candidate.potential_drop;
                }
            }
            best_k = band_k;
            best_j = band_j;
        }
        if (best_k < 0) {
            if (fallback_k < 0) break;
            best_k = fallback_k;
            best_j = fallback_k - 1;
            res.n_fallback_swaps++;
        }

        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }
    if (res.n_ops >= max_ops) res.hit_op_limit = 1;

    L.size_reduction(0, d);
    read_gso_wide(M, d, mu, r);
    res.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    res.delta0      = compute_delta0(B, d, m, r);
    res.final_var   = profile_variance(r, d);
    res.kappa_f     = max_relative_power_drop(mu, r, d, alpha);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_deep_inita(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                    double gamma_p, long double drop_fraction) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_inita_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM, gamma_p,
                                                  drop_fraction, THERMAL_EXACT);
    }
    return run_deep_inita_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, gamma_p,
                                              drop_fraction, THERMAL_EXACT);
}

static deep_result_t run_deep_band(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                   double gamma_p, long double drop_fraction) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_inita_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM, gamma_p,
                                                  drop_fraction, THERMAL_SHALLOW_BAND);
    }
    return run_deep_inita_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, gamma_p,
                                              drop_fraction, THERMAL_SHALLOW_BAND);
}

static deep_result_t run_deep_two_clock(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                        double gamma_p, long double drop_fraction) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_inita_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM, gamma_p,
                                                  drop_fraction,
                                                  THERMAL_POTENTIAL_BAND);
    }
    return run_deep_inita_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, gamma_p,
                                              drop_fraction, THERMAL_POTENTIAL_BAND);
}

/* ================================================================== */
/*  Thermal-Sched                                                      */
/*  Same α = A·(1+CV)^{-γ} formula, but α is re-estimated every        */
/*  `period` accepted insertions from the *current* GSO profile.       */
/*  Rationale: the profile flattens during reduction, so a schedule   */
/*  stays inside the Schur-convex branch and lets α track the shape.  */
/* ================================================================== */

template <class FT>
static deep_result_t run_deep_sched_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                         int gso_flags, double A, double gamma_p,
                                         int period) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops     = 500000;
    double  alpha_floor = 0.4;
    double  alpha       = 1.0;
    double *mu          = (double *)calloc((size_t)d * d, sizeof(double));
    double *r           = (double *)calloc((size_t)d, sizeof(double));
    double *wr          = (double *)calloc((size_t)d, sizeof(double));
    double *wP          = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    if (period <= 0) period = (d / 4 > 0 ? d / 4 : 1);

    /* Initial calibration from the raw profile (matches Thermal-Adaptive). */
    auto calibrate_from_r = [&](const double *rbuf) {
        double mean_lr = 0.0;
        for (int i = 0; i < d; i++) mean_lr += log(fmax(rbuf[i], 1e-300));
        mean_lr /= d;
        double var_lr = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = log(fmax(rbuf[i], 1e-300)) - mean_lr;
            var_lr += diff * diff;
        }
        var_lr /= d;
        double cv = sqrt(var_lr) / fmax(fabs(mean_lr), 1e-10);
        return fmax(alpha_floor, A * pow(1.0 + cv, -gamma_p));
    };

    {
        ZZ_mat<mpz_t>                      eu_cal, eut_cal;
        MatGSO<Z_NR<mpz_t>, FP_NR<double>> Mcal(B, eu_cal, eut_cal, GSO_DEFAULT);
        Mcal.update_gso();
        std::vector<double> r_raw(d);
        for (int i = 0; i < d; i++) {
            FP_NR<double> tmp;
            Mcal.get_r(tmp, i, i);
            r_raw[i] = fmax(tmp.get_d(), 1e-300);
        }
        alpha = calibrate_from_r(r_raw.data());
    }

    if (gso_flags == GSO_INT_GRAM) L.lll(0, 0, d);

    bool alpha_is_one      = (fabs(alpha - 1.0) < 1e-12);
    int  steps_since_reest = 0;
    int  sr_from           = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        int    best_k = -1, best_j = -1;
        double best = 0.0;
        for (int k = 1; k < d; k++) {
            double Pc = r[k];
            for (int j = k - 1; j >= 0; j--) {
                double mkj = mu[k * d + j];
                Pc += mkj * mkj * r[j];
                if (Pc < delta * r[j]) {
                    int span = k - j + 1;
                    cascade_r_new(mu, r, d, k, j, wr, wP);
                    double old_s = 0.0, new_s = 0.0;
                    if (alpha_is_one) {
                        for (int idx = 0; idx < span; idx++) {
                            old_s += r[j + idx];
                            new_s += wr[idx];
                        }
                    } else {
                        for (int idx = 0; idx < span; idx++) {
                            old_s += pow(fmax(r[j + idx], 1e-30), alpha);
                            new_s += pow(fmax(wr[idx], 1e-30), alpha);
                        }
                    }
                    double sc = old_s - new_s;
                    if (sc > best) {
                        best   = sc;
                        best_k = k;
                        best_j = j;
                    }
                }
            }
        }
        if (best_k < 0) break;

        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;

        if (++steps_since_reest >= period) {
            steps_since_reest = 0;
            /* Use the current r buffer (already post-insertion view is stale,
               refresh cheaply before recalibrating). */
            L.size_reduction(sr_from, d);
            read_gso(M, d, mu, r);
            alpha        = calibrate_from_r(r);
            alpha_is_one = (fabs(alpha - 1.0) < 1e-12);
        }
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    free(wr);
    free(wP);
    return res;
}

static deep_result_t run_deep_sched(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                    double A, double gamma_p, int period) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_sched_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM, A,
                                                  gamma_p, period);
    }
    return run_deep_sched_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, A, gamma_p,
                                              period);
}

/* ================================================================== */
/*  Hybrid selector: adaptive α (same as Thermal-Adaptive) × position  */
/*  weights (d - i + 1)^β from the weighted-linear branch of           */
/*  Proposition prop:canonical. β = 0 reduces exactly to Thermal.      */
/* ================================================================== */

template <class FT>
static deep_result_t run_deep_hybrid_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                          int gso_flags, double A, double gamma_p,
                                          double beta) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops     = 500000;
    double  alpha_floor = 0.4;
    double  alpha       = 1.0;
    double *mu          = (double *)calloc((size_t)d * d, sizeof(double));
    double *r           = (double *)calloc((size_t)d, sizeof(double));
    double *wr          = (double *)calloc((size_t)d, sizeof(double));
    double *wP          = (double *)calloc((size_t)d, sizeof(double));
    double *wpos        = (double *)calloc((size_t)d, sizeof(double));

    for (int i = 0; i < d; i++) wpos[i] = pow((double)(d - i), beta);

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    {
        ZZ_mat<mpz_t>                      eu_cal, eut_cal;
        MatGSO<Z_NR<mpz_t>, FP_NR<double>> Mcal(B, eu_cal, eut_cal, GSO_DEFAULT);
        Mcal.update_gso();
        std::vector<double> log_r_raw(d);
        for (int i = 0; i < d; i++) {
            FP_NR<double> tmp;
            Mcal.get_r(tmp, i, i);
            log_r_raw[i] = log(fmax(tmp.get_d(), 1e-300));
        }
        double mean_logr_raw = 0.0;
        for (int i = 0; i < d; i++) mean_logr_raw += log_r_raw[i];
        mean_logr_raw /= d;
        double var_logr_raw = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = log_r_raw[i] - mean_logr_raw;
            var_logr_raw += diff * diff;
        }
        var_logr_raw /= d;
        double cv0 = mean_logr_raw == 0.0
                         ? std::numeric_limits<double>::infinity()
                         : sqrt(var_logr_raw) / fabs(mean_logr_raw);
        alpha      = fmax(alpha_floor, A * pow(1.0 + cv0, -gamma_p));
    }

    if (gso_flags == GSO_INT_GRAM) L.lll(0, 0, d);

    const bool alpha_is_one = (fabs(alpha - 1.0) < 1e-12);

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        int    best_k = -1, best_j = -1;
        double best = 0.0;
        for (int k = 1; k < d; k++) {
            double Pc = r[k];
            for (int j = k - 1; j >= 0; j--) {
                double mkj = mu[k * d + j];
                Pc += mkj * mkj * r[j];
                if (Pc < delta * r[j]) {
                    int span = k - j + 1;
                    cascade_r_new(mu, r, d, k, j, wr, wP);
                    double old_s = 0.0, new_s = 0.0;
                    if (alpha_is_one) {
                        for (int idx = 0; idx < span; idx++) {
                            double w = wpos[j + idx];
                            old_s += w * r[j + idx];
                            new_s += w * wr[idx];
                        }
                    } else {
                        for (int idx = 0; idx < span; idx++) {
                            double w = wpos[j + idx];
                            old_s += w * pow(fmax(r[j + idx], 1e-30), alpha);
                            new_s += w * pow(fmax(wr[idx], 1e-30), alpha);
                        }
                    }
                    double sc = old_s - new_s;
                    if (sc > best) {
                        best   = sc;
                        best_k = k;
                        best_j = j;
                    }
                }
            }
        }
        if (best_k < 0) break;

        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    free(wr);
    free(wP);
    free(wpos);
    return res;
}

static deep_result_t run_deep_hybrid(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                     double A, double gamma_p, double beta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_hybrid_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM, A,
                                                   gamma_p, beta);
    }
    return run_deep_hybrid_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, A, gamma_p,
                                               beta);
}

/* ================================================================== */
/*  Fixed-alpha deep insertion                                         */
/* ================================================================== */

template <class FT>
static deep_alpha_result_t run_deep_fixed_alpha_impl(ZZ_mat<mpz_t> &B, int d, int m,
                                                     double delta, int gso_flags,
                                                     double alpha) {
    deep_alpha_result_t out;
    memset(&out, 0, sizeof(out));
    deep_result_t  &res = out.base;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int          max_ops = 500000;
    long double *mu      = (long double *)calloc((size_t)d * d, sizeof(long double));
    long double *r       = (long double *)calloc((size_t)d, sizeof(long double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    /* Compute CV₀ for reporting. */
    L.size_reduction(0, d);
    read_gso_wide(M, d, mu, r);
    long double mean_logr = 0.0L;
    for (int i = 0; i < d; i++) mean_logr += logl(positive_norm(r[i]));
    mean_logr /= d;
    long double var_logr = 0.0L;
    for (int i = 0; i < d; i++) {
        long double diff = logl(positive_norm(r[i])) - mean_logr;
        var_logr += diff * diff;
    }
    var_logr /= d;
    out.cv0 = (double)(sqrtl(var_logr) / fmaxl(fabsl(mean_logr), 1e-10L));

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso_wide(M, d, mu, r);

        int         best_k = -1, best_j = -1;
        int         fallback_k = -1;
        long double best       = 0.0L;
        for (int k = 1; k < d; k++) {
            long double P_next = r[k];
            long double drop   = 0.0L;
            for (int j = k - 1; j >= 0; j--) {
                long double mkj = mu[k * d + j];
                long double Pc  = P_next + mkj * mkj * r[j];
                drop            = extend_power_drop(drop, r[j], P_next, Pc, alpha);
                if (Pc < delta * r[j]) {
                    if (j == k - 1 && fallback_k < 0) fallback_k = k;
                    if (drop > best) {
                        best   = drop;
                        best_k = k;
                        best_j = j;
                    }
                }
                P_next = Pc;
            }
        }
        if (best_k < 0) {
            if (fallback_k < 0) break;
            best_k = fallback_k;
            best_j = fallback_k - 1;
            res.n_fallback_swaps++;
        }

        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }
    if (res.n_ops >= max_ops) res.hit_op_limit = 1;

    L.size_reduction(0, d);
    read_gso_wide(M, d, mu, r);
    res.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    res.delta0      = compute_delta0(B, d, m, r);
    res.final_var   = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return out;
}

static deep_alpha_result_t run_deep_fixed_alpha(ZZ_mat<mpz_t> &B, int d, int m,
                                                double delta, double alpha) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_fixed_alpha_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM,
                                                        alpha);
    }
    return run_deep_fixed_alpha_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, alpha);
}

/* ================================================================== */
/*  Batch deep insertion (sequential, deepest-valid)                  */
/* ================================================================== */

template <class FT>
static deep_result_t run_batch_deep_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                         int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops    = 500000;
    double  wall_limit = 30.0;
    int     iter_count = 0;
    double *mu         = (double *)calloc((size_t)d * d, sizeof(double));
    double *r          = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    /* Establish initial GSO. */
    L.size_reduction(0, d);
    read_gso(M, d, mu, r);

    int k = 1;
    while (k < d && res.n_ops < max_ops) {
        if (++iter_count % 50 == 0) {
            struct timespec tnow;
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed =
                (tnow.tv_sec - t0.tv_sec) + (tnow.tv_nsec - t0.tv_nsec) * 1e-9;
            if (elapsed > wall_limit) break;
        }

        /* Size-reduce row k (single row, cheap). */
        L.size_reduction(k, k + 1);
        read_gso(M, d, mu, r); /* refresh mu[k][*] */

        /* Find deepest valid j for row k. */
        int    best_j = -1;
        double Pc     = r[k];
        for (int j = k - 1; j >= 0; j--) {
            double mkj = mu[k * d + j];
            Pc += mkj * mkj * r[j];
            if (Pc < delta * r[j]) best_j = j;
        }

        if (best_j >= 0) {
            M.move_row(k, best_j);
            /* Re-reduce rows best_j..k from scratch. */
            L.size_reduction(best_j, d);
            read_gso(M, d, mu, r);
            res.n_ops++;
            res.n_equiv_swaps += k - best_j;
            k = (best_j > 1) ? best_j : 1;
        } else {
            k++;
        }
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    return res;
}

static deep_result_t run_batch_deep(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_batch_deep_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_batch_deep_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  Deficit-Greedy deep insertion                                      */
/* ================================================================== */

template <class FT>
static deep_result_t run_deep_deficit_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                           int gso_flags) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int     max_ops = 500000;
    double *mu      = (double *)calloc((size_t)d * d, sizeof(double));
    double *r       = (double *)calloc((size_t)d, sizeof(double));
    double *wr      = (double *)calloc((size_t)d, sizeof(double));
    double *wP      = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    while (res.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        int    best_k = -1, best_j = -1;
        double best = 0.0;
        for (int k = 1; k < d; k++) {
            double Pc = r[k];
            for (int j = k - 1; j >= 0; j--) {
                double mkj = mu[k * d + j];
                Pc += mkj * mkj * r[j];
                if (Pc < delta * r[j]) {
                    double sc = score_deep_deficit(mu, r, d, k, j, delta, wr, wP);
                    if (sc > best) {
                        best   = sc;
                        best_k = k;
                        best_j = j;
                    }
                }
            }
        }
        if (best_k < 0) break;

        M.move_row(best_k, best_j);
        res.n_ops++;
        res.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }

    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(mu);
    free(r);
    free(wr);
    free(wP);
    return res;
}

static deep_result_t run_deep_deficit(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_deep_deficit_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM);
    }
    return run_deep_deficit_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT);
}

/* ================================================================== */
/*  fplll-LLL baseline (lll_reduction wrapper, no swap count)         */
/* ================================================================== */

static deep_result_t run_fplll_lll(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    deep_result_t res;
    memset(&res, 0, sizeof(res));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int status =
        lll_reduction(B, delta, LLL_DEF_ETA, LM_WRAPPER, FT_DEFAULT, 0, LLL_DEFAULT);
    if (status != RED_SUCCESS)
        fprintf(stderr, "fplll LLL warning: status %d\n", status);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    res.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    /* Compute final GSO metrics using the appropriate precision. */
    int           mb = compute_max_bits(B, d, m);
    double       *mu = (double *)calloc((size_t)d * d, sizeof(double));
    double       *r  = (double *)calloc((size_t)d, sizeof(double));
    ZZ_mat<mpz_t> eu, eut;
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        MatGSO<Z_NR<mpz_t>, FP_NR<mpfr_t>> M(B, eu, eut, GSO_INT_GRAM);
        M.update_gso();
        read_gso(M, d, mu, r);
    } else {
        MatGSO<Z_NR<mpz_t>, FP_NR<double>> M(B, eu, eut, GSO_DEFAULT);
        M.update_gso();
        read_gso(M, d, mu, r);
    }
    res.delta0    = compute_delta0(B, d, m, r);
    res.final_var = profile_variance(r, d);
    free(mu);
    free(r);
    return res;
}

/* ================================================================== */
/*  BKZ-VarSkip                                                        */
/*                                                                     */
/*  Variance-guided BKZ block selection (Corollary cor:bkz_skip).     */
/*  At the start of each BKZ tour, compute the local sum-of-squares   */
/*    V_k = Σ_{i=k}^{k+β-1} p_i²                                     */
/*  for each block k ∈ {0,...,d-β}.  Skip blocks whose V_k falls      */
/*  below the skip_frac-th percentile of {V_k}.  By Cor. cor:bkz_skip,*/
/*  the one-tour variance overhead is bounded above by V_threshold per */
/*  skipped block.                                                     */
/*                                                                     */
/*  skip_frac ∈ [0,1): fraction of coldest blocks to skip per tour.   */
/*  block_size β: BKZ block size.                                      */
/*  max_tours: hard cap on BKZ tours (default 100).                   */
/*                                                                     */
/*  Implementation: uses BKZReduction<ZT,FT>::svp_reduction directly  */
/*  for block-level control.  Initial LLL is always run first.        */
/* ================================================================== */

/* Compute local block variance V_k = Σ_{i=k}^{k+β-1} p_i²
 * (uncentred sum-of-squares, consistent with the paper's Σ p_i²). */
static double block_var(const double *r, int k, int beta) {
    double v = 0.0;
    for (int i = k; i < k + beta; i++) {
        double p = 0.5 * log(fmax(r[i], 1e-30));
        v += p * p;
    }
    return v;
}

/* In-place nth_element-style percentile on a copy of src[n].
 * Returns the value at the floor(frac*(n-1)) position. */
static double percentile(const double *src, int n, double frac) {
    double *tmp = (double *)malloc((size_t)n * sizeof(double));
    memcpy(tmp, src, (size_t)n * sizeof(double));
    int idx = (int)(frac * (n - 1));
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    /* partial sort: find idx-th smallest using selection */
    for (int i = 0; i <= idx; i++) {
        int min_j = i;
        for (int j = i + 1; j < n; j++)
            if (tmp[j] < tmp[min_j]) min_j = j;
        double t   = tmp[i];
        tmp[i]     = tmp[min_j];
        tmp[min_j] = t;
    }
    double val = tmp[idx];
    free(tmp);
    return val;
}

typedef struct {
    deep_result_t base;
    int           n_skipped; /* total blocks skipped across all tours */
    int           n_tours;   /* number of BKZ tours completed         */
} bkz_result_t;

template <class FT>
static bkz_result_t run_bkz_var_skip_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                          int gso_flags, int block_size,
                                          double skip_frac, int max_tours) {
    bkz_result_t bres;
    memset(&bres, 0, sizeof(bres));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    double *mu = (double *)calloc((size_t)d * d, sizeof(double));
    double *r  = (double *)calloc((size_t)d, sizeof(double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    /* Initial LLL */
    if (!L.lll(0, 0, d))
        fprintf(stderr, "BKZ-VarSkip: initial LLL warning: status %d\n", L.status);

    /* Build BKZ param with empty strategies (uses fplll's internal defaults
     * for enumeration bounding; adequate for block_size ≤ 30). */
    vector<Strategy> strats;
    for (int b = 0; b <= block_size; b++)
        strats.emplace_back(Strategy::EmptyStrategy(b));
    BKZParam param(block_size, strats, delta, BKZ_NO_LLL, 1 /* max_loops per call */);
    BKZReduction<Z_NR<mpz_t>, FT> bkz(M, L, param);

    int     n_blocks = d - block_size + 1;
    double *bvars    = (double *)calloc((size_t)n_blocks, sizeof(double));

    for (int tour = 0; tour < max_tours; tour++) {
        /* Read current profile and compute per-block V_k. */
        M.update_gso();
        read_gso(M, d, mu, r);
        for (int k = 0; k < n_blocks; k++) bvars[k] = block_var(r, k, block_size);

        /* Adaptive threshold: skip_frac percentile of block variances. */
        double threshold =
            (skip_frac > 0.0) ? percentile(bvars, n_blocks, skip_frac) : -1.0;

        bool made_progress = false;
        for (int k = 0; k < n_blocks; k++) {
            if (bvars[k] <= threshold) {
                bres.n_skipped++;
                continue;
            }
            /* svp_reduction returns false if it modified the basis. */
            bool clean = bkz.svp_reduction(k, block_size, param);
            bres.base.n_ops++;
            if (!clean) made_progress = true;
        }

        bres.n_tours++;
        if (!made_progress) break;
    }

    /* Final LLL clean-up and metrics. */
    L.lll(0, 0, d);
    M.update_gso();
    read_gso(M, d, mu, r);
    bres.base.delta0        = compute_delta0(B, d, m, r);
    bres.base.final_var     = profile_variance(r, d);
    bres.base.n_equiv_swaps = bres.base.n_ops; /* 1:1 for BKZ oracle calls */

    clock_gettime(CLOCK_MONOTONIC, &t1);
    bres.base.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    free(mu);
    free(r);
    free(bvars);
    return bres;
}

/* Public wrapper: precision-dispatch. */
static bkz_result_t run_bkz_var_skip(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                     int block_size, double skip_frac, int max_tours) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_bkz_var_skip_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM,
                                                    block_size, skip_frac, max_tours);
    }
    return run_bkz_var_skip_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, block_size,
                                                skip_frac, max_tours);
}

/* Convenience: standard BKZ (skip_frac=0) for baseline comparison. */
static bkz_result_t run_bkz_standard(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                     int block_size, int max_tours) {
    return run_bkz_var_skip(B, d, m, delta, block_size, 0.0, max_tours);
}

/* ================================================================== */
/*  Geodesic Deep-LLL (G-DLLL)                                        */
/*                                                                     */
/*  The canonical selector exhaustively maximizes                     */
/*      S(k,j) = ΔV(k,j) / (k-j).                                  */
/*  The two-position recurrence evaluates each candidate in O(1),    */
/*  giving O(d^2) score work per iteration without a shortlist.       */
/*  n_equiv_swaps records W = Σ(k_s-j_s).                          */
/* ================================================================== */

/*
 * Comparison helper for argsort (descending by deficit).
 * We pack {deficit, index} into a struct to sort stably.
 */
typedef struct {
    double deficit;
    int    idx;
} deficit_entry_t;

static int cmp_deficit_desc(const void *a, const void *b) {
    const deficit_entry_t *da = (const deficit_entry_t *)a;
    const deficit_entry_t *db = (const deficit_entry_t *)b;
    if (db->deficit > da->deficit) return 1;
    if (db->deficit < da->deficit) return -1;
    return 0;
}

static int cmp_deficit_asc(const void *a, const void *b) {
    const deficit_entry_t *da = (const deficit_entry_t *)a;
    const deficit_entry_t *db = (const deficit_entry_t *)b;
    if (da->deficit > db->deficit) return 1;
    if (da->deficit < db->deficit) return -1;
    return 0;
}

typedef struct {
    deep_result_t base;
    double        V0;         /* initial excess variance              */
    double        alpha_used; /* c_delta used for deficit computation */
} gdlll_result_t;

static inline long double gdlll_roi_score(long double dV, int depth,
                                          double fixed_cost) {
    return dV / ((long double)fixed_cost + depth);
}

template <class FT>
static gdlll_result_t
run_gdlll_impl_scored(ZZ_mat<mpz_t> &B, int d, int m, double delta, int gso_flags,
                      double fixed_cost, long double drop_fraction) {
    gdlll_result_t gres;
    memset(&gres, 0, sizeof(gres));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int max_ops = 1000 * d;

    long double *mu = (long double *)calloc((size_t)d * d, sizeof(long double));
    long double *r  = (long double *)calloc((size_t)d, sizeof(long double));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    M.update_gso();
    read_gso_wide(M, d, mu, r);

    gres.V0 = (double)centered_log_energy(r, d);

    while (gres.base.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso_wide(M, d, mu, r);

        int         best_k = -1, best_j = -1;
        int         fallback_k = -1;
        long double best_roi   = 0.0L;
        long double min_drop   = drop_fraction * centered_log_energy(r, d);
        for (int k = 1; k < d; k++) {
            long double P_next = r[k];
            long double dV     = 0.0L;
            for (int j = k - 1; j >= 0; j--) {
                long double mkj = mu[k * d + j];
                long double Pc  = P_next + mkj * mkj * r[j];
                dV              = extend_variance_drop(dV, r[j], P_next, Pc);
                if (j == k - 1 && Pc < delta * r[j] && fallback_k < 0) {
                    fallback_k = k;
                }
                int         depth = k - j;
                long double roi   = gdlll_roi_score(dV, depth, fixed_cost);
                if (dV > min_drop && roi > best_roi) {
                    best_roi = roi;
                    best_k   = k;
                    best_j   = j;
                }
                P_next = Pc;
            }
        }
        if (best_k < 0) {
            if (fallback_k < 0) break;
            best_k = fallback_k;
            best_j = fallback_k - 1;
            gres.base.n_fallback_swaps++;
        }

        M.move_row(best_k, best_j);
        gres.base.n_ops++;
        gres.base.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }
    if (gres.base.n_ops >= max_ops) gres.base.hit_op_limit = 1;

    L.size_reduction(0, d);
    read_gso_wide(M, d, mu, r);
    gres.base.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    gres.base.delta0      = compute_delta0(B, d, m, r);
    gres.base.final_var   = profile_variance(r, d);
    gres.base.kappa_f     = max_relative_variance_drop(mu, r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    gres.base.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    free(mu);
    free(r);
    return gres;
}

static gdlll_result_t run_gdlll(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                long double drop_fraction) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_gdlll_impl_scored<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM, 0.0,
                                                    drop_fraction);
    }
    return run_gdlll_impl_scored<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, 0.0,
                                                drop_fraction);
}

static gdlll_result_t run_gdlll_costaware(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                          double      fixed_cost,
                                          long double drop_fraction) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_gdlll_impl_scored<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM,
                                                    fixed_cost, drop_fraction);
    }
    return run_gdlll_impl_scored<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, fixed_cost,
                                                drop_fraction);
}

template <class FT>
static gdlll_result_t
run_gdlll_residual_impl(ZZ_mat<mpz_t> &B, int d, int m, double delta, int gso_flags,
                        double tau_frac, double fixed_cost, int shortlist_b) {
    gdlll_result_t gres;
    memset(&gres, 0, sizeof(gres));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const int    max_ops = 1000 * d;
    const double c_delta = 0.5 * log(1.0 / fmax(delta - 0.25, 1e-15));
    gres.alpha_used      = c_delta;

    double          *mu      = (double *)calloc((size_t)d * d, sizeof(double));
    double          *r       = (double *)calloc((size_t)d, sizeof(double));
    double          *wr      = (double *)calloc((size_t)d, sizeof(double));
    double          *wP      = (double *)calloc((size_t)d, sizeof(double));
    double          *profile = (double *)calloc((size_t)d, sizeof(double));
    deficit_entry_t *src =
        (deficit_entry_t *)malloc((size_t)d * sizeof(deficit_entry_t));
    deficit_entry_t *snk =
        (deficit_entry_t *)malloc((size_t)d * sizeof(deficit_entry_t));

    ZZ_mat<mpz_t>                 eu, eut;
    MatGSO<Z_NR<mpz_t>, FT>       M(B, eu, eut, gso_flags);
    LLLReduction<Z_NR<mpz_t>, FT> L(M, delta, LLL_DEF_ETA, LLL_DEFAULT);

    int sr_from = 0;
    if (gso_flags == GSO_INT_GRAM) {
        L.lll(0, 0, d);
        L.size_reduction(0, d);
    }
    read_gso(M, d, mu, r);

    double V0 = 0.0;
    for (int i = 0; i < d; i++) {
        double p = 0.5 * log(fmax(r[i], 1e-30));
        V0 += p * p;
    }
    gres.V0 = V0;

    int Bshort = (shortlist_b > 0) ? shortlist_b : ((d >= 64) ? 8 : 6);
    if (Bshort > d) Bshort = d;

    while (gres.base.n_ops < max_ops) {
        L.size_reduction(sr_from, d);
        read_gso(M, d, mu, r);

        double sum_p = 0.0;
        double Vt    = 0.0;
        for (int i = 0; i < d; i++) {
            profile[i] = 0.5 * log(fmax(r[i], 1e-30));
            sum_p += profile[i];
            Vt += profile[i] * profile[i];
        }
        double mean_p       = sum_p / d;
        double dV_threshold = tau_frac * Vt;

        for (int i = 0; i < d; i++) {
            double p_star  = mean_p + c_delta * (double)(d - 1 - 2 * i) / 2.0;
            double z       = profile[i] - p_star;
            src[i].deficit = z;
            src[i].idx     = i;
            snk[i].deficit = z;
            snk[i].idx     = i;
        }
        qsort(src, (size_t)d, sizeof(deficit_entry_t), cmp_deficit_desc);
        qsort(snk, (size_t)d, sizeof(deficit_entry_t), cmp_deficit_asc);

        int    best_k = -1, best_j = -1;
        double best_score = 0.0;

        for (int si = 0; si < Bshort; si++) {
            int k = src[si].idx;
            if (k <= 0 || src[si].deficit <= 0.0) continue;

            for (int tj = 0; tj < Bshort; tj++) {
                int j = snk[tj].idx;
                if (j >= k || snk[tj].deficit >= 0.0) continue;

                double Pc = r[k];
                for (int ell = k - 1; ell >= j; ell--) {
                    double mkell = mu[k * d + ell];
                    Pc += mkell * mkell * r[ell];
                }
                if (Pc < delta * r[j]) {
                    int    depth = k - j;
                    double dV    = score_deep_var(mu, r, d, k, j, wr, wP);
                    if (dV < dV_threshold) continue;
                    double score = gdlll_roi_score(dV, depth, fixed_cost);
                    if (score > best_score) {
                        best_score = score;
                        best_k     = k;
                        best_j     = j;
                    }
                }
            }
        }

        if (best_k < 0) {
            double best_any = 0.0;
            int    any_k = -1, any_j = -1;
            for (int k = 1; k < d; k++) {
                double Pc = r[k];
                for (int j = k - 1; j >= 0; j--) {
                    double mkj = mu[k * d + j];
                    Pc += mkj * mkj * r[j];
                    if (Pc < delta * r[j]) {
                        int    depth = k - j;
                        double dV    = score_deep_var(mu, r, d, k, j, wr, wP);
                        double score = gdlll_roi_score(dV, depth, fixed_cost);
                        if (dV >= dV_threshold && score > best_score) {
                            best_score = score;
                            best_k     = k;
                            best_j     = j;
                        }
                        if (score > best_any) {
                            best_any = score;
                            any_k    = k;
                            any_j    = j;
                        }
                    }
                }
            }
            if (best_k < 0) {
                best_k = any_k;
                best_j = any_j;
            }
        }

        if (best_k < 0) break;

        M.move_row(best_k, best_j);
        gres.base.n_ops++;
        gres.base.n_equiv_swaps += best_k - best_j;
        sr_from = best_j;
    }

    if (gres.base.n_ops >= max_ops) gres.base.hit_op_limit = 1;
    L.size_reduction(0, d);
    read_gso(M, d, mu, r);
    gres.base.lll_reduced = is_lll_reduced(M, delta, LLL_DEF_ETA);
    gres.base.delta0      = compute_delta0(B, d, m, r);
    gres.base.final_var   = profile_variance(r, d);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    gres.base.elapsed_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    free(mu);
    free(r);
    free(wr);
    free(wP);
    free(profile);
    free(src);
    free(snk);
    return gres;
}

static gdlll_result_t run_gdlll_residual(ZZ_mat<mpz_t> &B, int d, int m, double delta,
                                         double tau_frac, double fixed_cost,
                                         int shortlist_b) {
    int mb = compute_max_bits(B, d, m);
    if (mb > 26) {
        FP_NR<mpfr_t>::set_prec(2 * mb + 64);
        return run_gdlll_residual_impl<FP_NR<mpfr_t>>(B, d, m, delta, GSO_INT_GRAM,
                                                      tau_frac, fixed_cost,
                                                      shortlist_b);
    }
    return run_gdlll_residual_impl<FP_NR<double>>(B, d, m, delta, GSO_DEFAULT, tau_frac,
                                                  fixed_cost, shortlist_b);
}

#endif /* DEEP_LLL_H */
