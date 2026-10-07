/*
 * lattice_gen.h — Lattice basis generators (single source of truth).
 *
 * Families
 * --------
 *   uniform         Generic Basis Lattice — i.i.d. uniform entries in [-bound, bound]
 *   gaussian        Generic Basis Lattice — i.i.d. Gaussian entries (Box-Muller)
 *   qary            Structured q-ary lattice [q*I_k, 0; A, I_{d-k}],  k = d/2
 *   qary_var        Same structure with caller-supplied rank k
 *   goldstein-mayer Random Lattice [GM03] — q-ary with k=1 and q prime
 *
 * Paper terminology note
 * ----------------------
 *   "Generic Basis Lattice" : uniform / gaussian families
 *   "Random Lattice"        : goldstein-mayer (Haar measure on SL_n(R)/SL_n(Z)
 *                             in the limit of large prime q)
 *
 * All functions are static to avoid multiple-definition errors when the header
 * is included in more than one translation unit.  Follows the same style as
 * deep_lll.h.
 *
 * Dependency: deep_lll.h (for copy_basis and the fplll / GMP types).
 */
#ifndef LATTICE_GEN_H
#define LATTICE_GEN_H

#include "deep_lll.h" /* ZZ_mat, copy_basis, lll_reduction, fplll types */

/* ================================================================== */
/*  Generic Basis Lattices                                            */
/* ================================================================== */

/* Uniform integer entries drawn from [-bound, bound].
 * Retries until the basis is full-rank (verified via a quick LLL pass). */
static void gen_uniform(ZZ_mat<mpz_t> &B, int d, int bound) {
    B.resize(d, d);
    int attempts = 0;
    while (attempts++ < 500) {
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) B[i][j] = (rand() % (2 * bound + 1)) - bound;
        /* Quick rank check via LLL — check that the reduced copy has no zero row */
        ZZ_mat<mpz_t> tmp = copy_basis(B);
        int s = lll_reduction(tmp, 0.99, 0.51, LM_WRAPPER, FT_DEFAULT, 0, LLL_DEFAULT);
        bool full_rank = (s == RED_SUCCESS);
        for (int i = 0; i < d && full_rank; i++) full_rank = !tmp[i].is_zero();
        if (full_rank) return;
    }
    fprintf(stderr, "gen_uniform: failed after 500 attempts\n");
    abort();
}

/* Gaussian entries with standard deviation sigma (Box-Muller sampling).
 * Retries until the basis is full-rank. */
static void gen_gaussian(ZZ_mat<mpz_t> &B, int d, double sigma) {
    B.resize(d, d);
    int attempts = 0;
    while (attempts++ < 500) {
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
                double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
                double g  = sigma * sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
                B[i][j]   = (long)lround(g);
            }
        ZZ_mat<mpz_t> tmp = copy_basis(B);
        int s = lll_reduction(tmp, 0.99, 0.51, LM_WRAPPER, FT_DEFAULT, 0, LLL_DEFAULT);
        bool full_rank = (s == RED_SUCCESS);
        for (int i = 0; i < d && full_rank; i++) full_rank = !tmp[i].is_zero();
        if (full_rank) return;
    }
    fprintf(stderr, "gen_gaussian: failed after 500 attempts\n");
    abort();
}

/* ================================================================== */
/*  Structured q-ary lattices                                         */
/* ================================================================== */

/* Standard q-ary lattice: [q*I_k, 0; A, I_{d-k}]  with k = d/2. */
static void gen_qary(ZZ_mat<mpz_t> &B, int d, int q) {
    B.resize(d, d);
    int k = d / 2;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) B[i][j] = 0;
    for (int i = 0; i < k; i++) B[i][i] = q;
    for (int i = k; i < d; i++) {
        for (int j = 0; j < k; j++) B[i][j] = rand() % q;
        B[i][i] = 1;
    }
}

/* q-ary lattice with caller-supplied rank k (used by alpha_sweep configs).
 * Negative rank_k is treated as a divisor: rank_k = d / (-rank_k). */
static void gen_qary_var(ZZ_mat<mpz_t> &B, int d, int q, int rank_k) {
    if (rank_k < 0) rank_k = d / (-rank_k);
    if (rank_k < 1) rank_k = 1;
    if (rank_k >= d) rank_k = d - 1;
    B.resize(d, d);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) B[i][j] = 0;
    for (int i = 0; i < rank_k; i++) B[i][i] = q;
    for (int i = rank_k; i < d; i++) {
        for (int j = 0; j < rank_k; j++) B[i][j] = rand() % q;
        B[i][i] = 1;
    }
}

/* ================================================================== */
/*  Random Lattice (Goldstein-Mayer [GM03])                           */
/* ================================================================== */

/* q-ary with k=1 and q a random 10d-bit prime — distributed as a
 * random lattice in the sense of Goldstein and Mayer.  Approximates
 * the Haar measure on SL_n(R)/SL_n(Z) in the limit of large prime q.
 *
 * Convention: q is a random prime of bit-length 10d, matching
 * Chen & Nguyen (2011) and Bhattacherjee et al. (2025).  This is
 * also the convention used by the Darmstadt SVP Challenge.
 *
 * Uses fplll's native gen_qary_prime which produces the HNF form
 * [I_{d-1}, x; 0, q].  The resulting lattice is the same (up to
 * unimodular transform) as the [q, 0; x, I] convention.
 *
 * Legacy wrapper with fixed int q is kept for backward compatibility. */
static void gen_goldstein_mayer(ZZ_mat<mpz_t> &B, int d, int /*q_unused*/) {
    B.resize(d, d);
    B.gen_qary_prime(1, 10 * d);
}

#endif /* LATTICE_GEN_H */
