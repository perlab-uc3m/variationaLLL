/*
 * gdlll_benchmark.cpp — Geodesic Deep-LLL (G-DLLL) benchmark driver.
 *
 * Benchmarks the G-DLLL selector (Section 4.4, Proposition prop:roi) against
 * Deep-Var, SS-GG, and Thermal-Adaptive across the same lattice families and
 * dimensions as benchmark.cpp.
 *
 * Key metric differences from benchmark.cpp:
 *   n_ops         — number of deep insertions (same as before)
 *   n_equiv_swaps — total cascade depth W = Σ(k_s - j_s)  ← primary metric
 *                   Lower W means fewer equivalent adjacent-swap operations.
 *
 * Usage:
 *   ./gdlll_benchmark [--dims 20,40,...] [--nlat N] [--seed 42] [--delta 0.99]
 *                     [--families uniform,gaussian,qary,goldstein-mayer]
 *                     [-o out.json]
 *
 * Build (after updating CMakeLists.txt):
 *   cd build && cmake .. && make gdlll_benchmark
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <climits>
#include <vector>
#include <omp.h>

#include "lattice_gen.h"

/* ================================================================== */
/*  Statistics                                                        */
/* ================================================================== */

typedef struct {
    double mean, std;
} stat_t;

static stat_t compute_stat(const double *vals, const double *invalid, int n) {
    stat_t s     = {0.0, 0.0};
    int    count = 0;
    for (int i = 0; i < n; i++) {
        if (invalid[i] != 0.0) continue;
        s.mean += vals[i];
        count++;
    }
    if (count == 0) {
        s.mean = s.std = NAN;
        return s;
    }
    s.mean /= count;
    for (int i = 0; i < n; i++) {
        if (invalid[i] != 0.0) continue;
        double d = vals[i] - s.mean;
        s.std += d * d;
    }
    s.std = sqrt(s.std / count);
    return s;
}

/* ================================================================== */
/*  JSON helpers                                                      */
/* ================================================================== */

static int  jff;
static void jo(FILE *f) {
    fprintf(f, "{");
    jff = 1;
}
static void jc(FILE *f) {
    fprintf(f, "}");
}
static void jk(FILE *f, const char *n) {
    if (!jff) fprintf(f, ",");
    jff = 0;
    fprintf(f, "\"%s\":", n);
}
static void jko(FILE *f, const char *n) {
    jk(f, n);
    jo(f);
}
static void ji(FILE *f, const char *n, int v) {
    jk(f, n);
    fprintf(f, "%d", v);
}
static void jd(FILE *f, const char *n, double v) {
    jk(f, n);
    if (isfinite(v)) fprintf(f, "%.10g", v);
    else fprintf(f, "null");
}

static void json_alg(FILE *f, const char *name, const double *ops, const double *eqs,
                     const double *fallback, const double *limits,
                     const double *reduced, const double *times, const double *d0s,
                     const double *vars, const double *kappas, const double *alphas,
                     int n) {
    jko(f, name);
    jk(f, "s_alpha");
    fprintf(f, "[");
    for (int i = 0; i < n; i++) fprintf(f, i ? ",%.10g" : "%.10g", alphas[i]);
    fprintf(f, "]");
    stat_t s;
    s = compute_stat(ops, limits, n);
    jd(f, "mean_ops", s.mean);
    jd(f, "std_ops", s.std);
    s = compute_stat(eqs, limits, n);
    jd(f, "mean_equiv_swaps", s.mean);
    jd(f, "std_equiv_swaps", s.std);
    s = compute_stat(fallback, limits, n);
    jd(f, "mean_fallback_swaps", s.mean);
    jd(f, "std_fallback_swaps", s.std);
    int limit_hits = 0;
    for (int i = 0; i < n; i++) limit_hits += (limits[i] != 0.0);
    int non_lll_outputs = 0;
    for (int i = 0; i < n; i++)
        non_lll_outputs += (limits[i] == 0.0 && reduced[i] == 0.0);
    ji(f, "completed_runs", n - limit_hits);
    ji(f, "limit_hits", limit_hits);
    ji(f, "non_lll_outputs", non_lll_outputs);
    s = compute_stat(times, limits, n);
    jd(f, "mean_time", s.mean);
    jd(f, "std_time", s.std);
    s = compute_stat(d0s, limits, n);
    jd(f, "mean_delta0", s.mean);
    jd(f, "std_delta0", s.std);
    s = compute_stat(vars, limits, n);
    jd(f, "mean_final_var", s.mean);
    jd(f, "std_final_var", s.std);
    s = compute_stat(kappas, limits, n);
    jd(f, "mean_kappa_f", s.mean);
    jd(f, "std_kappa_f", s.std);
    double max_kappa = NAN;
    for (int i = 0; i < n; i++) {
        if (limits[i] != 0.0) continue;
        if (!isfinite(max_kappa) || kappas[i] > max_kappa) max_kappa = kappas[i];
    }
    jd(f, "max_kappa_f", max_kappa);
    /* Per-sample values (same bases across algorithms) for paired statistics. */
    const double *arrs[] = {ops, eqs, fallback, times, d0s, vars, limits, reduced};
    const char   *keys[] = {"s_ops", "s_equiv_swaps", "s_fallback", "s_time",
                            "s_delta0", "s_final_var", "s_limit", "s_lll"};
    for (int a = 0; a < 8; a++) {
        jk(f, keys[a]);
        fprintf(f, "[");
        for (int i = 0; i < n; i++) fprintf(f, i ? ",%.10g" : "%.10g", arrs[a][i]);
        fprintf(f, "]");
    }
    jc(f);
}

/* ================================================================== */
/*  Argument parsing                                                  */
/* ================================================================== */

static int parse_ints(const char *s, int *out, int max_n) {
    int  n = 0;
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *tok            = strtok(buf, ",");
    while (tok && n < max_n) {
        out[n++] = atoi(tok);
        tok      = strtok(NULL, ",");
    }
    return n;
}

static int parse_strings(const char *s, char out[][32], int max_n) {
    int  n = 0;
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *tok            = strtok(buf, ",");
    while (tok && n < max_n) {
        strncpy(out[n], tok, 31);
        out[n][31] = '\0';
        n++;
        tok = strtok(NULL, ",");
    }
    return n;
}

/* ================================================================== */
/*  Algorithm wrappers (deep-insertion only)                          */
/* ================================================================== */

#define N_ALG 10
static const char *ALG_NAMES[N_ALG] = {"LLL",          "Deep-Var",
                                       "SS-GG",        "Thermal-Adaptive",
                                       "G-DLLL",       "G-DLLL-CA",
                                       "G-DLLL-RT",    "Schur-K",
                                       "Thermal-Band", "Thermal-TwoClock"};

/* Parameters used only by the exploratory appendix variants. */
static double g_tau        = 0.01;
static double g_fixed      = 8.0;
static double g_score_drop = 1e-6;
static int    g_targetb    = 0;

static deep_result_t wrap_lll(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    return run_lll_standard(B, d, m, delta);
}
static deep_result_t wrap_deepvar(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    return run_deep_var(B, d, m, delta, g_score_drop);
}
static deep_result_t wrap_ssgg(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    return run_deep_ssgg(B, d, m, delta, g_score_drop);
}
static deep_result_t wrap_thermal(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    /* γ = 2 gives the q-ary anchor α(CV=1) = 1. */
    return run_deep_inita(B, d, m, delta, 2.0, g_score_drop);
}
static deep_result_t wrap_gdlll(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    gdlll_result_t gr = run_gdlll(B, d, m, delta, g_score_drop);
    return gr.base;
}
static deep_result_t wrap_gdlll_ca(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    gdlll_result_t gr = run_gdlll_costaware(B, d, m, delta, g_fixed, g_score_drop);
    return gr.base;
}
static deep_result_t wrap_gdlll_rt(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    gdlll_result_t gr = run_gdlll_residual(B, d, m, delta, g_tau, g_fixed, g_targetb);
    return gr.base;
}
static deep_result_t wrap_schurK(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    /* Non-separable Schur-convex score: top-K partial sum of sorted */
    /* log-profile. Prototype; probes the gap in Prop. canonical.    */
    return run_deep_schurK(B, d, m, delta);
}
static deep_result_t wrap_thermal_band(ZZ_mat<mpz_t> &B, int d, int m, double delta) {
    return run_deep_band(B, d, m, delta, 2.0, g_score_drop);
}
static deep_result_t wrap_thermal_two_clock(ZZ_mat<mpz_t> &B, int d, int m,
                                            double delta) {
    return run_deep_two_clock(B, d, m, delta, 2.0, g_score_drop);
}

typedef deep_result_t (*alg_fn_t)(ZZ_mat<mpz_t> &, int, int, double);
static alg_fn_t ALG_FNS[N_ALG] = {wrap_lll,          wrap_deepvar,
                                  wrap_ssgg,         wrap_thermal,
                                  wrap_gdlll,        wrap_gdlll_ca,
                                  wrap_gdlll_rt,     wrap_schurK,
                                  wrap_thermal_band, wrap_thermal_two_clock};

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(int argc, char **argv) {
    /* ── Defaults ─────────────────────────────────────────────────── */
    int      dims[16] = {20, 30, 40, 50, 60, 80, 100, 120, 140, 160};
    int      n_dims   = 10;
    int      n_lat    = -1; /* -1 = auto-scale */
    unsigned seed     = 42;
    double   delta    = 0.99;
    char     families[8][32];
    strcpy(families[0], "gaussian");
    strcpy(families[1], "qary");
    int         n_fam                     = 2;
    const char *outpath                   = NULL;
    int         nthreads                  = 0;
    bool        main_only                 = false;
    bool        thermal_only              = false;
    int         alg_sel[N_ALG];
    int         n_alg_sel                 = 0;
    int         basis_scale               = 1;
    bool        compare_thermal_band      = false;
    bool        compare_thermal_two_clock = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dims") && i + 1 < argc)
            n_dims = parse_ints(argv[++i], dims, 16);
        else if (!strcmp(argv[i], "--nlat") && i + 1 < argc) n_lat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--delta") && i + 1 < argc) delta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--score-drop") && i + 1 < argc)
            g_score_drop = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tau") && i + 1 < argc) g_tau = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fixed-cost") && i + 1 < argc)
            g_fixed = atof(argv[++i]);
        else if (!strcmp(argv[i], "--target-b") && i + 1 < argc)
            g_targetb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--families") && i + 1 < argc)
            n_fam = parse_strings(argv[++i], families, 8);
        else if (!strcmp(argv[i], "--nthreads") && i + 1 < argc)
            nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--main-only")) main_only = true;
        else if (!strcmp(argv[i], "--thermal-only")) thermal_only = true;
        else if (!strcmp(argv[i], "--algs") && i + 1 < argc)
            n_alg_sel = parse_ints(argv[++i], alg_sel, N_ALG);
        else if (!strcmp(argv[i], "--thermal-fastpow")) g_thermal_fastpow = 1;
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) basis_scale = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fixed-alpha") && i + 1 < argc)
            g_thermal_force_alpha = atof(argv[++i]);
        else if (!strcmp(argv[i], "--compare-thermal-band"))
            compare_thermal_band = true;
        else if (!strcmp(argv[i], "--compare-thermal-two-clock"))
            compare_thermal_two_clock = true;
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outpath = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf(
                "Usage: gdlll_benchmark [options]\n"
                "  --dims   d1,d2,...  dimensions (default: 20..160)\n"
                "  --nlat   N          lattices per cell (-1=auto)\n"
                "  --seed   S          RNG seed (default 42)\n"
                "  --delta  D          LLL delta (default 0.99)\n"
                "  --score-drop R      minimum relative score drop (default 1e-6)\n"
                "  --tau    T          threshold for G-DLLL-RT only (default 0.01)\n"
                "  --fixed-cost C      cost-aware ROI offset for G-DLLL-CA (default "
                "8.0)\n"
                "  --target-b B        residual shortlist size for G-DLLL-RT (0=auto)\n"

                "  --families f1,...   families (default: gaussian,qary)\n"
                "  --nthreads T        OpenMP threads (0=auto)\n"
                "  --main-only         run LLL and the four main selectors only\n"
                "  --compare-thermal-band  run Thermal-Adaptive and Thermal-Band\n"
                "  --compare-thermal-two-clock  run Thermal-Adaptive and "
                "Thermal-TwoClock\n"
                "  -o file.json        output path (default: stdout)\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return 2;
        }
    }
    bool bad_dimension = n_dims == 0;
    for (int i = 0; i < n_dims; i++) bad_dimension = bad_dimension || dims[i] < 2;
    bool bad_family = n_fam == 0;
    for (int i = 0; i < n_fam; i++) {
        bool known = !strcmp(families[i], "uniform") ||
                     !strcmp(families[i], "gaussian") || !strcmp(families[i], "qary") ||
                     !strcmp(families[i], "goldstein-mayer");
        bad_family = bad_family || !known;
    }
    if (delta <= 0.25 || delta >= 1.0 || g_score_drop <= 0.0 || g_score_drop >= 1.0 ||
        g_tau < 0.0 || g_fixed < 0.0 || g_targetb < 0 || n_lat < -1 || n_lat == 0 ||
        nthreads < 0 || bad_family || bad_dimension) {
        fprintf(stderr, "Invalid benchmark parameter.\n");
        return 2;
    }

    FILE *out = outpath ? fopen(outpath, "w") : stdout;
    if (!out) {
        perror("fopen");
        return 1;
    }

    if (nthreads > 0) omp_set_num_threads(nthreads);
    int actual_threads = omp_get_max_threads();

    fprintf(stderr,
            "G-DLLL benchmark: %d families, %d dims, delta=%.2f, score_drop=%.1e, "
            "RT_tau=%.4f, fixed_cost=%.2f, target_b=%d, threads=%d\n",
            n_fam, n_dims, delta, g_score_drop, g_tau, g_fixed, g_targetb,
            actual_threads);

    /* Top-level JSON object */
    jo(out);
    fprintf(out, "\n");

    /* Metadata */
    jko(out, "meta");
    ji(out, "seed", (int)seed);
    jd(out, "delta", delta);
    jd(out, "score_drop_fraction", g_score_drop);
    jd(out, "residual_tau", g_tau);
    jd(out, "size_reduction_eta", LLL_DEF_ETA);
    jk(out, "lll_count_convention");
    fprintf(out, "\"batched_row_moves_and_observed_depth\"");
    jk(out, "canonical_gdlll_scan");
    fprintf(out, "\"exhaustive\"");
    jd(out, "fixed_cost", g_fixed);
    jd(out, "forced_alpha", g_thermal_force_alpha);
    ji(out, "basis_scale", basis_scale);
    ji(out, "thermal_fastpow", g_thermal_fastpow);
    ji(out, "target_b", g_targetb);
    ji(out, "threads", actual_threads);
    ji(out, "main_only", main_only ? 1 : 0);
    ji(out, "thermal_band_comparison", compare_thermal_band ? 1 : 0);
    ji(out, "thermal_two_clock_comparison", compare_thermal_two_clock ? 1 : 0);
    jc(out);
    fprintf(out, "\n");

    jko(out, "results");

    for (int fi = 0; fi < n_fam; fi++) {
        jko(out, families[fi]);

        for (int di = 0; di < n_dims; di++) {
            int d       = dims[di];
            int n_lat_d = (n_lat >= 0) ? n_lat
                          : (d <= 80)  ? 12 * actual_threads
                          : (d <= 120) ? 6 * actual_threads
                                       : 3 * actual_threads;

            std::vector<double> a_ops(n_lat_d), a_eqs(n_lat_d);
            std::vector<double> a_fallback(n_lat_d), a_limits(n_lat_d);
            std::vector<double> a_reduced(n_lat_d);
            std::vector<double> a_time(n_lat_d), a_d0(n_lat_d), a_var(n_lat_d);
            std::vector<double> a_kappa(n_lat_d), a_alpha(n_lat_d);

            char dim_str[16];
            snprintf(dim_str, sizeof(dim_str), "%d", d);
            jko(out, dim_str);

            fprintf(stderr, "  %s d=%d n=%d\n", families[fi], d, n_lat_d);

            /* Pre-generate bases */
            ZZ_mat<mpz_t> *bases = new ZZ_mat<mpz_t>[n_lat_d];
            for (int li = 0; li < n_lat_d; li++) {
                /* Sample identity depends only on (seed, family, d, index), so the
                   same bases recur across separate runs; GMP's generator (used by
                   fplll's q-ary prime generator) is reseeded as well. */
                unsigned fam_code = !strcmp(families[fi], "gaussian")          ? 0u
                                    : !strcmp(families[fi], "qary")            ? 1u
                                    : !strcmp(families[fi], "goldstein-mayer") ? 2u
                                                                               : 3u;
                unsigned sample_seed =
                    seed + fam_code * 100000u + (unsigned)d * 100u + (unsigned)li;
                srand(sample_seed);
                RandGen::init_with_seed(sample_seed);
                if (!strcmp(families[fi], "uniform")) gen_uniform(bases[li], d, 10);
                else if (!strcmp(families[fi], "gaussian"))
                    gen_gaussian(bases[li], d, 5.0);
                else if (!strcmp(families[fi], "qary")) gen_qary(bases[li], d, 1009);
                else if (!strcmp(families[fi], "goldstein-mayer"))
                    gen_goldstein_mayer(bases[li], d, 1009);
                else {
                    fprintf(stderr, "Unknown family: %s\n", families[fi]);
                    return 1;
                }
                if (basis_scale > 1)
                    for (int r0 = 0; r0 < d; r0++)
                        for (int c0 = 0; c0 < d; c0++)
                            bases[li][r0][c0].mul_si(bases[li][r0][c0], basis_scale);
            }

            std::vector<int> algorithm_indices;
            if (n_alg_sel > 0) {
                for (int q = 0; q < n_alg_sel; q++) algorithm_indices.push_back(alg_sel[q]);
            } else if (thermal_only) {
                algorithm_indices.push_back(3);
            } else if (compare_thermal_two_clock) {
                algorithm_indices.push_back(3);
                algorithm_indices.push_back(9);
            } else if (compare_thermal_band) {
                algorithm_indices.push_back(3);
                algorithm_indices.push_back(8);
            } else {
                int n_algorithms = main_only ? 5 : N_ALG;
                for (int ai = 0; ai < n_algorithms; ai++)
                    algorithm_indices.push_back(ai);
            }
            for (int ai : algorithm_indices) {
/* Run in parallel over lattice instances */
#pragma omp parallel for schedule(dynamic)
                for (int li = 0; li < n_lat_d; li++) {
                    ZZ_mat<mpz_t> B   = copy_basis(bases[li]);
                    deep_result_t res = ALG_FNS[ai](B, d, d, delta);
                    a_ops[li]         = (double)res.n_ops;
                    a_eqs[li]         = (double)res.n_equiv_swaps;
                    a_fallback[li]    = (double)res.n_fallback_swaps;
                    a_limits[li]      = (double)res.hit_op_limit;
                    a_reduced[li]     = (double)res.lll_reduced;
                    a_time[li]        = res.elapsed_sec;
                    a_d0[li]          = res.delta0;
                    a_var[li]         = res.final_var;
                    a_kappa[li]       = res.kappa_f;
                    a_alpha[li]       = res.alpha_used;
                }
                json_alg(out, ALG_NAMES[ai], a_ops.data(), a_eqs.data(),
                         a_fallback.data(), a_limits.data(), a_reduced.data(),
                         a_time.data(), a_d0.data(), a_var.data(), a_kappa.data(),
                         a_alpha.data(), n_lat_d);
                fprintf(out, "\n");
                fflush(out);
            }

            delete[] bases;
            jc(out); /* close dim_str */
            fprintf(out, "\n");
            fflush(out);
        }

        jc(out); /* close family */
        fprintf(out, "\n");
    }

    jc(out); /* close results */
    fprintf(out, "\n");
    jc(out); /* close root */
    fprintf(out, "\n");

    if (outpath) fclose(out);
    return 0;
}
