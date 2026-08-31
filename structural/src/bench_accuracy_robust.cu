// Robust accuracy harness for the COVE value axis.
//
// Goal: harden the ACCURACY claim with a FAITHFUL, reusable harness that runs the
// REAL BitBSR 8x4 operator + the PRODUCTION value codecs + the PRODUCTION decode
// kernels (it reuses the exact builders/device-wrappers from the headers, so the
// reported error is the operator's error, not a re-derivation). It is modeled on
// structural/src/check_outlier_globalnorm.cu (production builders + flag load-balance
// decode device wrappers) and structural/src/bench_bitbsr_value.cu (real BitBSR build
// + the legacy fixed input vector). Standalone (not in the Makefile); build with:
//
//   export PATH=/usr/local/cuda/bin:$PATH CPATH=/usr/local/cuda/include
//   nvcc -O3 -std=c++17 --extended-lambda \
//     -gencode arch=compute_120a,code=sm_120a -Istructural/include \
//     -o structural/build/bench_accuracy_robust structural/src/bench_accuracy_robust.cu -lcusparse
//
// For a given matrix and the value codecs {bfp8, bfp8_outlier(P=0.005), bfp4,
// bf16, fp16} (reference = the fp64 `original` LB kernel) it does THREE things:
//
//  (1) x-ENSEMBLE + L2: an ensemble of N>=16 input vectors x (uniform(-1,1),
//      N(0,1), the canonical zero-mean x (i%17-8)*0.125 used by the production
//      check_outlier_globalnorm.cu §12 metric [the "fixed-x" anchor], the legacy
//      fixed x 1+(i%7)*0.1, all-ones), deterministically seeded. For each x:
//      fp64 reference y0 = original(fp64)*x (double accumulate)
//      and each codec's y_hat (production decode + fp32 accumulate). TWO metrics
//      per (codec, x): max-row global-norm  max_r|y_hat_r - y0_r| / max_r|y0_r|
//      and L2 relative  ||y_hat - y0||_2 / ||y0||_2. Reports per (matrix,codec)
//      the worst (max over x) and mean over x for each metric.
//
//  (3) ERROR DECOMPOSITION: a second bfp8 variant that decodes the SAME int8 +
//      bf16 block scale but ACCUMULATES IN FP64 (quantization-only error). The
//      gap to fp32-accumulate bfp8 is the fp32-accumulation contribution.
//
//  (2) is the python Pareto figure (separate script).
//
// CSV: structural/results/studies/accuracy_robust_2026-06-09.csv with columns
//   matrix, codec, accum(fp32|fp64), bytes_per_nnz, rel_maxrow_worst,
//   rel_maxrow_mean, rel_l2_worst, rel_l2_mean, nnz, num_blocks.

#include "structural/gpu_convert.cuh"
#include "structural/gpu_value_codecs.cuh"
#include "structural/matrix_market.hpp"
#include "structural/operators/bitbsr_spmv_8x4_common.cuh"
#include "structural/spmv.hpp"
#include "structural/value_compression.hpp"  // bf16_bits_to_float for fp64-accum decode

#include <thrust/device_vector.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using structural::Index;
using structural::Offset;

// ---------------------------------------------------------------------------
// x ensemble: N>=16 vectors, mixed distributions, deterministically seeded.
// kinds: uniform(-1,1) [seeds 0..K-1], N(0,1) [seeds K..2K-1], legacy fixed
// 1+(i%7)*0.1, all-ones.
// ---------------------------------------------------------------------------
enum class XKind { kRandom, kFixedAnchor, kFixedLegacy, kAllOnes };
struct XVector {
    std::string label;
    XKind kind;
    std::vector<double> x;
};

static std::vector<XVector> make_x_ensemble(Index cols) {
    std::vector<XVector> ensemble;
    const std::size_t n = static_cast<std::size_t>(cols);
    const int kUniform = 7;  // seeds 0..6
    const int kNormal = 7;   // seeds 100..106
    // uniform(-1, 1)
    for (int s = 0; s < kUniform; ++s) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(s));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        XVector v;
        v.label = "uniform_s" + std::to_string(s);
        v.kind = XKind::kRandom;
        v.x.resize(n);
        for (std::size_t i = 0; i < n; ++i) v.x[i] = dist(rng);
        ensemble.push_back(std::move(v));
    }
    // N(0, 1)
    for (int s = 0; s < kNormal; ++s) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(100 + s));
        std::normal_distribution<double> dist(0.0, 1.0);
        XVector v;
        v.label = "normal_s" + std::to_string(s);
        v.kind = XKind::kRandom;
        v.x.resize(n);
        for (std::size_t i = 0; i < n; ++i) v.x[i] = dist(rng);
        ensemble.push_back(std::move(v));
    }
    // Canonical zero-mean x used by the production §12 global-norm metric
    // (check_outlier_globalnorm.cu): (i%17 - 8) * 0.125, range [-1, 1]. This is
    // the "fixed-x" anchor whose bfp8 numbers match the documented sanity values.
    {
        XVector v;
        v.label = "fixed_anchor";
        v.kind = XKind::kFixedAnchor;
        v.x.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            v.x[i] = static_cast<double>(static_cast<int>(i % 17) - 8) * 0.125;
        ensemble.push_back(std::move(v));
    }
    // legacy fixed x (the one bench_bitbsr_value.cu uses): 1 + (i%7)*0.1 (all
    // positive -> stresses cancellation-heavy matrices).
    {
        XVector v;
        v.label = "fixed_legacy";
        v.kind = XKind::kFixedLegacy;
        v.x.resize(n);
        for (std::size_t i = 0; i < n; ++i)
            v.x[i] = 1.0 + static_cast<double>(i % 7) * 0.1;
        ensemble.push_back(std::move(v));
    }
    // all ones (row-sum probe; adversarial for cancellation-heavy matrices).
    {
        XVector v;
        v.label = "all_ones";
        v.kind = XKind::kAllOnes;
        v.x.assign(n, 1.0);
        ensemble.push_back(std::move(v));
    }
    return ensemble;  // 7 + 7 + 1 + 1 + 1 = 17
}

// ---------------------------------------------------------------------------
// Error metrics vs the fp64 reference y0.
// ---------------------------------------------------------------------------
static double rel_maxrow(const std::vector<double>& y0, const std::vector<double>& y) {
    double max_abs_err = 0.0, max_abs_ref = 0.0;
    for (std::size_t r = 0; r < y0.size(); ++r) {
        max_abs_err = std::max(max_abs_err, std::abs(y[r] - y0[r]));
        max_abs_ref = std::max(max_abs_ref, std::abs(y0[r]));
    }
    return max_abs_err / (max_abs_ref + 1e-30);
}

static double rel_l2(const std::vector<double>& y0, const std::vector<double>& y) {
    double num = 0.0, den = 0.0;
    for (std::size_t r = 0; r < y0.size(); ++r) {
        const double d = y[r] - y0[r];
        num += d * d;
        den += y0[r] * y0[r];
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

// Accumulator for worst/mean over the x ensemble (one per metric). Also tracks
// the worst split by INPUT CLASS: the RANDOM subset (uniform(-1,1) + N(0,1)),
// where ||y0|| is well-conditioned and the "L2 < max-row" relationship can be
// reported cleanly, and the STRUCTURED subset (all-ones, legacy 1+(i%7)*.1,
// zero-mean fixed (i%17-8)*.125) whose all-positive / cancellation-collapsing
// inputs drive ||y0||->0 on cancellation-heavy matrices and inflate the error
// for EVERY reduced-precision codec (a matrix x input conditioning effect, not a
// codec property). The headline figure uses the RANDOM worst; the structured
// worst is reported as a separately-noted conditioning caveat.
struct MetricAccum {
    double maxrow_worst = 0.0, maxrow_sum = 0.0;
    double l2_worst = 0.0, l2_sum = 0.0;
    double maxrow_worst_rand = 0.0, l2_worst_rand = 0.0;
    double maxrow_worst_struct = 0.0, l2_worst_struct = 0.0;
    int count = 0;
    void add(double maxrow, double l2, bool is_random) {
        maxrow_worst = std::max(maxrow_worst, maxrow);
        l2_worst = std::max(l2_worst, l2);
        maxrow_sum += maxrow;
        l2_sum += l2;
        ++count;
        if (is_random) {
            maxrow_worst_rand = std::max(maxrow_worst_rand, maxrow);
            l2_worst_rand = std::max(l2_worst_rand, l2);
        } else {
            // Every non-random ensemble member is a structured probe.
            maxrow_worst_struct = std::max(maxrow_worst_struct, maxrow);
            l2_worst_struct = std::max(l2_worst_struct, l2);
        }
    }
    double maxrow_mean() const { return count ? maxrow_sum / count : 0.0; }
    double l2_mean() const { return count ? l2_sum / count : 0.0; }
};

// ---------------------------------------------------------------------------
// (3) bfp8 decode with FP64 accumulation (quantization-only error).
// Reuses the EXACT production bfp8 payload (codec.int8_values, codec.block_scale_bf16)
// and the production bf16 scale decode (bf16_bits_to_float). The ONLY difference
// vs the production kernel is double-precision x / accumulate / y -- so the error
// vs the fp64 ref isolates quantization (no fp32-accumulation noise). The block /
// bit-rank walk mirrors spmv_bitbsr (host) and the device kernels exactly.
// ---------------------------------------------------------------------------
template <typename T>
static std::vector<double> bfp8_decode_fp64_accum(
    const structural::BitBsrMatrix<T>& host,
    const structural::Bfp8BlkScaleValueCodec& bfp8,
    const std::vector<double>& x) {
    std::vector<double> y(static_cast<std::size_t>(host.rows), 0.0);
    for (Index br = 0; br < host.num_block_rows; ++br) {
        for (Offset b = host.block_row_ptr[br]; b < host.block_row_ptr[br + 1]; ++b) {
            const Index bc = host.block_col_idx[b];
            const double scale = static_cast<double>(structural::bf16_bits_to_float(
                bfp8.block_scale_bf16[static_cast<std::size_t>(b)]));
            const auto bitmap =
                static_cast<std::uint32_t>(host.bitmap_words[static_cast<std::size_t>(b)]);
            const Offset value_base = host.block_val_ptr[b];
            for (int bit = 0; bit < 32; ++bit) {
                if ((bitmap & (std::uint32_t{1} << bit)) == 0U) continue;
                const std::uint32_t before = bitmap & ((std::uint32_t{1} << bit) - 1U);
                const Offset rank = structural::popcount_word(before);
                const Index local_row = bit / host.block_cols;
                const Index local_col = bit % host.block_cols;
                const Index row = br * host.block_rows + local_row;
                const Index col = bc * host.block_cols + local_col;
                if (row >= host.rows || col >= host.cols) continue;
                const double decoded =
                    static_cast<double>(bfp8.int8_values[static_cast<std::size_t>(value_base + rank)]) *
                    scale;
                y[static_cast<std::size_t>(row)] += decoded * x[static_cast<std::size_t>(col)];
            }
        }
    }
    return y;
}

// A row in the CSV / report.
struct CodecResult {
    std::string codec;
    std::string accum;  // "fp32" or "fp64"
    double bytes_per_nnz = 0.0;
    MetricAccum acc;
    // Per-input maxrow captured for the report: the canonical "fixed-x" anchor
    // (production §12 metric x) used for the worst/fixed robustness ratio, plus
    // the all-positive legacy and all-ones probes.
    double maxrow_anchor = 0.0;
    double maxrow_legacy = 0.0;
    double maxrow_allones = 0.0;
    double l2_anchor = 0.0;
    // Record one (codec, x) measurement.
    void record(double maxrow, double l2, XKind kind) {
        acc.add(maxrow, l2, kind == XKind::kRandom);
        if (kind == XKind::kFixedAnchor) { maxrow_anchor = maxrow; l2_anchor = l2; }
        else if (kind == XKind::kFixedLegacy) maxrow_legacy = maxrow;
        else if (kind == XKind::kAllOnes) maxrow_allones = maxrow;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s matrix.mtx out.csv [append=0] [outlier_fraction=0.005]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string csv_path = argv[2];
    const bool append = (argc > 3) ? (std::atoi(argv[3]) != 0) : false;
    const double outlier_fraction = (argc > 4) ? std::atof(argv[4]) : 0.005;

    const char* name = path.c_str();
    if (const char* slash = std::strrchr(name, '/')) name = slash + 1;
    std::string matrix_name(name);
    if (matrix_name.size() > 4 && matrix_name.substr(matrix_name.size() - 4) == ".mtx")
        matrix_name = matrix_name.substr(0, matrix_name.size() - 4);

    // ---- Load matrix + build the REAL BitBSR 8x4 + packed_meta + flag LB layout.
    // (Same production pipeline as check_outlier_globalnorm.cu.)
    const auto csr = structural::read_matrix_market<double>(path);
    structural::GpuBitBsrWorkspace<double> workspace;
    structural::DeviceBitBsrMatrix<double> device;
    structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(csr, workspace, device);
    const auto host_bitbsr = device.to_host();

    thrust::device_vector<std::uint64_t> packed_meta;
    structural::make_bitbsr_8x4_col_bitmap_packed(device, packed_meta);
    structural::BitBsrLoadBalanceLayout8x4 layout;
    structural::make_bitbsr_8x4_col_bitmap_flag_load_balance_work_items(
        device, layout, /*chunk_blocks=*/32, /*split_threshold_blocks=*/32, packed_meta);

    const Offset nnz = device.nnz;
    const Offset num_blocks = device.num_blocks;
    const double nnz_d = static_cast<double>(nnz);

    // ---- Build the PRODUCTION value codecs once (payload is x-independent).
    const auto bfp8 = structural::build_checked_bfp8_blkscale_value_codec(host_bitbsr);
    const auto outlier = structural::build_checked_bfp8_outlier_blkscale_value_codec(
        host_bitbsr, outlier_fraction);
    const auto bfp4 = structural::build_checked_bfp4_blkscale_value_codec(host_bitbsr);
    const auto bf16 = structural::build_checked_bf16_nnz_value_codec(host_bitbsr);
    const auto fp16 = structural::build_checked_fp16_nnz_value_codec(host_bitbsr);

    // Upload the x-independent codec payloads to device once.
    thrust::device_vector<signed char> d_int8(bfp8.int8_values.begin(), bfp8.int8_values.end());
    thrust::device_vector<std::uint16_t> d_scale(bfp8.block_scale_bf16.begin(),
                                                 bfp8.block_scale_bf16.end());
    thrust::device_vector<signed char> d_int8_o(outlier.int8_values.begin(),
                                                outlier.int8_values.end());
    thrust::device_vector<std::uint16_t> d_scale_o(outlier.block_scale_bf16.begin(),
                                                   outlier.block_scale_bf16.end());
    thrust::device_vector<std::int32_t> d_orows(outlier.outlier_rows.begin(),
                                                outlier.outlier_rows.end());
    thrust::device_vector<std::int32_t> d_ocols(outlier.outlier_cols.begin(),
                                                outlier.outlier_cols.end());
    thrust::device_vector<double> d_ovals(outlier.outlier_vals.begin(), outlier.outlier_vals.end());
    thrust::device_vector<std::uint8_t> d_int4(bfp4.packed_int4.begin(), bfp4.packed_int4.end());
    thrust::device_vector<std::uint16_t> d_scale4(bfp4.block_scale_bf16.begin(),
                                                  bfp4.block_scale_bf16.end());
    thrust::device_vector<std::uint16_t> d_bf16(bf16.values16.begin(), bf16.values16.end());
    thrust::device_vector<std::uint16_t> d_fp16(fp16.values16.begin(), fp16.values16.end());

    // bytes/nnz per codec (from the production payload_bytes()).
    const double bpn_bfp8 = static_cast<double>(bfp8.payload_bytes()) / nnz_d;
    const double bpn_outlier = static_cast<double>(outlier.payload_bytes()) / nnz_d;
    const double bpn_bfp4 = static_cast<double>(bfp4.payload_bytes()) / nnz_d;
    const double bpn_bf16 = static_cast<double>(bf16.payload_bytes()) / nnz_d;
    const double bpn_fp16 = static_cast<double>(fp16.payload_bytes()) / nnz_d;

    // Result accumulators (one per output row).
    CodecResult r_bfp8{"bfp8", "fp32", bpn_bfp8};
    CodecResult r_bfp8_q{"bfp8", "fp64", bpn_bfp8};  // quantization-only
    CodecResult r_outlier{"bfp8_outlier", "fp32", bpn_outlier};
    CodecResult r_bfp4{"bfp4", "fp32", bpn_bfp4};
    CodecResult r_bf16{"bf16", "fp32", bpn_bf16};
    CodecResult r_fp16{"fp16", "fp32", bpn_fp16};

    const auto ensemble = make_x_ensemble(csr.cols);

    thrust::device_vector<double> d_x;
    thrust::device_vector<double> d_y;
    std::vector<double> y_host(static_cast<std::size_t>(device.rows));

    auto copy_back = [&]() {
        thrust::copy(d_y.begin(), d_y.end(), y_host.begin());
        return y_host;
    };

    // Record y (already copied to y_host) into a codec result for this x. If the
    // decoded output is non-finite (e.g. fp16 OVERFLOWS on matrices whose values
    // exceed the fp16 max ~6.55e4 -- a real, faithful property of the production
    // fp16-no-scale codec), report both metrics as +inf (a single clean sentinel
    // meaning "codec failed") so nan never poisons the worst/mean aggregates.
    auto record_y = [&](CodecResult& r, const std::vector<double>& y0,
                        const std::vector<double>& y, XKind kind) {
        bool finite = true;
        for (double v : y) {
            if (!std::isfinite(v)) { finite = false; break; }
        }
        if (finite) {
            r.record(rel_maxrow(y0, y), rel_l2(y0, y), kind);
        } else {
            const double kInf = std::numeric_limits<double>::infinity();
            r.record(kInf, kInf, kind);
        }
    };

    for (const auto& xv : ensemble) {
        // fp64 reference: original (fp64) * x, double accumulate (the production
        // CSR fp64 reference, T=double => fp64 sum).
        const std::vector<double> y0 = structural::spmv_csr(csr, xv.x);
        d_x.assign(xv.x.begin(), xv.x.end());

        // --- bfp8 (production decode + fp32 accumulate, flag LB path) ---
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp8_blkscale_device(
            device, packed_meta, layout, d_int8, d_scale, d_x, d_y);
        record_y(r_bfp8, y0, copy_back(), xv.kind);

        // --- bfp8 fp64-accumulate (quantization-only; same encoded payload) ---
        record_y(r_bfp8_q, y0, bfp8_decode_fp64_accum(host_bitbsr, bfp8, xv.x), xv.kind);

        // --- bfp8 + outlier two-pass (production decode + fp32 accumulate) ---
        structural::spmv_bitbsr_gpu_8x4_bfp8_outlier_flag_load_balance_device(
            device, packed_meta, layout, d_int8_o, d_scale_o, d_orows, d_ocols, d_ovals,
            d_x, d_y);
        record_y(r_outlier, y0, copy_back(), xv.kind);

        // --- bfp4 (production decode + fp32 accumulate, flag LB path) ---
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_bfp4_blkscale_device(
            device, packed_meta, layout, d_int4, d_scale4, d_x, d_y);
        record_y(r_bfp4, y0, copy_back(), xv.kind);

        // --- bf16 (production per-nnz decode, fp32 accumulate) ---
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_device<
            /*Bf16=*/true>(device, packed_meta, layout, d_bf16, "bf16_nnz", d_x, d_y);
        record_y(r_bf16, y0, copy_back(), xv.kind);

        // --- fp16 (production per-nnz decode, fp32 accumulate) ---
        structural::spmv_bitbsr_gpu_8x4_col_bitmap_packed_flag_load_balance_round16_nnz_device<
            /*Bf16=*/false>(device, packed_meta, layout, d_fp16, "fp16_nnz", d_x, d_y);
        record_y(r_fp16, y0, copy_back(), xv.kind);
    }

    // ---- CSV output. Order: bfp8(fp32), bfp8(fp64-quant), bfp8_outlier, bfp4, bf16, fp16.
    const std::vector<const CodecResult*> rows = {
        &r_bfp8, &r_bfp8_q, &r_outlier, &r_bfp4, &r_bf16, &r_fp16};

    FILE* f = std::fopen(csv_path.c_str(), append ? "a" : "w");
    if (!f) {
        std::fprintf(stderr, "cannot open %s for writing\n", csv_path.c_str());
        return 1;
    }
    if (!append) {
        // rel_*_worst / rel_*_mean = full 17-vector ensemble (kept for reference).
        // rel_*_worst_rand   = worst over the RANDOM subset (uniform + normal) --
        //                      the headline figure uses these (well-conditioned).
        // rel_*_worst_struct = worst over the STRUCTURED subset (all-ones, legacy,
        //                      zero-mean fixed) -- a separately-reported
        //                      conditioning caveat (cancellation collapses ||y0||).
        std::fprintf(f,
                     "matrix,codec,accum,bytes_per_nnz,rel_maxrow_worst,rel_maxrow_mean,"
                     "rel_l2_worst,rel_l2_mean,rel_maxrow_worst_rand,rel_l2_worst_rand,"
                     "rel_maxrow_worst_struct,rel_l2_worst_struct,nnz,num_blocks\n");
    }
    for (const CodecResult* r : rows) {
        std::fprintf(f, "%s,%s,%s,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%lld,%lld\n",
                     matrix_name.c_str(), r->codec.c_str(), r->accum.c_str(),
                     r->bytes_per_nnz, r->acc.maxrow_worst, r->acc.maxrow_mean(),
                     r->acc.l2_worst, r->acc.l2_mean(), r->acc.maxrow_worst_rand,
                     r->acc.l2_worst_rand, r->acc.maxrow_worst_struct,
                     r->acc.l2_worst_struct, (long long)nnz, (long long)num_blocks);
    }
    std::fclose(f);

    // ---- Console summary (per matrix): headline numbers + robustness ratio +
    // decomposition. This is what the report table is built from.
    std::printf("== %-18s nnz=%lld blocks=%lld outliers=%lld(%.3f%%) | ensemble N=%d ==\n",
                matrix_name.c_str(), (long long)nnz, (long long)num_blocks,
                (long long)outlier.num_outliers,
                100.0 * static_cast<double>(outlier.num_outliers) / (nnz_d + 1e-30),
                (int)ensemble.size());
    const double ratio_bfp8 =
        (r_bfp8.maxrow_anchor > 0) ? r_bfp8.acc.maxrow_worst / r_bfp8.maxrow_anchor : 0.0;
    std::printf("  bfp8(fp32)  bytes/nnz=%.4f | maxrow anchor=%.4e worst=%.4e mean=%.4e (worst/anchor=%.2fx) | l2 worst=%.4e (rand-only worst=%.4e) mean=%.4e\n",
                r_bfp8.bytes_per_nnz, r_bfp8.maxrow_anchor, r_bfp8.acc.maxrow_worst,
                r_bfp8.acc.maxrow_mean(), ratio_bfp8, r_bfp8.acc.l2_worst,
                r_bfp8.acc.l2_worst_rand, r_bfp8.acc.l2_mean());
    std::printf("              maxrow per-input: legacy(1+i%%7*.1)=%.4e all_ones=%.4e | maxrow worst RAND=%.4e STRUCT=%.4e FULL=%.4e\n",
                r_bfp8.maxrow_legacy, r_bfp8.maxrow_allones, r_bfp8.acc.maxrow_worst_rand,
                r_bfp8.acc.maxrow_worst_struct, r_bfp8.acc.maxrow_worst);
    std::printf("  bfp8(fp64)  QUANT-ONLY     | maxrow anchor=%.4e worst=%.4e | l2 worst=%.4e  (accum-gap maxrow worst: %.3e, l2 worst: %.3e)\n",
                r_bfp8_q.maxrow_anchor, r_bfp8_q.acc.maxrow_worst, r_bfp8_q.acc.l2_worst,
                r_bfp8.acc.maxrow_worst - r_bfp8_q.acc.maxrow_worst,
                r_bfp8.acc.l2_worst - r_bfp8_q.acc.l2_worst);
    std::printf("  bfp8_outlier bytes/nnz=%.4f | maxrow anchor=%.4e worst=%.4e mean=%.4e | l2 worst=%.4e\n",
                r_outlier.bytes_per_nnz, r_outlier.maxrow_anchor, r_outlier.acc.maxrow_worst,
                r_outlier.acc.maxrow_mean(), r_outlier.acc.l2_worst);
    std::printf("  bfp4         bytes/nnz=%.4f | maxrow anchor=%.4e worst=%.4e mean=%.4e | l2 worst=%.4e\n",
                r_bfp4.bytes_per_nnz, r_bfp4.maxrow_anchor, r_bfp4.acc.maxrow_worst,
                r_bfp4.acc.maxrow_mean(), r_bfp4.acc.l2_worst);
    std::printf("  bf16         bytes/nnz=%.4f | maxrow anchor=%.4e worst=%.4e mean=%.4e | l2 worst=%.4e\n",
                r_bf16.bytes_per_nnz, r_bf16.maxrow_anchor, r_bf16.acc.maxrow_worst,
                r_bf16.acc.maxrow_mean(), r_bf16.acc.l2_worst);
    std::printf("  fp16         bytes/nnz=%.4f | maxrow anchor=%.4e worst=%.4e mean=%.4e | l2 worst=%.4e\n",
                r_fp16.bytes_per_nnz, r_fp16.maxrow_anchor, r_fp16.acc.maxrow_worst,
                r_fp16.acc.maxrow_mean(), r_fp16.acc.l2_worst);
    return 0;
}
