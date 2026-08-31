#include "structural/gpu_convert.cuh"
#include "structural/matrix_market.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct Options {
    std::string matrix_path;
    int block_rows = 8;
    int block_cols = 4;
    std::string mode = "warp_direct";
};

void print_usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " MATRIX.mtx [--br 8] [--bc 4] [--mode warp_direct]"
              << " [--device-only] [--no-cpu]\n"
              << "default: --mode warp_direct --device-only\n";
}

void reject_retired_host_materialization() {
    throw std::runtime_error(
        "high-performance conversion path is device-resident only; "
        "use tests/helpers for CPU materialization");
}

void reject_retired_mode() {
    throw std::runtime_error("only warp_direct 8x4 is exposed for high-performance conversion");
}

Options parse_options(int argc, char** argv) {
    Options opts;
    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(2);
    }
    opts.matrix_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--br" && i + 1 < argc) {
            opts.block_rows = std::stoi(argv[++i]);
        } else if (arg == "--bc" && i + 1 < argc) {
            opts.block_cols = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            opts.mode = argv[++i];
            if (opts.mode == "warp-direct" || opts.mode == "warp") {
                opts.mode = "warp_direct";
            } else if (opts.mode == "global-sort" || opts.mode == "global" ||
                       opts.mode == "global_sort" || opts.mode == "warp-direct-8x8" ||
                       opts.mode == "warp_8x8" || opts.mode == "warp_direct_8x8") {
                reject_retired_mode();
            } else if (opts.mode == "direct" || opts.mode == "segmented") {
                throw std::runtime_error("unsupported GPU conversion mode: " + opts.mode);
            } else if (opts.mode != "warp_direct") {
                throw std::runtime_error("unknown GPU conversion mode: " + opts.mode);
            }
        } else if (arg == "--verify-cpu" || arg == "--host-output" ||
                   arg == "--materialize") {
            reject_retired_host_materialization();
        } else if (arg == "--device-only" || arg == "--no-materialize" ||
                   arg == "--no-cpu") {
            continue;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (opts.block_rows != 8 || opts.block_cols != 4 || opts.mode != "warp_direct") {
        reject_retired_mode();
    }
    return opts;
}

template <typename F>
auto time_call(F&& fn, double& ms) {
    const auto t0 = std::chrono::steady_clock::now();
    auto result = fn();
    const auto t1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

template <typename T>
double device_bitbsr_position_bits_per_nnz(
    const structural::CsrMatrix<T>& csr,
    const structural::DeviceBitBsrMatrix<T>& bitbsr) {
    if (csr.nnz() == 0) {
        return 0.0;
    }
    const long long total_position_bits =
        static_cast<long long>(bitbsr.num_block_rows + 1) * 32 +
        static_cast<long long>(bitbsr.num_block_rows + 1) * 32 +
        static_cast<long long>(bitbsr.num_blocks) * 32 +
        static_cast<long long>(bitbsr.num_blocks) * bitbsr.words_per_block *
            structural::kBitmapWordBits;
    return static_cast<double>(total_position_bits) / static_cast<double>(csr.nnz());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_options(argc, argv);

        double load_ms = 0.0;
        double cuda_context_ms = 0.0;
        double gpu_convert_ms = 0.0;
        float gpu_convert_event_ms = 0.0f;

        const auto loaded = time_call(
            [&] { return structural::read_matrix_market_with_info<double>(opts.matrix_path); },
            load_ms);
        const int cuda_context_status = time_call(
            [&] {
                structural::cuda_check(cudaFree(nullptr), "cudaFree(nullptr)", __FILE__, __LINE__);
                return 0;
            },
            cuda_context_ms);
        (void)cuda_context_status;

        structural::GpuBitBsrWorkspace<double> workspace;
        structural::DeviceBitBsrMatrix<double> gpu;
        (void)time_call(
            [&] {
                structural::convert_to_bitbsr_gpu_warp_direct_8x4_device(
                    loaded.matrix, workspace, gpu, &gpu_convert_event_ms);
                return 0;
            },
            gpu_convert_ms);

        std::cout << "matrix: " << opts.matrix_path << "\n";
        std::cout << "matrix_market_field: " << structural::to_string(loaded.info.field) << "\n";
        std::cout << "matrix_market_symmetry: " << structural::to_string(loaded.info.symmetry)
                  << "\n";
        std::cout << "shape: " << loaded.matrix.rows << " x " << loaded.matrix.cols << "\n";
        std::cout << "nnz: " << loaded.matrix.nnz() << "\n";
        std::cout << "block: " << opts.block_rows << " x " << opts.block_cols << "\n";
        std::cout << "gpu_convert_mode: " << opts.mode << "\n";
        std::cout << "gpu_output_residency: device\n";
        std::cout << "gpu_num_blocks: " << gpu.num_blocks << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "gpu_bitbsr_position_bits_per_nnz: "
                  << device_bitbsr_position_bits_per_nnz(loaded.matrix, gpu) << "\n";
        std::cout << "verify_cpu: SKIP_DEVICE_RESIDENT\n";
        std::cout << "timing_ms: load=" << load_ms
                  << ", cuda_context=" << cuda_context_ms
                  << ", cpu_convert=0"
                  << ", gpu_convert_host=" << gpu_convert_ms
                  << ", gpu_convert_event=" << gpu_convert_event_ms
                  << ", total=" << (load_ms + cuda_context_ms + gpu_convert_ms)
                  << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
