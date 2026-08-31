#include "structural/convert.hpp"
#include "structural/matrix_market.hpp"
#include "structural/spmv.hpp"
#include "structural/stats.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string matrix_path;
    int block_rows = 8;
    int block_cols = 4;
};

void print_usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " MATRIX.mtx [--br N] [--bc N]\n";
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
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    return opts;
}

std::vector<double> make_deterministic_x(int n) {
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>((i % 17) + 1) / 17.0;
    }
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_options(argc, argv);
        const auto csr = structural::read_matrix_market<double>(opts.matrix_path);
        const auto bitbsr = structural::convert_to_bitbsr(csr, opts.block_rows, opts.block_cols);
        const auto x = make_deterministic_x(csr.cols);
        const auto y_csr = structural::spmv_csr(csr, x);
        const auto y_bitbsr = structural::spmv_bitbsr(bitbsr, x);
        const auto error = structural::compare_vectors(y_csr, y_bitbsr);
        const auto stats = structural::storage_stats(csr, bitbsr);

        structural::print_human_stats(std::cout, opts.matrix_path, stats);
        std::cout << std::scientific << std::setprecision(12);
        std::cout << "max_abs_error: " << error.max_abs << "\n";
        std::cout << "max_rel_error: " << error.max_rel << "\n";
        std::cout << "status: " << (error.max_abs == 0.0 ? "PASS" : "CHECK") << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
