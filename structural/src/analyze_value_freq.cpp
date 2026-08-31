// Value-FREQUENCY concentration analysis: for a matrix, histogram distinct values
// (exact bit-key) and report what fraction of nnz the top-K most-FREQUENT distinct
// values cover. Tests the "codebook-the-frequent-head + quantize-the-tail" hypothesis:
// if top-256 covers most nnz -> frequency-concentrated -> codebook head wins.
//
// Build (host-only): g++ -O2 -std=c++17 -Istructural/include \
//     -o structural/build/analyze_value_freq structural/src/analyze_value_freq.cpp
// CSV line: name,nnz,cardinality,cov_top1,cov_top16,cov_top64,cov_top256,cov_top1024

#include "structural/matrix_market.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s matrix.mtx name\n", argv[0]);
        return 2;
    }
    structural::CsrMatrix<double> csr;
    try {
        csr = structural::read_matrix_market<double>(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load fail %s: %s\n", argv[2], e.what());
        return 1;
    }
    const auto& vals = csr.values;
    const long long nnz = static_cast<long long>(vals.size());
    std::unordered_map<unsigned long long, long long> hist;
    hist.reserve(vals.size() / 2 + 16);
    for (double v : vals) {
        unsigned long long k;
        std::memcpy(&k, &v, sizeof(k));
        ++hist[k];
    }
    const long long card = static_cast<long long>(hist.size());
    std::vector<long long> counts;
    counts.reserve(hist.size());
    for (const auto& p : hist) counts.push_back(p.second);
    std::sort(counts.begin(), counts.end(), std::greater<long long>());
    auto cov = [&](int k) -> double {
        long long s = 0;
        const int lim = std::min(static_cast<int>(counts.size()), k);
        for (int i = 0; i < lim; ++i) s += counts[i];
        return nnz ? 100.0 * static_cast<double>(s) / static_cast<double>(nnz) : 0.0;
    };
    std::printf("%s,%lld,%lld,%.2f,%.2f,%.2f,%.2f,%.2f\n", argv[2], nnz, card, cov(1), cov(16),
                cov(64), cov(256), cov(1024));
    return 0;
}
