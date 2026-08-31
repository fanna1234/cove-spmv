NVCC ?= nvcc
ARCH ?= -gencode arch=compute_120a,code=sm_120a

STRUCTURAL_BUILD_DIR := structural/build
STRUCTURAL_NVCC_FLAGS ?= -O3 -std=c++17 $(ARCH) -Istructural/include
STRUCTURAL_CUSPARSE_LIBS ?= -lcusparse
STRUCTURAL_GPU_TEST_BIN := $(STRUCTURAL_BUILD_DIR)/struct_gpu_tests
STRUCTURAL_GPU_CONVERT_BIN := $(STRUCTURAL_BUILD_DIR)/struct_gpu_convert
STRUCTURAL_GPU_SPMV_BIN := $(STRUCTURAL_BUILD_DIR)/struct_gpu_spmv
STRUCTURAL_GPU_SPMV_OPTIONS_TEST_BIN := $(STRUCTURAL_BUILD_DIR)/struct_spmv_cli_options_tests
STRUCTURAL_GPU_CSR_BITBSR_SPMV_BIN := $(STRUCTURAL_BUILD_DIR)/struct_gpu_csr_bitbsr_spmv
STRUCTURAL_GPU_CLI_TEST := structural/tests/test_gpu_convert_cli.sh
STRUCTURAL_GPU_SPMV_CLI_TEST := structural/tests/test_gpu_spmv_cli.sh
STRUCTURAL_GPU_SPMV_FAMILY_SWEEP_TEST := structural/tests/test_gpu_spmv_family_sweep.py
STRUCTURAL_GPU_CSR_BITBSR_SPMV_CLI_TEST := structural/tests/test_gpu_csr_bitbsr_spmv_cli.sh
all: structural-build

structural-configure:
	cmake -S structural -B $(STRUCTURAL_BUILD_DIR)

structural-build: structural-configure
	cmake --build $(STRUCTURAL_BUILD_DIR) -j

structural-test: structural-build
	ctest --test-dir $(STRUCTURAL_BUILD_DIR) --output-on-failure

STRUCTURAL_GPU_HEADERS := \
	structural/include/structural/bitbsr.hpp \
	structural/include/structural/convert.hpp \
	structural/include/structural/csr.hpp \
	structural/include/structural/csr_bitbsr_split.hpp \
	structural/include/structural/gpu_convert.cuh \
	structural/include/structural/gpu_spmv.cuh \
	structural/include/structural/gpu_value_codecs.cuh \
	structural/include/structural/load_balance_policy.hpp \
	structural/include/structural/operators/bitbsr_spmv_8x4.cuh \
	structural/include/structural/operators/bitbsr_spmv_8x4_common.cuh \
	structural/include/structural/operators/bitbsr_spmv_8x4_dispatch.cuh \
	structural/include/structural/operators/bitbsr_spmv_8x4_lossless.cuh \
	structural/include/structural/operators/bitbsr_spmv_8x4_lossy.cuh \
	structural/include/structural/operators/bitbsr_spmv_8x4_raw.cuh \
	structural/include/structural/spmv.hpp \
	structural/include/structural/spmv_cli_options.hpp \
	structural/include/structural/spmv_operator_plan.hpp \
	structural/include/structural/types.hpp \
	structural/include/structural/value_compression.hpp

$(STRUCTURAL_GPU_TEST_BIN): structural/tests/test_gpu_convert.cu $(STRUCTURAL_GPU_HEADERS) | structural-build
	$(NVCC) $(STRUCTURAL_NVCC_FLAGS) -o $@ $<

structural-gpu-test: $(STRUCTURAL_GPU_TEST_BIN) structural-gpu-cli-test structural-gpu-spmv-cli-test
	$(STRUCTURAL_GPU_TEST_BIN)

$(STRUCTURAL_GPU_CONVERT_BIN): structural/src/main_gpu_convert.cu $(STRUCTURAL_GPU_HEADERS) structural/include/structural/matrix_market.hpp structural/include/structural/stats.hpp | structural-build
	$(NVCC) $(STRUCTURAL_NVCC_FLAGS) -o $@ $<

structural-gpu-convert: $(STRUCTURAL_GPU_CONVERT_BIN)

$(STRUCTURAL_GPU_SPMV_BIN): structural/src/main_gpu_spmv.cu $(STRUCTURAL_GPU_HEADERS) structural/include/structural/matrix_market.hpp | structural-build
	$(NVCC) $(STRUCTURAL_NVCC_FLAGS) -o $@ $< $(STRUCTURAL_CUSPARSE_LIBS)

structural-gpu-spmv: $(STRUCTURAL_GPU_SPMV_BIN)

$(STRUCTURAL_GPU_SPMV_OPTIONS_TEST_BIN): structural/tests/test_spmv_cli_options.cu $(STRUCTURAL_GPU_HEADERS) | structural-build
	$(NVCC) $(STRUCTURAL_NVCC_FLAGS) -o $@ $<

structural-gpu-spmv-options-test: $(STRUCTURAL_GPU_SPMV_OPTIONS_TEST_BIN)
	$(STRUCTURAL_GPU_SPMV_OPTIONS_TEST_BIN)

$(STRUCTURAL_GPU_CSR_BITBSR_SPMV_BIN): structural/src/main_gpu_csr_bitbsr_spmv.cu $(STRUCTURAL_GPU_HEADERS) structural/include/structural/matrix_market.hpp | structural-build
	$(NVCC) $(STRUCTURAL_NVCC_FLAGS) -o $@ $< $(STRUCTURAL_CUSPARSE_LIBS)

structural-gpu-csr-bitbsr-spmv: $(STRUCTURAL_GPU_CSR_BITBSR_SPMV_BIN)

structural-gpu-cli-test: structural-gpu-convert
	bash $(STRUCTURAL_GPU_CLI_TEST)

structural-gpu-spmv-cli-test: structural-gpu-spmv structural-gpu-spmv-options-test
	python3 $(STRUCTURAL_GPU_SPMV_FAMILY_SWEEP_TEST)
	bash $(STRUCTURAL_GPU_SPMV_CLI_TEST)

structural-gpu-csr-bitbsr-spmv-cli-test: structural-gpu-csr-bitbsr-spmv
	bash $(STRUCTURAL_GPU_CSR_BITBSR_SPMV_CLI_TEST)

clean:
	rm -f $(STRUCTURAL_GPU_TEST_BIN) $(STRUCTURAL_GPU_CONVERT_BIN) $(STRUCTURAL_GPU_SPMV_BIN) $(STRUCTURAL_GPU_SPMV_OPTIONS_TEST_BIN) $(STRUCTURAL_GPU_CSR_BITBSR_SPMV_BIN)
	rm -rf structural/scripts/__pycache__
	rm -f structural/results/*.log

distclean: clean
	rm -rf $(STRUCTURAL_BUILD_DIR) bin

.PHONY: all structural-configure structural-build structural-test structural-gpu-test structural-gpu-convert structural-gpu-spmv structural-gpu-spmv-options-test structural-gpu-csr-bitbsr-spmv structural-gpu-cli-test structural-gpu-spmv-cli-test structural-gpu-csr-bitbsr-spmv-cli-test clean distclean
