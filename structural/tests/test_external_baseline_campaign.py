#!/usr/bin/env python3
import importlib.util
import os
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_external_baseline_campaign.py"
spec = importlib.util.spec_from_file_location("run_canonical25_baselines", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_parse_dasp_double_stdout():
    stdout = "\n".join(
        [
            "input matrix A: ( 1083, 1083 ) nnz = 18437",
            "DASP(Double):  0.123 ms  0.045 ms  819.422 GFlop/s",
            "cusparse:  0.010 ms  0.050 ms  738.300 GFlop/s",
            "compute succeed!",
        ]
    )

    row = campaign.parse_dasp_stdout(
        "bcsstk09.mtx", "dasp_double", "0", stdout, "", 0, 1.25, "cmd"
    )

    assert row["status"] == "ok"
    assert row["baseline"] == "dasp_double"
    assert row["precision_group"] == "double"
    assert row["rows"] == "1083"
    assert row["nnz"] == "18437"
    assert row["pre_ms"] == "0.123"
    assert row["spmv_ms"] == "0.045"
    assert row["gflops"] == "819.422"


def test_parse_dasp_nonzero_return_keeps_exit_status():
    row = campaign.parse_dasp_stdout(
        "bad.mtx",
        "dasp_half",
        "0",
        "input matrix A: ( 10, 10 ) nnz = 20",
        "segmentation fault",
        139,
        0.25,
        "cmd",
    )

    assert row["status"] == "fail:139"


def test_parse_packsell_elapsed_seconds_to_per_spmv_ms():
    stdout = "bcsstk09,1083,1083,18437,20000,pack16,256,0,0.003,12.291,0.0004\n"

    row = campaign.parse_packsell_stdout(
        "bcsstk09.mtx", "packsell_pack16", "1", stdout, "", 0, 0.5, "cmd", iters=30
    )

    assert row["status"] == "ok"
    assert row["precision_group"] == "half"
    assert row["rows"] == "1083"
    assert row["spmv_ms"] == "0.100000"
    assert row["avg_ms"] == "0.100000"
    assert row["gflops"] == "12.291"
    assert row["error_metric"] == "0.0004"


def test_parse_spaden_bitmap_stdout():
    stdout = "\n".join(
        [
            "  Shape:       1083 x 1083",
            "  NNZ:         18437 (1.5720% density)",
            "spmv_warp16: 0.0700 ms 526.7000 Gflops",
            "  Conversion time: 1.25 ms",
            "    Bitmap: 0.30 MB (55.0% of CSR)",
            "[Bitmap Spaden]   SpMV:   0.0400 ms, 921.8500 Gflops",
            "  Relative error:        1.000000e-07",
            "PASSED",
        ]
    )

    row = campaign.parse_spaden_stdout(
        "bcsstk09.mtx", "spaden_bitmap_float", "2", stdout, "", 0, 2.0, "cmd"
    )

    assert row["status"] == "ok"
    assert row["precision_group"] == "float"
    assert row["rows"] == "1083"
    assert row["nnz"] == "18437"
    assert row["spmv_ms"] == "0.0400"
    assert row["ref_baseline"] == "spaden_csr_warp16_float"
    assert row["ref_spmv_ms"] == "0.0700"
    assert row["conversion_ms"] == "1.25"
    assert row["storage_ratio"] == "55.0"


def test_balanced_shards_keep_large_matrices_apart():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = []
        for name, size in [("a.mtx", 100), ("b.mtx", 90), ("c.mtx", 10), ("d.mtx", 1)]:
            path = root / name
            path.write_bytes(b"x" * size)
            paths.append(path)

        shards = campaign.balance_shards(paths, 2)

    names = [[path.name for path in shard] for shard in shards]
    assert names == [["a.mtx", "d.mtx"], ["b.mtx", "c.mtx"]]


def test_matrix_id_preserves_group_path_for_duplicate_basenames():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        matrix_a = root / "G1" / "M" / "dup.mtx"
        matrix_b = root / "G2" / "M" / "dup.mtx"
        matrix_a.parent.mkdir(parents=True)
        matrix_b.parent.mkdir(parents=True)
        matrix_a.write_text("a")
        matrix_b.write_text("b")

        assert campaign.matrix_id(matrix_a, root) == "G1/M/dup.mtx"
        assert campaign.matrix_id(matrix_b, root) == "G2/M/dup.mtx"


def test_cusparse_float_command_uses_float_precision():
    class Args:
        repo_root = "/repo"
        warmup = 2
        iters = 10

    cmd = campaign.build_command(Args(), "cusparse_float", Path("/data/G/M/a.mtx"))

    assert "structural/build/struct_gpu_spmv" in cmd[0]
    assert cmd[cmd.index("--precision") + 1] == "float"
    assert cmd[cmd.index("--layout") + 1] == "cusparse_csr"


def test_parse_cusparse_float_stdout():
    stdout = "\n".join(
        [
            "shape: 1083 x 1083",
            "nnz: 18437",
            "timing_ms: gpu_spmv_avg_event=0.045, gpu_spmv_min_event=0.040, cusparse_preprocess_event=0.010",
        ]
    )

    row = campaign.parse_cusparse_stdout(
        "bcsstk09.mtx", "cusparse_float", "0", stdout, "", 0, 1.0, "cmd"
    )

    assert row["status"] == "ok"
    assert row["precision_group"] == "float"
    assert row["rows"] == "1083"
    assert row["spmv_ms"] == "0.045"


def test_run_command_respects_cuda_home():
    old_cuda_home = os.environ.get("CUDA_HOME")
    old_cuda_root = os.environ.get("CUDA_ROOT")
    try:
        os.environ["CUDA_HOME"] = "/opt/cuda-test"
        os.environ.pop("CUDA_ROOT", None)
        rc, stdout, stderr, _elapsed = campaign.run_command(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "print(os.environ['CUDA_VISIBLE_DEVICES']); "
                    "print(os.environ['PATH'].split(':')[0]); "
                    "print(os.environ['LD_LIBRARY_PATH'].split(':')[0])"
                ),
            ],
            gpu="7",
            timeout_s=5,
        )
    finally:
        if old_cuda_home is None:
            os.environ.pop("CUDA_HOME", None)
        else:
            os.environ["CUDA_HOME"] = old_cuda_home
        if old_cuda_root is None:
            os.environ.pop("CUDA_ROOT", None)
        else:
            os.environ["CUDA_ROOT"] = old_cuda_root

    assert rc == 0
    assert stderr == ""
    assert stdout.splitlines() == ["7", "/opt/cuda-test/bin", "/opt/cuda-test/lib64"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
