#!/usr/bin/env python3
# Port CSR5_cuda (2015, CUDA 6.5) to modern CUDA: disable its double __shfl_* and
# atomicAdd polyfills, which now collide with native built-ins (sm_60+ / CUDA 9+).
import pathlib

f = pathlib.Path("detail/cuda/utils_cuda.h")
s = f.read_text()
orig = s

# 1) double __shfl_down/up/xor polyfills: guard is `#if __CUDA_ARCH__ <= 300`, but
#    __CUDA_ARCH__ is undefined (=>0) in the host pass, so it compiled there and
#    collided. Require __CUDA_ARCH__ to be defined AND <=300 (true sm_3x only).
s = s.replace("#if __CUDA_ARCH__ <= 300",
              "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 300)")

# 2) double atomicAdd polyfill is unguarded; native since sm_60. Wrap the function.
start_key = "__forceinline__ __device__\nstatic double atomicAdd(double *addr, double val)"
start = s.index(start_key)
brace_open = s.index("{", start)
depth, i = 0, brace_open
while i < len(s):
    if s[i] == "{":
        depth += 1
    elif s[i] == "}":
        depth -= 1
        if depth == 0:
            break
    i += 1
end = i + 1
block = s[start:end]
s = (s[:start]
     + "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)\n"
     + block + "\n#endif" + s[end:])

assert s != orig, "no change applied"
f.write_text(s)
print("patched utils_cuda.h: shfl guard + atomicAdd guard OK")
