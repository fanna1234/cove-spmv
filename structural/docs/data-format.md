# Data Format Contract

This stage is limited to loading and format conversion. SpMV kernels and
performance kernels are downstream consumers, not part of the default format
pipeline.

## MatrixMarket Input

Supported header shape:

```text
%%MatrixMarket matrix coordinate <field> <symmetry>
```

Supported scalar fields:

```text
real
integer
pattern
```

`pattern` carries no value column; the loader materializes every stored entry
with value `1`.

Supported symmetry modes:

```text
general
symmetric
skew-symmetric
```

`symmetric` expands an off-diagonal entry `(i, j, v)` into both `(i, j, v)` and
`(j, i, v)`. `skew-symmetric` expands the mirror as `(j, i, -v)`.

`complex` and `hermitian` are rejected by the current scalar loader because they
need a complex value type and conjugation-aware mirroring.

## Normalized CSR

The loader returns:

```cpp
MatrixMarketLoadResult<T> {
  MatrixMarketInfo info;
  CsrMatrix<T> matrix;
  Offset stored_entries;
  Offset expanded_entries;
  Offset coalesced_duplicates;
}
```

CSR is normalized after loading:

```text
row_ptr[rows + 1]
col_idx[nnz]
values[nnz]
```

Rows are sorted by column. Duplicate entries in the same row and column are
summed, and `coalesced_duplicates` records how many expanded entries collapsed.

## Value Profile

The loader does not compress values yet, but it records simple value-shape
signals:

```text
explicit_zero_count
all_values_one
binary_zero_one
sign_only
unique_values_exact_cap16
unique_values_overflow_cap16
min / max
```

These signals are dataset filters for later value codecs. For example, a
MatrixMarket `pattern` matrix is structurally binary by header, while a `real`
matrix can still be detected as binary if every explicit value is `0` or `1`.

## BitBSR Conversion

The first converted format is BitBSR:

```text
block_row_ptr[num_block_rows + 1]
block_col_idx[num_blocks]
bitmap_words[num_blocks * words_per_block]
block_val_ptr[num_blocks + 1]
values[nnz]
```

`bitmap_words` uses 32-bit words. The current 8x4 BitBSR format uses one word
for one 32-position block. The 8x8 candidate uses two words for one 64-position
block.

`block_val_ptr` is part of the format contract. It gives each block an O(1)
base pointer into `values`; without it, a kernel would need a prefix popcount
over previous blocks.

The conversion is lossless at the structural layer. It changes only how
positions are represented; values are copied unchanged and kept in bitmap scan
order inside each block.
