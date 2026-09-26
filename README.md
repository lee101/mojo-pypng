# mojo-pypng

`mojo-pypng` is a standalone PNG encoder and decoder with a deliberately
limited, pypng-compatible Python API and Mojo implementations of the
byte-oriented hot paths. It imports as `png`, so code using the covered subset
can switch without changing imports or call signatures.

The project does not wrap pypng at runtime. PNG chunk handling and DEFLATE
integration are implemented in this repository; pypng 0.20220715.0 is installed
only in the development environment for parity tests and benchmarks.

## Coverage

The tested API subset is:

- `Writer`, including `write`, `write_array`, `write_packed`,
  `write_passes`, and preamble/chunk generation
- `Reader`, including `read`, `read_flat`, `chunks`, `palette`, `asDirect`,
  `asRGB`, `asRGBA`, `asRGB8`, and `asRGBA8`
- `Image`, `from_array`/`fromarray`, `write_chunk`, and `write_chunks`
- non-interlaced greyscale, greyscale-alpha, RGB, RGBA, and indexed PNGs
- all legal 1-, 2-, 4-, 8-, and 16-bit combinations for those color types
- adaptive None/Sub/Up/Average/Paeth filtering, filter reversal, packed
  sample conversion, CRC validation, palettes, `tRNS`, `bKGD`, `gAMA`, and
  `pHYs`

This is not a drop-in implementation of all of upstream pypng. It does not
cover Adam7 interlacing, automatic promotion of nonstandard or per-channel
source depths through `sBIT`, PNM helpers, command-line tools, or incremental
row-at-a-time compression. Input rows are materialized in contiguous memory
before entering the Mojo kernels. APIs not named above should be treated as
unsupported.

## Install and build

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-pypng.so`. The Pixi environment adds `python/`
to `PYTHONPATH`, so `pixi run python` imports this repository's `png` package.

## Usage

```python
import io
import numpy as np
import png

pixels = np.array([
    [255, 0, 0, 0, 255, 0],
    [0, 0, 255, 255, 255, 255],
], dtype=np.uint8)

stream = io.BytesIO()
png.Writer(width=2, height=2, greyscale=False).write(stream, pixels)

width, height, rows, info = png.Reader(bytes=stream.getvalue()).read()
decoded = np.array([list(row) for row in rows], dtype=np.uint8)
assert (width, height) == (2, 2)
assert np.array_equal(decoded, pixels)
assert info["planes"] == 3
```

CPU execution is the default. Adaptive filtering can be requested explicitly
on the GPU with `png.Writer(..., device="gpu")`. An explicit GPU request raises
`RuntimeError` if the accelerator is unavailable, lacks the required free
memory, exceeds the device-allocation limit, or cannot run the kernel.

Save that example as `example.py` and run it with:

```bash
pixi run python example.py
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Times are the median of three runs. Speedup is
pypng time divided by mojo-pypng time.

| Workload | mojo-pypng | pypng | Speedup | Notes |
|---|---:|---:|---:|---|
| Encode RGB8 2048x1024 | 49.95 ms | 144.31 ms | 2.89x | adaptive 26 KiB; pypng none 3113 KiB |
| Encode RGB8 2048x1024 GPU | 99.35 ms | 144.31 ms | 1.45x | explicit device=gpu; includes transfers |
| Decode adaptive RGB8 | 49.19 ms | 3149.72 ms | 64.04x | same mojo-pypng file |
| Decode filter-none RGB8 | 39.61 ms | 35.96 ms | 0.91x | same pypng file |
| Encode grayscale 2-bit | 19.54 ms | 298.78 ms | 15.29x | 2048x1024, includes bit packing |

The adaptive filter scorer and emitter use SIMD with scalar remainder loops.
Large inputs distribute rows over physical CPU cores; smaller inputs remain
serial. Filter-none decoding avoids materializing an unfiltered copy and
exposes rows as zero-copy NumPy views. Decompression writes directly into its
NumPy destination through zlib's native ABI.

Filtering has enough integer arithmetic per byte to justify an optional GPU
implementation, but transfers and context setup make it slower than the
parallel CPU path at this image size. It remains explicit rather than
automatic. The benchmark performs one bounded GPU measurement only when
`nvidia-smi` reports at least 4000 MiB free.

## Toolchain

The kernels are written against Mojo `1.1.0.dev2026081105` / `max`
`26.6.0.dev2026081105` as pinned in `pixi.toml`. Two primitives moved out of
`std` in this release and are imported from `max` instead: CPU row parallelism
is `max.algorithm.parallelize`, and the GPU host driver is
`max.gpu.host.DeviceContext` (`std.gpu.host` no longer defines it).
`std.runtime.initialize_runtime()` must be called before the first
`parallelize`. Device kernels must take fixed-width arguments — the entry in
`filter_rows_gpu_kernel` takes `Int32` and widens to `Int` in the body, because
`Int` does not satisfy `DevicePassable`. No kernel had to be reduced to serial.

`MOJO_NOTES.md` records the rest of the dialect changes the port absorbed:
`fn` is now `def`, `UnsafePointer` is now `Pointer`, `int()`/`float()` are now
`Int()`/`Float64()`, and `simd_width_of` is imported from `std.sys.info`.

## How it works

Python owns parsing, validation, chunk generation, CRC checks, and calls to
native `zlib`. NumPy provides contiguous input and output buffers. Their
addresses and exact byte lengths cross the C ABI into one Mojo compilation
unit. Mojo validates non-null pointers, lengths, dimensions, and element
widths before reconstructing mutable byte or word pointers; it never owns or
frees host memory. The Python locals retain each NumPy allocation for the
entire synchronous foreign call.

The encoder packs sub-byte or big-endian 16-bit samples, scores all five PNG
filters per scanline, and writes the lowest-cost residual. The decoder applies
the inverse predictor directly into a contiguous output buffer, then unpacks
samples into pypng-compatible row iterables.

Run all parity tests with:

```bash
pixi run build
pixi run test
```
