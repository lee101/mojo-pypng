"""Benchmarks against pypng on identical PNG data."""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import io
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

import png


def load_upstream():
    distribution = importlib.metadata.distribution("pypng")
    source = Path(distribution.locate_file("png.py"))
    spec = importlib.util.spec_from_file_location("_benchmark_pypng", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def measured(function, repeats=3):
    timings = []
    result = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def encode(module, pixels, **options):
    stream = io.BytesIO()
    module.Writer(**options).write(stream, pixels)
    return stream.getvalue()


def decode(module, data):
    rows = module.Reader(bytes=data).read()[2]
    return b"".join(map(bytes, rows))


def cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def gpu_memory_free_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return min(int(line.strip()) for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def result_row(name, mojo_seconds, upstream_seconds, note):
    speedup = upstream_seconds / mojo_seconds
    return (
        name,
        f"{mojo_seconds * 1000:.2f} ms",
        f"{upstream_seconds * 1000:.2f} ms",
        f"{speedup:.2f}x",
        note,
    )


def main():
    upstream = load_upstream()
    width, height = 2048, 1024
    x = np.arange(width, dtype=np.uint16)
    y = np.arange(height, dtype=np.uint16)[:, None]
    red = ((x + y) & 255).astype(np.uint8)
    green = ((2 * x + y // 2) & 255).astype(np.uint8)
    blue = ((x // 4 + 3 * y) & 255).astype(np.uint8)
    rgb = np.stack((red, green, blue), axis=2).reshape(height, width * 3)
    options = dict(
        width=width, height=height, greyscale=False, bitdepth=8, compression=6
    )

    mojo_encode, mojo_data = measured(lambda: encode(png, rgb, **options))
    upstream_encode, upstream_data = measured(
        lambda: encode(upstream, rgb, **options)
    )
    free_mib = gpu_memory_free_mib()
    gpu_encode = None
    if free_mib is not None and free_mib >= 4000:
        gpu_options = dict(options, device="gpu")
        gpu_encode, gpu_data = measured(lambda: encode(png, rgb, **gpu_options))
        assert gpu_data == mojo_data

    mojo_decode_adaptive, mojo_values = measured(
        lambda: decode(png, mojo_data)
    )
    upstream_decode_adaptive, upstream_values = measured(
        lambda: decode(upstream, mojo_data)
    )
    assert bytes(mojo_values) == bytes(upstream_values)

    mojo_decode_none, mojo_values = measured(
        lambda: decode(png, upstream_data)
    )
    upstream_decode_none, upstream_values = measured(
        lambda: decode(upstream, upstream_data)
    )
    assert bytes(mojo_values) == bytes(upstream_values)

    packed_pixels = (
        (np.arange(width, dtype=np.uint16)[None, :] // 17)
        + np.arange(height, dtype=np.uint16)[:, None]
    ).astype(np.uint8) & 3
    packed_options = dict(
        width=width, height=height, greyscale=True, bitdepth=2, compression=6
    )
    mojo_pack, mojo_packed_data = measured(
        lambda: encode(png, packed_pixels, **packed_options)
    )
    upstream_pack, upstream_packed_data = measured(
        lambda: encode(upstream, packed_pixels, **packed_options)
    )
    assert bytes(decode(png, mojo_packed_data)) == bytes(
        decode(upstream, upstream_packed_data)
    )

    rows = [
        result_row(
            "Encode RGB8 2048x1024",
            mojo_encode,
            upstream_encode,
            f"adaptive {len(mojo_data) / 1024:.0f} KiB; pypng none {len(upstream_data) / 1024:.0f} KiB",
        ),
        result_row(
            "Decode adaptive RGB8",
            mojo_decode_adaptive,
            upstream_decode_adaptive,
            "same mojo-pypng file",
        ),
        result_row(
            "Decode filter-none RGB8",
            mojo_decode_none,
            upstream_decode_none,
            "same pypng file",
        ),
        result_row(
            "Encode grayscale 2-bit",
            mojo_pack,
            upstream_pack,
            f"{width}x{height}, includes bit packing",
        ),
    ]
    if gpu_encode is not None:
        rows.insert(
            1,
            result_row(
                "Encode RGB8 2048x1024 GPU",
                gpu_encode,
                upstream_encode,
                "explicit device=gpu; includes transfers",
            ),
        )

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}")
    print()
    print("| Workload | mojo-pypng | pypng | Speedup | Notes |")
    print("|---|---:|---:|---:|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if free_mib is None:
        print("\nGPU benchmark skipped: no NVIDIA GPU was detected.")
    elif free_mib < 4000:
        print(f"\nGPU benchmark skipped: only {free_mib} MiB was free.")


if __name__ == "__main__":
    main()
