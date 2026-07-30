from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYPNG_LIB", os.path.join(ROOT, "dist", "libmojo-pypng.so"))

I = ctypes.c_int64
_SIGNATURES = {
    "mpp_filter_rows": ([I, I, I, I, I, I, I], I),
    "mpp_filter_rows_gpu": ([I, I, I, I, I, I, I], I),
    "mpp_unfilter_rows": ([I, I, I, I, I, I, I], I),
    "mpp_pack_bits": ([I, I, I, I, I, I, I], I),
    "mpp_unpack_bits": ([I, I, I, I, I, I, I], I),
    "mpp_pack_u16be": ([I, I, I, I, I], I),
    "mpp_unpack_u16be": ([I, I, I, I, I], I),
}

_library: ctypes.CDLL | None = None
_zlib: ctypes.CDLL | None | bool = None
_MAX_BUFFER = 2**31 - 1


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo PNG library is not built; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _positive_dimensions(**values: int) -> None:
    for name, value in values.items():
        if not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _checked_size(*factors: int) -> int:
    result = 1
    for factor in factors:
        if factor > _MAX_BUFFER // result:
            raise ValueError("requested buffer exceeds the 2 GiB FFI limit")
        result *= factor
    return result


def _integer_array(a: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    source = np.asarray(a)
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype")
    target = np.dtype(dtype)
    if source.size:
        info = np.iinfo(target)
        if np.min(source) < info.min or np.max(source) > info.max:
            raise ValueError(f"{name} cannot be represented as {target.name}")
    return np.ascontiguousarray(source, dtype=target).reshape(-1)


def gpu_memory_free_mib() -> int | None:
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


def decompress(data: bytes | bytearray, expected: int) -> np.ndarray:
    global _zlib
    _positive_dimensions(expected=expected)
    if _zlib is None:
        path = ctypes.util.find_library("z")
        if path is None:
            _zlib = False
        else:
            _zlib = ctypes.CDLL(path)
            _zlib.uncompress.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.c_void_p,
                ctypes.c_ulong,
            ]
            _zlib.uncompress.restype = ctypes.c_int
    if _zlib is False:
        return np.frombuffer(zlib.decompress(data), dtype=np.uint8)
    source = np.frombuffer(data, dtype=np.uint8)
    if not source.size:
        raise zlib.error("zlib decompression input is empty")
    destination = np.empty(expected, dtype=np.uint8)
    actual = ctypes.c_ulong(expected)
    result = _zlib.uncompress(
        addr(destination),
        ctypes.byref(actual),
        addr(source),
        source.size,
    )
    if result != 0:
        raise zlib.error(f"zlib decompression failed with status {result}")
    return destination[: actual.value]


def filter_rows(
    raw: np.ndarray,
    height: int,
    row_bytes: int,
    bpp: int,
    device: str = "cpu",
) -> np.ndarray:
    _positive_dimensions(height=height, row_bytes=row_bytes, bpp=bpp)
    if bpp > row_bytes:
        raise ValueError("bpp must not exceed row_bytes")
    raw = _integer_array(raw, np.uint8, "raw")
    expected_raw = _checked_size(height, row_bytes)
    if raw.size != expected_raw:
        raise ValueError(f"raw has {raw.size} bytes; expected {expected_raw}")
    filtered = np.empty(_checked_size(height, row_bytes + 1), dtype=np.uint8)
    total_gpu_bytes = raw.size + filtered.size
    if device == "gpu":
        free_mib = gpu_memory_free_mib()
        if total_gpu_bytes >= 2 * 1024**3:
            raise RuntimeError("GPU filtering requires less than 2 GiB of buffers")
        if free_mib is None:
            raise RuntimeError("GPU filtering requested but no NVIDIA GPU was detected")
        if free_mib < 4000:
            raise RuntimeError(
                f"GPU filtering requires 4000 MiB free; only {free_mib} MiB is available"
            )
        ok = lib().mpp_filter_rows_gpu(
            addr(raw), raw.nbytes, addr(filtered), filtered.nbytes,
            height, row_bytes, bpp
        )
        if ok:
            return filtered
        raise RuntimeError("Mojo GPU row filter failed")
    ok = lib().mpp_filter_rows(
        addr(raw), raw.nbytes, addr(filtered), filtered.nbytes,
        height, row_bytes, bpp
    )
    if not ok:
        raise RuntimeError("Mojo row filter rejected validated buffers")
    return filtered


def unfilter_rows(data: bytes, height: int, row_bytes: int, bpp: int) -> np.ndarray:
    _positive_dimensions(height=height, row_bytes=row_bytes, bpp=bpp)
    if bpp > row_bytes:
        raise ValueError("bpp must not exceed row_bytes")
    source = np.ascontiguousarray(np.frombuffer(data, dtype=np.uint8))
    expected_source = _checked_size(height, row_bytes + 1)
    if source.size != expected_source:
        raise ValueError(
            f"filtered data has {source.size} bytes; expected {expected_source}"
        )
    raw = np.empty(_checked_size(height, row_bytes), dtype=np.uint8)
    ok = lib().mpp_unfilter_rows(
        addr(source), source.nbytes, addr(raw), raw.nbytes,
        height, row_bytes, bpp
    )
    if not ok:
        raise ValueError("invalid PNG filter type")
    return raw


def pack_bits(samples: np.ndarray, height: int, samples_per_row: int, bitdepth: int) -> np.ndarray:
    _positive_dimensions(height=height, samples_per_row=samples_per_row)
    if bitdepth not in (1, 2, 4):
        raise ValueError("bitdepth must be 1, 2, or 4")
    samples = _integer_array(samples, np.uint8, "samples")
    expected_samples = _checked_size(height, samples_per_row)
    if samples.size != expected_samples:
        raise ValueError(
            f"samples has {samples.size} values; expected {expected_samples}"
        )
    row_bytes = (samples_per_row * bitdepth + 7) // 8
    packed = np.empty(_checked_size(height, row_bytes), dtype=np.uint8)
    ok = lib().mpp_pack_bits(
        addr(samples), samples.nbytes, addr(packed), packed.nbytes,
        height, samples_per_row, bitdepth
    )
    if not ok:
        raise ValueError("sample outside bit depth")
    return packed


def unpack_bits(packed: np.ndarray, height: int, samples_per_row: int, bitdepth: int) -> np.ndarray:
    _positive_dimensions(height=height, samples_per_row=samples_per_row)
    if bitdepth not in (1, 2, 4):
        raise ValueError("bitdepth must be 1, 2, or 4")
    packed = _integer_array(packed, np.uint8, "packed")
    row_bytes = (samples_per_row * bitdepth + 7) // 8
    expected_packed = _checked_size(height, row_bytes)
    if packed.size != expected_packed:
        raise ValueError(
            f"packed has {packed.size} bytes; expected {expected_packed}"
        )
    samples = np.empty(_checked_size(height, samples_per_row), dtype=np.uint8)
    ok = lib().mpp_unpack_bits(
        addr(packed), packed.nbytes, addr(samples), samples.nbytes,
        height, samples_per_row, bitdepth
    )
    if not ok:
        raise ValueError("invalid packed bit depth")
    return samples


def pack_u16be(samples: np.ndarray) -> np.ndarray:
    samples = _integer_array(samples, np.uint16, "samples")
    if not samples.size:
        return np.empty(0, dtype=np.uint8)
    packed = np.empty(samples.size * 2, dtype=np.uint8)
    ok = lib().mpp_pack_u16be(
        addr(samples), samples.nbytes, addr(packed), packed.nbytes, samples.size
    )
    if not ok:
        raise RuntimeError("Mojo 16-bit pack rejected validated buffers")
    return packed


def unpack_u16be(packed: np.ndarray) -> np.ndarray:
    packed = _integer_array(packed, np.uint8, "packed")
    if packed.size % 2:
        raise ValueError("packed 16-bit data must contain an even number of bytes")
    if not packed.size:
        return np.empty(0, dtype=np.uint16)
    samples = np.empty(packed.size // 2, dtype=np.uint16)
    ok = lib().mpp_unpack_u16be(
        addr(packed), packed.nbytes, addr(samples), samples.nbytes, samples.size
    )
    if not ok:
        raise RuntimeError("Mojo 16-bit unpack rejected validated buffers")
    return samples
