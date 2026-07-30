from __future__ import annotations

from array import array
import io
import struct
import zlib

import numpy as np
import pytest

import png
from png import _lib


rng = np.random.default_rng(20260730)


def encode(module, rows, **options):
    stream = io.BytesIO()
    module.Writer(**options).write(stream, rows)
    return stream.getvalue()


def flat(module, data):
    width, height, values, info = module.Reader(bytes=data).read_flat()
    return width, height, np.asarray(values), info


@pytest.mark.parametrize(
    "options",
    [
        dict(width=37, height=19, greyscale=True, bitdepth=1),
        dict(width=23, height=17, greyscale=True, bitdepth=2),
        dict(width=29, height=13, greyscale=True, bitdepth=4),
        dict(width=31, height=11, greyscale=True, bitdepth=8),
        dict(width=17, height=9, greyscale=True, bitdepth=16),
        dict(width=19, height=12, greyscale=False, bitdepth=8),
        dict(width=13, height=10, greyscale=False, bitdepth=16),
        dict(width=21, height=8, greyscale=True, alpha=True, bitdepth=8),
        dict(width=16, height=7, greyscale=False, alpha=True, bitdepth=8),
    ],
)
def test_mojo_encode_upstream_decode(options, upstream_png):
    planes = (1 if options["greyscale"] else 3) + options.get("alpha", False)
    dtype = np.uint16 if options["bitdepth"] == 16 else np.uint8
    pixels = rng.integers(
        0,
        2 ** options["bitdepth"],
        size=(options["height"], options["width"] * planes),
        dtype=dtype,
    )
    data = encode(png, pixels, **options)
    width, height, got, info = flat(upstream_png, data)
    assert (width, height) == (options["width"], options["height"])
    assert np.array_equal(got.reshape(pixels.shape), pixels)
    assert info["bitdepth"] == options["bitdepth"]
    assert info["greyscale"] is options["greyscale"]
    assert info["alpha"] is bool(options.get("alpha", False))


@pytest.mark.parametrize(
    "options",
    [
        dict(width=35, height=7, greyscale=True, bitdepth=1),
        dict(width=17, height=9, greyscale=True, bitdepth=4),
        dict(width=24, height=12, greyscale=True, bitdepth=8),
        dict(width=11, height=10, greyscale=True, bitdepth=16),
        dict(width=18, height=8, greyscale=False, bitdepth=8),
        dict(width=12, height=6, greyscale=False, alpha=True, bitdepth=16),
    ],
)
def test_upstream_encode_mojo_decode(options, upstream_png):
    planes = (1 if options["greyscale"] else 3) + options.get("alpha", False)
    maximum = 2 ** options["bitdepth"]
    pixels = rng.integers(
        0,
        maximum,
        size=(options["height"], options["width"] * planes),
        dtype=np.uint16 if options["bitdepth"] == 16 else np.uint8,
    )
    data = encode(upstream_png, pixels, **options)
    width, height, got, info = flat(png, data)
    assert (width, height) == (options["width"], options["height"])
    assert np.array_equal(got.reshape(pixels.shape), pixels)
    assert info["planes"] == planes


@pytest.mark.parametrize(
    "greyscale,alpha,bitdepth",
    [
        (True, False, 1),
        (True, False, 2),
        (True, False, 4),
        (True, False, 8),
        (True, False, 16),
        (True, True, 8),
        (True, True, 16),
        (False, False, 8),
        (False, False, 16),
        (False, True, 8),
        (False, True, 16),
    ],
)
def test_every_supported_direct_colour_combination(
    greyscale, alpha, bitdepth, upstream_png
):
    width, height = 9, 5
    planes = (1 if greyscale else 3) + alpha
    pixels = rng.integers(
        0,
        2**bitdepth,
        size=(height, width * planes),
        dtype=np.uint16 if bitdepth == 16 else np.uint8,
    )
    options = dict(
        width=width,
        height=height,
        greyscale=greyscale,
        alpha=alpha,
        bitdepth=bitdepth,
    )
    for encoder, decoder in ((png, upstream_png), (upstream_png, png)):
        data = encode(encoder, pixels, **options)
        got = flat(decoder, data)[2]
        assert np.array_equal(got.reshape(pixels.shape), pixels)


@pytest.mark.parametrize("bitdepth", [1, 2, 4, 8])
def test_every_supported_palette_bitdepth(bitdepth, upstream_png):
    width, height = 9, 5
    entry_count = min(2**bitdepth, 16)
    palette = [
        (index, 255 - index, index * 7 % 256, index * 13 % 256)
        for index in range(entry_count)
    ]
    pixels = rng.integers(
        0, entry_count, size=(height, width), dtype=np.uint8
    )
    options = dict(
        width=width,
        height=height,
        greyscale=False,
        bitdepth=bitdepth,
        palette=palette,
    )
    data = encode(png, pixels, **options)
    assert np.array_equal(
        flat(upstream_png, data)[2].reshape(pixels.shape), pixels
    )


def test_palette_roundtrip_and_as_direct(upstream_png):
    palette = [(4, 10, 250, 0), (20, 30, 40, 128), (200, 5, 60)]
    indexes = np.array([[0, 1, 2, 1, 0], [2, 2, 1, 0, 1]], dtype=np.uint8)
    data = encode(
        png,
        indexes,
        width=5,
        height=2,
        greyscale=False,
        bitdepth=2,
        palette=palette,
    )
    _, _, upstream_rows, upstream_info = upstream_png.Reader(bytes=data).asDirect()
    _, _, mojo_rows, mojo_info = png.Reader(bytes=data).asDirect()
    assert list(map(list, mojo_rows)) == list(map(list, upstream_rows))
    assert mojo_info["palette"] == upstream_info["palette"]
    assert mojo_info["planes"] == upstream_info["planes"] == 4


def test_metadata_matches_upstream(upstream_png):
    options = dict(
        width=3,
        height=2,
        greyscale=False,
        bitdepth=8,
        gamma=0.45455,
        transparent=(1, 2, 3),
        background=(10, 20, 30),
        x_pixels_per_unit=3780,
        y_pixels_per_unit=3780,
        unit_is_meter=True,
    )
    rows = [[1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2
    data = encode(png, rows, **options)
    _, _, _, got = upstream_png.Reader(bytes=data).read()
    assert got["gamma"] == pytest.approx(0.45455)
    assert got["transparent"] == (1, 2, 3)
    assert got["background"] == (10, 20, 30)
    assert tuple(got["physical"]) == (3780, 3780, True)


def paeth(a, b, c):
    p = a + b - c
    distances = (abs(p - a), abs(p - b), abs(p - c))
    return (a, b, c)[distances.index(min(distances))]


def apply_filter(raw, previous, filter_type, bpp):
    result = bytearray(len(raw))
    for i, value in enumerate(raw):
        left = raw[i - bpp] if i >= bpp else 0
        up = previous[i] if previous is not None else 0
        upper_left = previous[i - bpp] if previous is not None and i >= bpp else 0
        predictor = (
            0,
            left,
            up,
            (left + up) // 2,
            paeth(left, up, upper_left),
        )[filter_type]
        result[i] = (value - predictor) & 255
    return result


def filtered_png(filter_types):
    width, height, bpp = 7, len(filter_types), 3
    rows = [
        bytes(((x * 29 + y * 47 + x * y * 3) & 255) for x in range(width * bpp))
        for y in range(height)
    ]
    scanlines = bytearray()
    previous = None
    for row, filter_type in zip(rows, filter_types):
        scanlines.append(filter_type)
        scanlines.extend(apply_filter(row, previous, filter_type, bpp))
        previous = row
    stream = io.BytesIO()
    png.write_chunks(
        stream,
        [
            (b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)),
            (b"IDAT", zlib.compress(scanlines)),
            (b"IEND", b""),
        ],
    )
    return stream.getvalue(), b"".join(rows)


def test_all_five_published_filter_algorithms():
    data, expected = filtered_png([0, 1, 2, 3, 4])
    _, _, values, _ = png.Reader(bytes=data).read_flat()
    assert bytes(values) == expected


def test_adaptive_writer_uses_filters_and_is_upstream_readable(upstream_png):
    x = np.arange(256, dtype=np.uint8)
    pixels = np.tile(x, (64, 3))
    data = encode(
        png, pixels, width=256, height=64, greyscale=False, bitdepth=8
    )
    idat = b"".join(
        payload for tag, payload in png.Reader(bytes=data).chunks() if tag == b"IDAT"
    )
    raw = zlib.decompress(idat)
    filter_types = raw[:: 256 * 3 + 1]
    assert any(value != 0 for value in filter_types)
    assert np.array_equal(flat(upstream_png, data)[2], pixels.reshape(-1))


@pytest.mark.parametrize("bitdepth", [1, 2, 4])
def test_bit_pack_unpack_matches_reference(bitdepth):
    height, width = 13, 39
    samples = rng.integers(0, 2**bitdepth, size=(height, width), dtype=np.uint8)
    packed = _lib.pack_bits(samples, height, width, bitdepth)
    reference = bytearray()
    for row in samples:
        accumulator = 0
        occupied = 0
        for value in row:
            accumulator = (accumulator << bitdepth) | int(value)
            occupied += bitdepth
            if occupied == 8:
                reference.append(accumulator)
                accumulator = occupied = 0
        if occupied:
            reference.append(accumulator << (8 - occupied))
    assert packed.tobytes() == bytes(reference)
    got = _lib.unpack_bits(packed, height, width, bitdepth)
    assert np.array_equal(got.reshape(samples.shape), samples)


def test_u16_big_endian_conversion():
    samples = np.array([0, 1, 255, 256, 32768, 65535], dtype=np.uint16)
    packed = _lib.pack_u16be(samples)
    assert packed.tobytes() == struct.pack("!6H", *samples)
    assert np.array_equal(_lib.unpack_u16be(packed), samples)


def test_filter_simd_tail_roundtrip():
    height, row_bytes, bpp = 9, 1031, 3
    raw = rng.integers(0, 256, size=(height, row_bytes), dtype=np.uint8)
    filtered = _lib.filter_rows(raw, height, row_bytes, bpp)
    restored = _lib.unfilter_rows(
        filtered.tobytes(), height, row_bytes, bpp
    )
    assert np.array_equal(restored.reshape(raw.shape), raw)


def test_filter_parallel_threshold_roundtrip():
    height, row_bytes, bpp = 65, 4097, 4
    raw = rng.integers(0, 256, size=(height, row_bytes), dtype=np.uint8)
    filtered = _lib.filter_rows(raw, height, row_bytes, bpp)
    restored = _lib.unfilter_rows(
        filtered.tobytes(), height, row_bytes, bpp
    )
    assert np.array_equal(restored.reshape(raw.shape), raw)


@pytest.mark.parametrize("height,row_bytes", [(7, 1031), (257, 4099)])
def test_filter_none_serial_and_parallel_thresholds(height, row_bytes):
    raw = rng.integers(0, 256, size=(height, row_bytes), dtype=np.uint8)
    filtered = np.zeros((height, row_bytes + 1), dtype=np.uint8)
    filtered[:, 1:] = raw
    restored = _lib.unfilter_rows(
        filtered.tobytes(), height, row_bytes, 3
    )
    assert np.array_equal(restored.reshape(raw.shape), raw)


def test_gpu_filter_matches_cpu_or_reports_unavailable():
    height, row_bytes, bpp = 17, 1031, 3
    raw = rng.integers(0, 256, size=(height, row_bytes), dtype=np.uint8)
    cpu = _lib.filter_rows(raw, height, row_bytes, bpp)
    free_mib = _lib.gpu_memory_free_mib()
    if free_mib is None or free_mib < 4000:
        with pytest.raises(RuntimeError):
            _lib.filter_rows(raw, height, row_bytes, bpp, device="gpu")
    else:
        gpu = _lib.filter_rows(raw, height, row_bytes, bpp, device="gpu")
        assert np.array_equal(gpu, cpu)


def test_ffi_wrappers_reject_wrong_lengths_and_lossy_dtypes():
    raw = np.zeros(11, dtype=np.uint8)
    with pytest.raises(ValueError, match="expected 12"):
        _lib.filter_rows(raw, 3, 4, 1)
    with pytest.raises(ValueError, match="expected 15"):
        _lib.unfilter_rows(bytes(14), 3, 4, 1)
    with pytest.raises(ValueError, match="expected 12"):
        _lib.pack_bits(np.zeros(11, dtype=np.uint8), 3, 4, 2)
    with pytest.raises(ValueError, match="expected 3"):
        _lib.unpack_bits(np.zeros(2, dtype=np.uint8), 3, 4, 2)
    with pytest.raises(ValueError, match="cannot be represented"):
        _lib.pack_u16be(np.array([65536], dtype=np.uint32))
    with pytest.raises(ValueError, match="cannot be represented"):
        _lib.pack_bits(np.array([256], dtype=np.uint16), 1, 1, 1)
    with pytest.raises(TypeError, match="integer dtype"):
        _lib.pack_u16be(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="even number"):
        _lib.unpack_u16be(np.zeros(3, dtype=np.uint8))


def test_raw_ffi_rejects_null_and_short_buffers():
    library = _lib.lib()
    assert library.mpp_filter_rows(0, 4, 0, 5, 1, 4, 1) == 0
    source = np.zeros(4, dtype=np.uint8)
    destination = np.zeros(5, dtype=np.uint8)
    assert (
        library.mpp_filter_rows(
            _lib.addr(source), 3, _lib.addr(destination), 5, 1, 4, 1
        )
        == 0
    )


def test_rgb_and_rgba_conversion_parity(upstream_png):
    pixels = np.array([[0, 15, 255], [7, 11, 13]], dtype=np.uint8)
    data = encode(upstream_png, pixels, width=3, height=2, greyscale=True)
    for method in ("asRGB8", "asRGBA8"):
        upstream = getattr(upstream_png.Reader(bytes=data), method)()
        mojo = getattr(png.Reader(bytes=data), method)()
        assert upstream[:2] == mojo[:2]
        assert list(map(list, upstream[2])) == list(map(list, mojo[2]))
        assert upstream[3]["planes"] == mojo[3]["planes"]


def test_from_array_image_write_and_save(tmp_path):
    pixels = np.arange(60, dtype=np.uint8).reshape(4, 15)
    image = png.from_array(pixels, "RGB")
    stream = io.BytesIO()
    image.write(stream)
    assert np.array_equal(flat(png, stream.getvalue())[2], pixels.reshape(-1))
    image = png.from_array(pixels, "RGB")
    target = tmp_path / "image.png"
    image.save(target)
    assert target.read_bytes().startswith(png.signature)
    assert isinstance(png.fromarray(pixels, "RGB"), png.Image)


def test_documented_writer_entry_points_and_reader_conversions():
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 9)
    options = dict(width=3, height=2, greyscale=False, bitdepth=8)
    streams = []
    for method, rows in (
        ("write", pixels),
        ("write_passes", pixels),
        ("write_array", pixels.reshape(-1)),
    ):
        stream = io.BytesIO()
        getattr(png.Writer(**options), method)(stream, rows)
        streams.append(stream.getvalue())
    assert all(flat(png, data)[2].tolist() == pixels.reshape(-1).tolist() for data in streams)

    grey = encode(png, [[0, 127, 255]], width=3, height=1, greyscale=True)
    rgb = png.Reader(bytes=grey).asRGB()
    rgba = png.Reader(bytes=grey).asRGBA()
    assert list(rgb[2]) == [[0, 0, 0, 127, 127, 127, 255, 255, 255]]
    assert list(rgba[2]) == [array("B", [0, 0, 0, 255, 127, 127, 127, 255, 255, 255, 255, 255])]


def test_write_packed():
    packed = [b"\x12\x30", b"\xab\xc0"]
    writer = png.Writer(4, 2, greyscale=True, bitdepth=4)
    stream = io.BytesIO()
    assert writer.write_packed(stream, packed) == 2
    assert flat(png, stream.getvalue())[2].tolist() == [1, 2, 3, 0, 10, 11, 12, 0]


def test_chunk_crc_error_and_lenient_warning():
    data, _ = filtered_png([0])
    corrupt = bytearray(data)
    corrupt[29] ^= 1
    with pytest.raises(png.ChunkError):
        list(png.Reader(bytes=corrupt).chunks())
    with pytest.warns(RuntimeWarning):
        reader = png.Reader(bytes=corrupt)
        reader.validate_signature()
        reader.chunk(lenient=True)


def test_protocol_validation():
    with pytest.raises(png.ProtocolError):
        png.Writer(0, 2)
    with pytest.raises(png.ProtocolError):
        png.Writer(2, 2, greyscale=False, bitdepth=4)
    with pytest.raises(png.ProtocolError):
        png.Writer(2, 2, interlace=True)
    with pytest.raises(png.ProtocolError):
        encode(png, [[0, 1]], width=3, height=1)
    with pytest.raises(png.ProtocolError):
        png.Writer(2, 2, device="tpu")


def test_reader_rejects_invalid_signature():
    with pytest.raises(png.FormatError):
        png.Reader(bytes=b"not a png").read()
