"""pypng-compatible PNG encode/decode with Mojo scanline kernels."""

from __future__ import annotations

from array import array
from collections import namedtuple
import io
import itertools
import os
import re
import struct
import warnings
import zlib

import numpy as np

from . import _lib

__version__ = "0.1.0"
signature = b"\x89PNG\r\n\x1a\n"
Resolution = namedtuple(
    "Resolution", "x y unit_is_meter", defaults=(False,)
)


class Error(Exception):
    def __str__(self):
        return self.__class__.__name__ + ": " + " ".join(self.args)


class FormatError(Error):
    pass


class ProtocolError(Error):
    pass


class ChunkError(FormatError):
    pass


class Default:
    pass


def _size(size, width, height):
    if size is not None:
        if width is not None or height is not None:
            raise ProtocolError("size and width/height cannot both be used")
        try:
            width, height = size
        except (TypeError, ValueError) as exc:
            raise ProtocolError("size must be a pair") from exc
    return width, height


def _color(value, greyscale, name):
    if value is None:
        return None
    if greyscale:
        if isinstance(value, int):
            value = (value,)
        if len(value) != 1:
            raise ProtocolError(f"{name} for greyscale must be 1-tuple")
    elif len(value) != 3:
        raise ProtocolError(f"{name} colour must be a triple of integers")
    if any(not isinstance(v, (int, np.integer)) or v < 0 for v in value):
        raise ProtocolError(f"{name} colour must contain non-negative integers")
    return tuple(int(v) for v in value)


def _palette(value, bitdepth):
    if value is None:
        return None
    entries = [tuple(int(x) for x in entry) for entry in value]
    if not entries or len(entries) > 2**bitdepth:
        raise ProtocolError("palette has invalid number of entries")
    seen_rgb = False
    for entry in entries:
        if len(entry) not in (3, 4) or any(x < 0 or x > 255 for x in entry):
            raise ProtocolError("palette entries must be RGB or RGBA bytes")
        if len(entry) == 3:
            seen_rgb = True
        elif seen_rgb:
            raise ProtocolError("RGBA palette entries must precede RGB entries")
    return entries


class Writer:
    def __init__(
        self,
        width=None,
        height=None,
        size=None,
        greyscale=Default,
        alpha=False,
        bitdepth=8,
        palette=None,
        transparent=None,
        background=None,
        gamma=None,
        compression=None,
        interlace=False,
        planes=None,
        colormap=None,
        maxval=None,
        chunk_limit=2**20,
        x_pixels_per_unit=None,
        y_pixels_per_unit=None,
        unit_is_meter=False,
        device="cpu",
    ):
        width, height = _size(size, width, height)
        if not isinstance(width, (int, np.integer)) or not isinstance(
            height, (int, np.integer)
        ):
            raise ProtocolError("width and height must be integers")
        if width <= 0 or height <= 0 or width > 2**31 - 1 or height > 2**31 - 1:
            raise ProtocolError("width and height must be between 1 and 2**31-1")
        if maxval is not None:
            raise ProtocolError("maxval inference is not supported; specify bitdepth")
        if isinstance(bitdepth, (tuple, list)):
            if len(set(bitdepth)) != 1:
                raise ProtocolError("per-channel bit depths are not supported")
            bitdepth = bitdepth[0]
        if bitdepth not in (1, 2, 4, 8, 16):
            raise ProtocolError("bitdepth must be 1, 2, 4, 8, or 16")
        if interlace:
            raise ProtocolError("Adam7 interlaced encoding is not supported")
        if compression is not None and compression not in range(-1, 10):
            raise ProtocolError("compression must be from -1 to 9")
        if device not in ("cpu", "gpu"):
            raise ProtocolError("device must be 'cpu' or 'gpu'")

        self.width = int(width)
        self.height = int(height)
        self.bitdepth = int(bitdepth)
        self.palette = _palette(palette, self.bitdepth)
        if greyscale is Default:
            greyscale = self.palette is None
        self.greyscale = bool(greyscale)
        self.alpha = bool(alpha)
        self.colormap = self.palette is not None
        if self.colormap and (self.greyscale or self.alpha or transparent is not None):
            raise ProtocolError("palette images cannot be greyscale, alpha, or transparent")
        if self.alpha and transparent is not None:
            raise ProtocolError("transparent colour not allowed with alpha channel")
        if self.bitdepth < 8 and not (self.greyscale or self.colormap):
            raise ProtocolError("bit depths below 8 require greyscale or palette")
        if self.bitdepth == 16 and self.colormap:
            raise ProtocolError("palette images cannot have 16-bit indexes")

        self.color_planes = 1 if self.greyscale or self.colormap else 3
        self.planes = 1 if self.colormap else self.color_planes + self.alpha
        self.color_type = 3 if self.colormap else (
            4 * self.alpha + 2 * (not self.greyscale)
        )
        self.transparent = _color(transparent, self.greyscale, "transparent")
        self.background = _color(background, self.greyscale, "background")
        self.gamma = gamma
        if gamma is not None and gamma <= 0:
            raise ProtocolError("gamma must be positive")
        self.compression = compression
        self.interlace = False
        self.chunk_limit = int(chunk_limit)
        if self.chunk_limit <= 0:
            raise ProtocolError("chunk_limit must be positive")
        self.x_pixels_per_unit = x_pixels_per_unit
        self.y_pixels_per_unit = y_pixels_per_unit
        self.unit_is_meter = bool(unit_is_meter)
        self.device = device
        self.psize = self.bitdepth / 8 * self.planes

    def _samples(self, rows):
        expected = self.width * self.planes
        if isinstance(rows, np.ndarray) and rows.ndim == 2:
            if rows.shape != (self.height, expected):
                raise ProtocolError(
                    f"Expected shape ({self.height}, {expected}) but got {rows.shape}"
                )
            if not np.issubdtype(rows.dtype, np.integer):
                if np.any(rows != np.floor(rows)):
                    raise ProtocolError("pixel values must be integers")
            if rows.size and (
                np.min(rows) < 0 or np.max(rows) >= 2**self.bitdepth
            ):
                raise ProtocolError("pixel value outside bit depth")
            dtype = np.uint16 if self.bitdepth == 16 else np.uint8
            return np.ascontiguousarray(rows, dtype=dtype)
        materialized = []
        for i, row in enumerate(rows):
            values = np.asarray(list(row) if not hasattr(row, "__len__") else row)
            values = values.reshape(-1)
            if values.size != expected:
                raise ProtocolError(
                    f"Expected {expected} values but got {values.size} values, in row {i}"
                )
            if not np.issubdtype(values.dtype, np.integer):
                if np.any(values != np.floor(values)):
                    raise ProtocolError("pixel values must be integers")
            if values.size and (
                np.min(values) < 0 or np.max(values) >= 2**self.bitdepth
            ):
                raise ProtocolError("pixel value outside bit depth")
            materialized.append(values)
        if len(materialized) != self.height:
            raise ProtocolError(
                f"rows supplied ({len(materialized)}) does not match height ({self.height})"
            )
        dtype = np.uint16 if self.bitdepth == 16 else np.uint8
        return np.ascontiguousarray(np.stack(materialized), dtype=dtype)

    def _pack(self, samples):
        if self.bitdepth < 8:
            return _lib.pack_bits(
                samples, self.height, self.width * self.planes, self.bitdepth
            )
        if self.bitdepth == 16:
            return _lib.pack_u16be(samples).reshape(self.height, -1)
        return np.ascontiguousarray(samples, dtype=np.uint8)

    def _write_raw(self, outfile, packed):
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim == 1:
            packed = packed.reshape(self.height, -1)
        row_bytes = packed.shape[1]
        bpp = max(1, (self.bitdepth * self.planes + 7) // 8)
        filtered = _lib.filter_rows(
            packed, self.height, row_bytes, bpp, device=self.device
        )
        self.write_preamble(outfile)
        level = zlib.Z_DEFAULT_COMPRESSION if self.compression is None else self.compression
        compressed = zlib.compress(filtered.tobytes(), level)
        for offset in range(0, len(compressed), self.chunk_limit):
            write_chunk(outfile, b"IDAT", compressed[offset : offset + self.chunk_limit])
        write_chunk(outfile, b"IEND")
        return self.height

    def write(self, outfile, rows):
        return self._write_raw(outfile, self._pack(self._samples(rows)))

    def write_passes(self, outfile, rows):
        return self.write(outfile, rows)

    def write_packed(self, outfile, rows):
        materialized = [bytes(row) for row in rows]
        if len(materialized) != self.height:
            raise ProtocolError(
                f"rows supplied ({len(materialized)}) does not match height ({self.height})"
            )
        row_bytes = (self.width * self.planes * self.bitdepth + 7) // 8
        if any(len(row) != row_bytes for row in materialized):
            raise ProtocolError("packed row has wrong length")
        packed = np.frombuffer(b"".join(materialized), dtype=np.uint8).reshape(
            self.height, row_bytes
        )
        return self._write_raw(outfile, packed)

    def write_preamble(self, outfile):
        try:
            outfile.write(signature)
        except TypeError as exc:
            raise ProtocolError("PNG must be written to a binary stream") from exc
        write_chunk(
            outfile,
            b"IHDR",
            struct.pack(
                "!2I5B",
                self.width,
                self.height,
                self.bitdepth,
                self.color_type,
                0,
                0,
                0,
            ),
        )
        if self.gamma is not None:
            write_chunk(outfile, b"gAMA", struct.pack("!I", round(self.gamma * 100000)))
        if self.palette:
            write_chunk(outfile, b"PLTE", bytes(x for p in self.palette for x in p[:3]))
            alphas = bytes(p[3] for p in self.palette if len(p) == 4)
            if alphas:
                write_chunk(outfile, b"tRNS", alphas)
        if self.transparent is not None:
            write_chunk(
                outfile,
                b"tRNS",
                struct.pack("!H" if self.greyscale else "!3H", *self.transparent),
            )
        if self.background is not None:
            data = (
                struct.pack("B", self.background[0])
                if self.colormap
                else struct.pack(
                    "!H" if self.greyscale else "!3H", *self.background
                )
            )
            write_chunk(outfile, b"bKGD", data)
        if self.x_pixels_per_unit is not None and self.y_pixels_per_unit is not None:
            write_chunk(
                outfile,
                b"pHYs",
                struct.pack(
                    "!IIB",
                    self.x_pixels_per_unit,
                    self.y_pixels_per_unit,
                    int(self.unit_is_meter),
                ),
            )

    def write_array(self, outfile, pixels):
        return self.write(outfile, self.array_scanlines(pixels))

    def array_scanlines(self, pixels):
        values_per_row = self.width * self.planes
        for y in range(self.height):
            yield pixels[y * values_per_row : (y + 1) * values_per_row]

    def array_scanlines_interlace(self, pixels):
        raise ProtocolError("Adam7 interlaced encoding is not supported")


def write_chunk(outfile, tag, data=b""):
    tag = bytes(tag)
    data = bytes(data)
    if len(tag) != 4:
        raise ValueError("chunk tag must be four bytes")
    outfile.write(struct.pack("!I", len(data)))
    outfile.write(tag)
    outfile.write(data)
    outfile.write(struct.pack("!I", zlib.crc32(data, zlib.crc32(tag)) & 0xFFFFFFFF))


def write_chunks(out, chunks):
    out.write(signature)
    for chunk in chunks:
        write_chunk(out, *chunk)


class Reader:
    def __init__(self, _guess=None, filename=None, file=None, bytes=None):
        supplied = sum(x is not None for x in (_guess, filename, file, bytes))
        if supplied != 1:
            raise TypeError("Reader() takes exactly 1 argument")
        if _guess is not None:
            if isinstance(_guess, (str, os.PathLike)):
                filename = _guess
            elif hasattr(_guess, "read"):
                file = _guess
            else:
                bytes = _guess
        if filename is not None:
            self.file = open(filename, "rb")
        elif file is not None:
            self.file = file
        else:
            self.file = io.BytesIO(bytes if isinstance(bytes, type(b"")) else bytearray(bytes))
        self.signature = None
        self.atchunk = None
        self.transparent = None
        self.background = None
        self.gamma = None
        self.plte = None
        self.trns = None
        self.sbit = None

    def validate_signature(self):
        if self.signature is not None:
            return
        self.signature = self.file.read(8)
        if not self.signature:
            raise EOFError("End of PNG stream.")
        if self.signature != signature:
            raise FormatError("PNG file has invalid signature.")

    def _chunk_len_type(self):
        header = self.file.read(8)
        if not header:
            return None
        if len(header) != 8:
            raise FormatError("End of file whilst reading chunk length and type.")
        length, tag = struct.unpack("!I4s", header)
        if length > 2**31 - 1:
            raise FormatError(f"Chunk {tag!r} is too large: {length}.")
        if not all(65 <= b <= 90 or 97 <= b <= 122 for b in tag):
            raise FormatError(f"Chunk {tag!r} has invalid Chunk Type.")
        return length, tag

    def chunk(self, lenient=False):
        self.validate_signature()
        current = self.atchunk or self._chunk_len_type()
        self.atchunk = None
        if current is None:
            raise ChunkError("No more chunks.")
        length, tag = current
        data = self.file.read(length)
        checksum = self.file.read(4)
        if len(data) != length or len(checksum) != 4:
            raise ChunkError(f"Chunk {tag!r} is truncated.")
        actual = struct.unpack("!I", checksum)[0]
        expected = zlib.crc32(data, zlib.crc32(tag)) & 0xFFFFFFFF
        if actual != expected:
            message = (
                f"Checksum error in {tag.decode('ascii')} chunk: "
                f"0x{actual:08X} != 0x{expected:08X}."
            )
            if lenient:
                warnings.warn(message, RuntimeWarning)
            else:
                raise ChunkError(message)
        return tag, data

    def chunks(self):
        while True:
            tag, data = self.chunk()
            yield tag, data
            if tag == b"IEND":
                return

    def preamble(self, lenient=False):
        self.validate_signature()
        while True:
            if self.atchunk is None:
                self.atchunk = self._chunk_len_type()
            if self.atchunk is None:
                raise FormatError("This PNG file has no IDAT chunks.")
            if self.atchunk[1] == b"IDAT":
                return
            self.process_chunk(lenient=lenient)

    def process_chunk(self, lenient=False):
        tag, data = self.chunk(lenient=lenient)
        method = getattr(self, "_process_" + tag.decode("ascii"), None)
        if method:
            method(data)

    def _process_IHDR(self, data):
        if len(data) != 13:
            raise FormatError("IHDR chunk has incorrect length.")
        (
            self.width,
            self.height,
            self.bitdepth,
            self.color_type,
            compression,
            filter_method,
            self.interlace,
        ) = struct.unpack("!2I5B", data)
        if self.width == 0 or self.height == 0:
            raise FormatError("width and height must be nonzero")
        valid = {
            0: (1, 2, 4, 8, 16),
            2: (8, 16),
            3: (1, 2, 4, 8),
            4: (8, 16),
            6: (8, 16),
        }
        if self.color_type not in valid or self.bitdepth not in valid[self.color_type]:
            raise FormatError("invalid bit depth or colour type")
        if compression != 0 or filter_method != 0:
            raise FormatError("unknown PNG compression or filter method")
        if self.interlace not in (0, 1):
            raise FormatError("unknown PNG interlace method")
        self.colormap = self.color_type == 3
        self.greyscale = self.color_type in (0, 4)
        self.alpha = self.color_type in (4, 6)
        self.color_planes = 1 if self.greyscale or self.colormap else 3
        self.planes = 1 if self.colormap else self.color_planes + self.alpha
        self.psize = self.bitdepth / 8 * self.planes
        self.row_bytes = (self.width * self.planes * self.bitdepth + 7) // 8

    def _process_PLTE(self, data):
        if not data or len(data) % 3 or len(data) > 768:
            raise FormatError("PLTE chunk has incorrect length.")
        self.plte = data

    def _process_tRNS(self, data):
        self.trns = data
        if self.colormap:
            if self.plte is not None and len(data) > len(self.plte) // 3:
                raise FormatError("tRNS chunk is too long.")
        elif self.alpha:
            raise FormatError("tRNS chunk is not valid with alpha")
        else:
            expected = 2 if self.greyscale else 6
            if len(data) != expected:
                raise FormatError("tRNS chunk has incorrect length.")
            self.transparent = struct.unpack("!H" if self.greyscale else "!3H", data)

    def _process_bKGD(self, data):
        fmt = "B" if self.colormap else ("!H" if self.greyscale else "!3H")
        try:
            self.background = struct.unpack(fmt, data)
        except struct.error as exc:
            raise FormatError("bKGD chunk has incorrect length.") from exc

    def _process_gAMA(self, data):
        if len(data) != 4:
            raise FormatError("gAMA chunk has incorrect length.")
        self.gamma = struct.unpack("!I", data)[0] / 100000.0

    def _process_sBIT(self, data):
        self.sbit = data

    def _process_pHYs(self, data):
        if len(data) != 9:
            raise FormatError("pHYs chunk has incorrect length.")
        x, y, unit = struct.unpack("!IIB", data)
        self.x_pixels_per_unit = x
        self.y_pixels_per_unit = y
        self.unit_is_meter = bool(unit)

    def read(self, lenient=False):
        self.preamble(lenient=lenient)
        if self.interlace:
            raise FormatError("Adam7 interlaced decoding is not supported")
        compressed = bytearray()
        while True:
            tag, data = self.chunk(lenient=lenient)
            if tag == b"IDAT":
                compressed.extend(data)
            elif tag == b"IEND":
                break
        try:
            expected = self.height * (self.row_bytes + 1)
            filtered = _lib.decompress(compressed, expected)
        except zlib.error as exc:
            raise FormatError(f"zlib decompression failed: {exc}") from exc
        if len(filtered) != expected:
            raise FormatError("Wrong size for decompressed IDAT chunk.")
        bpp = max(1, (self.bitdepth * self.planes + 7) // 8)
        scanlines = np.frombuffer(filtered, dtype=np.uint8).reshape(
            self.height, self.row_bytes + 1
        )
        if np.all(scanlines[:, 0] == 0):
            packed = scanlines[:, 1:]
        else:
            try:
                packed = _lib.unfilter_rows(
                    filtered, self.height, self.row_bytes, bpp
                ).reshape(self.height, self.row_bytes)
            except ValueError as exc:
                raise FormatError("Invalid PNG Filter Type.") from exc

        samples_per_row = self.width * self.planes
        if self.bitdepth < 8:
            values = _lib.unpack_bits(
                packed, self.height, samples_per_row, self.bitdepth
            ).reshape(self.height, samples_per_row)
            rows = [bytearray(row.tobytes()) for row in values]
        elif self.bitdepth == 16:
            values = _lib.unpack_u16be(packed).reshape(
                self.height, samples_per_row
            )
            rows = []
            for row in values:
                converted = array("H")
                converted.frombytes(row.tobytes())
                rows.append(converted)
        else:
            rows = iter(packed)
        info = {
            "greyscale": self.greyscale,
            "alpha": self.alpha,
            "planes": self.planes,
            "bitdepth": self.bitdepth,
            "interlace": self.interlace,
            "size": (self.width, self.height),
        }
        for name in ("gamma", "transparent", "background"):
            value = getattr(self, name, None)
            if value is not None:
                info[name] = value
        if hasattr(self, "x_pixels_per_unit"):
            info["physical"] = Resolution(
                self.x_pixels_per_unit,
                self.y_pixels_per_unit,
                self.unit_is_meter,
            )
        if self.plte:
            info["palette"] = self.palette()
        return self.width, self.height, iter(rows), info

    def read_flat(self):
        width, height, rows, info = self.read()
        typecode = "H" if info["bitdepth"] > 8 else "B"
        return width, height, array(typecode, itertools.chain.from_iterable(rows)), info

    def palette(self, alpha="natural"):
        if not self.plte:
            raise FormatError("Required PLTE chunk is missing in colour type 3 image.")
        rgb = [tuple(self.plte[i : i + 3]) for i in range(0, len(self.plte), 3)]
        if self.trns or alpha == "force":
            alphas = list(self.trns or b"") + [255] * (len(rgb) - len(self.trns or b""))
            return [color + (alphas[i],) for i, color in enumerate(rgb)]
        return rgb

    def asDirect(self):
        self.preamble()
        if not self.colormap and not self.trns and not self.sbit:
            return self.read()
        width, height, rows, info = self.read()
        rows = list(rows)
        if self.colormap:
            palette = self.palette()
            converted = []
            for row in rows:
                converted.append(
                    array("B", itertools.chain.from_iterable(palette[index] for index in row))
                )
            rows = converted
            info["colormap"] = False
            info["greyscale"] = False
            info["alpha"] = bool(self.trns)
            info["bitdepth"] = 8
            info["planes"] = 4 if self.trns else 3
        elif self.trns:
            maximum = 2**info["bitdepth"] - 1
            planes = info["planes"]
            converted = []
            for row in rows:
                target = []
                for offset in range(0, len(row), planes):
                    pixel = tuple(row[offset : offset + planes])
                    target.extend(pixel)
                    target.append(0 if pixel == self.transparent else maximum)
                converted.append(array("H" if info["bitdepth"] > 8 else "B", target))
            rows = converted
            info["alpha"] = True
            info["planes"] += 1
        if self.sbit:
            depths = tuple(self.sbit)
            target = max(depths)
            if target <= 0 or target > info["bitdepth"]:
                raise Error("invalid sBIT chunk")
            shift = info["bitdepth"] - target
            rows = [[value >> shift for value in row] for row in rows]
            info["bitdepth"] = target
        return width, height, iter(rows), info

    def _as_rescale(self, get, targetbitdepth):
        width, height, rows, info = get()
        source_max = 2**info["bitdepth"] - 1
        target_max = 2**targetbitdepth - 1
        info["bitdepth"] = targetbitdepth
        if source_max == target_max:
            return width, height, rows, info
        factor = target_max / source_max
        return (
            width,
            height,
            ([round(value * factor) for value in row] for row in rows),
            info,
        )

    def asRGB8(self):
        return self._as_rescale(self.asRGB, 8)

    def asRGBA8(self):
        return self._as_rescale(self.asRGBA, 8)

    def asRGB(self):
        width, height, rows, info = self.asDirect()
        if info["alpha"]:
            raise Error("will not convert image with alpha channel to RGB")
        if not info["greyscale"]:
            return width, height, rows, info
        info["greyscale"] = False
        info["planes"] = 3
        return (
            width,
            height,
            ([value for grey in row for value in (grey, grey, grey)] for row in rows),
            info,
        )

    def asRGBA(self):
        width, height, rows, info = self.asDirect()
        if info["alpha"] and not info["greyscale"]:
            return width, height, rows, info
        maximum = 2**info["bitdepth"] - 1
        source_greyscale = info["greyscale"]
        source_alpha = info["alpha"]

        def convert():
            for row in rows:
                target = []
                if source_greyscale and source_alpha:
                    for offset in range(0, len(row), 2):
                        grey, alpha = row[offset : offset + 2]
                        target.extend((grey, grey, grey, alpha))
                elif source_greyscale:
                    for grey in row:
                        target.extend((grey, grey, grey, maximum))
                else:
                    for offset in range(0, len(row), 3):
                        target.extend(row[offset : offset + 3])
                        target.append(maximum)
                yield array("H" if info["bitdepth"] > 8 else "B", target)

        info["alpha"] = True
        info["greyscale"] = False
        info["planes"] = 4
        return width, height, convert(), info


_mode_re = re.compile(r"(LA?|RGBA?);?([0-9]*)", re.IGNORECASE)


def from_array(a, mode=None, info={}):
    info = dict(info)
    match = _mode_re.fullmatch(mode or "")
    if not match:
        raise Error("mode string should be 'RGB' or 'L;16' or similar.")
    color_mode, depth = match.groups()
    color_mode = color_mode.upper()
    greyscale = color_mode.startswith("L")
    alpha = color_mode.endswith("A")
    if "greyscale" in info and bool(info["greyscale"]) != greyscale:
        raise ProtocolError("info['greyscale'] should match mode.")
    if "alpha" in info and bool(info["alpha"]) != alpha:
        raise ProtocolError("info['alpha'] should match mode.")
    info["greyscale"] = greyscale
    info["alpha"] = alpha
    if depth:
        depth = int(depth)
        if "bitdepth" in info and info["bitdepth"] != depth:
            raise ProtocolError("bitdepth should match mode")
        info["bitdepth"] = depth
    iterator, probe = itertools.tee(iter(a))
    try:
        first = next(probe)
    except StopIteration as exc:
        raise ProtocolError("array must contain at least one row") from exc
    if "height" not in info:
        try:
            info["height"] = len(a)
        except TypeError as exc:
            raise ProtocolError("supply info['height'] for an iterator") from exc
    planes = len(color_mode)
    if "width" not in info:
        info["width"] = len(first) // planes
    if "bitdepth" not in info:
        dtype = getattr(first, "dtype", None)
        if dtype is not None:
            info["bitdepth"] = 1 if dtype.kind == "b" else dtype.itemsize * 8
        else:
            info["bitdepth"] = getattr(first, "itemsize", 1) * 8
    return Image(iterator, info)


fromarray = from_array


class Image:
    def __init__(self, rows, info):
        self.rows = rows
        self.info = info

    def save(self, file):
        with open(file, "wb") as stream:
            self.write(stream)

    def stream(self):
        self.rows = list(self.rows)

    def write(self, file):
        Writer(**self.info).write(file, self.rows)


def decompress(data_blocks):
    decompressor = zlib.decompressobj()
    for block in data_blocks:
        yield bytearray(decompressor.decompress(block))
    yield bytearray(decompressor.flush())


__all__ = [
    "ChunkError",
    "Default",
    "Error",
    "FormatError",
    "Image",
    "ProtocolError",
    "Reader",
    "Resolution",
    "Writer",
    "decompress",
    "from_array",
    "fromarray",
    "signature",
    "write_chunk",
    "write_chunks",
]
