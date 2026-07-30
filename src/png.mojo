from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.runtime import initialize_runtime
from std.sys.info import num_physical_cores, simd_width_of


comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U16Ptr = UnsafePointer[UInt16, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_FILTER_BYTES = 262144
comptime PARALLEL_COPY_BYTES = 1048576
comptime MAX_GPU_BYTES = 2 * 1024 * 1024 * 1024 - 1


def valid_buffers(
    src_addr: Int, src_len: Int, expected_src: Int,
    dst_addr: Int, dst_len: Int, expected_dst: Int,
) -> Bool:
    return (
        src_addr != 0 and dst_addr != 0
        and src_len >= 0 and dst_len >= 0
        and expected_src >= 0 and expected_dst >= 0
        and src_len == expected_src and dst_len == expected_dst
    )


def paeth(a: Int, b: Int, c: Int) -> Int:
    var p = a + b - c
    var pa = abs(p - a)
    var pb = abs(p - b)
    var pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def residual_cost(value: Int) -> Int:
    var wrapped = value & 255
    if wrapped < 128:
        return wrapped
    return 256 - wrapped


def vector_residual_cost[
    width: Int
](
    raw: SIMD[DType.uint8, width],
    predictor: SIMD[DType.uint8, width],
) -> Int:
    var residual = raw - predictor
    var negative = SIMD[DType.uint8, width](0) - residual
    return Int(
        min(residual, negative).cast[DType.uint32]().reduce_add()
    )


def paeth_vector[
    width: Int
](
    a_u8: SIMD[DType.uint8, width],
    b_u8: SIMD[DType.uint8, width],
    c_u8: SIMD[DType.uint8, width],
) -> SIMD[DType.uint8, width]:
    var a = a_u8.cast[DType.int16]()
    var b = b_u8.cast[DType.int16]()
    var c = c_u8.cast[DType.int16]()
    var p = a + b - c
    var pa = abs(p - a)
    var pb = abs(p - b)
    var pc = abs(p - c)
    var predictor = pb.le(pc).select(b, c)
    predictor = (pa.le(pb) & pa.le(pc)).select(a, predictor)
    return predictor.cast[DType.uint8]()


def filter_score(src: U8Ptr, previous: U8Ptr, row_offset: Int, row_bytes: Int,
                 bpp: Int, filter_type: Int, has_previous: Bool) -> Int:
    var score = 0
    var prefix = min(bpp, row_bytes)
    var x = 0
    while x < prefix:
        var raw = Int(src[row_offset + x])
        var left = 0
        var up = 0
        var upper_left = 0
        if x >= bpp:
            left = Int(src[row_offset + x - bpp])
        if has_previous:
            up = Int(previous[x])
            if x >= bpp:
                upper_left = Int(previous[x - bpp])
        var predictor = 0
        if filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = paeth(left, up, upper_left)
        score += residual_cost(raw - predictor)
        x += 1

    var vector_end = prefix + ((row_bytes - prefix) // W) * W
    while x < vector_end:
        var raw = src.load[width=W](row_offset + x)
        var left = src.load[width=W](row_offset + x - bpp)
        var up = SIMD[DType.uint8, W](0)
        var upper_left = SIMD[DType.uint8, W](0)
        if has_previous:
            up = previous.load[width=W](x)
            upper_left = previous.load[width=W](x - bpp)
        var predictor = SIMD[DType.uint8, W](0)
        if filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (
                (
                    left.cast[DType.uint16]()
                    + up.cast[DType.uint16]()
                )
                >> 1
            ).cast[DType.uint8]()
        elif filter_type == 4:
            predictor = paeth_vector(left, up, upper_left)
        score += vector_residual_cost(raw, predictor)
        x += W

    while x < row_bytes:
        var raw = Int(src[row_offset + x])
        var left = Int(src[row_offset + x - bpp])
        var up = 0
        var upper_left = 0
        if has_previous:
            up = Int(previous[x])
            upper_left = Int(previous[x - bpp])
        var predictor = 0
        if filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = paeth(left, up, upper_left)
        score += residual_cost(raw - predictor)
        x += 1
    return score


def filter_one_row(
    src: U8Ptr,
    dst: U8Ptr,
    y: Int,
    row_bytes: Int,
    bpp: Int,
):
    var row_offset = y * row_bytes
    var dst_offset = y * (row_bytes + 1)
    var previous = src
    if y > 0:
        previous = src + row_offset - row_bytes
    var has_previous = y > 0
    var best_type = 0
    var best_score = filter_score(
        src, previous, row_offset, row_bytes, bpp, 0, has_previous
    )
    for filter_type in range(1, 5):
        var score = filter_score(
            src, previous, row_offset, row_bytes, bpp, filter_type, has_previous
        )
        if score < best_score:
            best_score = score
            best_type = filter_type
    dst[dst_offset] = UInt8(best_type)

    var prefix = min(bpp, row_bytes)
    var x = 0
    while x < prefix:
        var raw = Int(src[row_offset + x])
        var up = 0
        if has_previous:
            up = Int(previous[x])
        var predictor = 0
        if best_type == 2:
            predictor = up
        elif best_type == 3:
            predictor = up // 2
        elif best_type == 4:
            predictor = paeth(0, up, 0)
        dst[dst_offset + 1 + x] = UInt8((raw - predictor) & 255)
        x += 1

    var vector_end = prefix + ((row_bytes - prefix) // W) * W
    while x < vector_end:
        var raw = src.load[width=W](row_offset + x)
        var left = src.load[width=W](row_offset + x - bpp)
        var up = SIMD[DType.uint8, W](0)
        var upper_left = SIMD[DType.uint8, W](0)
        if has_previous:
            up = previous.load[width=W](x)
            upper_left = previous.load[width=W](x - bpp)
        var predictor = SIMD[DType.uint8, W](0)
        if best_type == 1:
            predictor = left
        elif best_type == 2:
            predictor = up
        elif best_type == 3:
            predictor = (
                (
                    left.cast[DType.uint16]()
                    + up.cast[DType.uint16]()
                )
                >> 1
            ).cast[DType.uint8]()
        elif best_type == 4:
            predictor = paeth_vector(left, up, upper_left)
        dst.store(dst_offset + 1 + x, raw - predictor)
        x += W

    while x < row_bytes:
        var raw = Int(src[row_offset + x])
        var left = Int(src[row_offset + x - bpp])
        var up = 0
        var upper_left = 0
        if has_previous:
            up = Int(previous[x])
            upper_left = Int(previous[x - bpp])
        var predictor = 0
        if best_type == 1:
            predictor = left
        elif best_type == 2:
            predictor = up
        elif best_type == 3:
            predictor = (left + up) // 2
        elif best_type == 4:
            predictor = paeth(left, up, upper_left)
        dst[dst_offset + 1 + x] = UInt8((raw - predictor) & 255)
        x += 1


@export("mpp_filter_rows")
def mpp_filter_rows(src_addr: Int, src_len: Int, dst_addr: Int, dst_len: Int,
                    height: Int, row_bytes: Int, bpp: Int) abi("C") -> Int:
    if height <= 0 or row_bytes <= 0 or bpp <= 0 or bpp > row_bytes:
        return 0
    if height > (MAX_GPU_BYTES // row_bytes):
        return 0
    var src_size = height * row_bytes
    if height > (MAX_GPU_BYTES // (row_bytes + 1)):
        return 0
    var dst_size = height * (row_bytes + 1)
    if not valid_buffers(
        src_addr, src_len, src_size, dst_addr, dst_len, dst_size
    ):
        return 0
    var src = U8Ptr(unsafe_from_address=src_addr)
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    if height * row_bytes >= PARALLEL_FILTER_BYTES and height > 1:
        initialize_runtime()
        var workers = min(height, num_physical_cores())
        var rows_per_worker = (height + workers - 1) // workers

        @parameter
        @__copy_capture(
            src, dst, height, row_bytes, bpp, rows_per_worker
        )
        def process_rows(worker: Int):
            var first = worker * rows_per_worker
            var last = min(first + rows_per_worker, height)
            for y in range(first, last):
                filter_one_row(src, dst, y, row_bytes, bpp)

        parallelize[process_rows](workers)
    else:
        for y in range(height):
            filter_one_row(src, dst, y, row_bytes, bpp)
    return 1


def filter_rows_gpu_kernel(
    src: U8Ptr,
    dst: U8Ptr,
    height: Int,
    row_bytes: Int,
    bpp: Int,
):
    var y = global_idx.x
    if y < height:
        filter_one_row(src, dst, y, row_bytes, bpp)


@export("mpp_filter_rows_gpu")
def mpp_filter_rows_gpu(
    src_addr: Int,
    src_len: Int,
    dst_addr: Int,
    dst_len: Int,
    height: Int,
    row_bytes: Int,
    bpp: Int,
) abi("C") -> Int:
    try:
        if height <= 0 or row_bytes <= 0 or bpp <= 0 or bpp > row_bytes:
            return 0
        if height > (MAX_GPU_BYTES // row_bytes):
            return 0
        var src_size = height * row_bytes
        if height > (MAX_GPU_BYTES // (row_bytes + 1)):
            return 0
        var dst_size = height * (row_bytes + 1)
        if src_size > MAX_GPU_BYTES - dst_size:
            return 0
        if not valid_buffers(
            src_addr, src_len, src_size, dst_addr, dst_len, dst_size
        ):
            return 0
        var src = U8Ptr(unsafe_from_address=src_addr)
        var dst = U8Ptr(unsafe_from_address=dst_addr)
        var ctx = DeviceContext()
        var device_src = ctx.enqueue_create_buffer[DType.uint8](src_size)
        var device_dst = ctx.enqueue_create_buffer[DType.uint8](dst_size)
        ctx.enqueue_copy(device_src, src)
        comptime block_size = 128
        var grid_size = (height + block_size - 1) // block_size
        ctx.enqueue_function[filter_rows_gpu_kernel](
            device_src,
            device_dst,
            height,
            row_bytes,
            bpp,
            grid_dim=grid_size,
            block_dim=block_size,
        )
        ctx.enqueue_copy(dst, device_dst)
        ctx.synchronize()
        return 1
    except:
        return 0


def copy_none_row(
    src: U8Ptr,
    dst: U8Ptr,
    y: Int,
    row_bytes: Int,
):
    var src_offset = y * (row_bytes + 1) + 1
    var dst_offset = y * row_bytes
    var vector_end = (row_bytes // W) * W
    var x = 0
    while x < vector_end:
        dst.store(dst_offset + x, src.load[width=W](src_offset + x))
        x += W
    while x < row_bytes:
        dst[dst_offset + x] = src[src_offset + x]
        x += 1


@export("mpp_unfilter_rows")
def mpp_unfilter_rows(src_addr: Int, src_len: Int, dst_addr: Int, dst_len: Int,
                      height: Int, row_bytes: Int, bpp: Int) abi("C") -> Int:
    if height <= 0 or row_bytes <= 0 or bpp <= 0 or bpp > row_bytes:
        return 0
    if height > (MAX_GPU_BYTES // row_bytes):
        return 0
    var dst_size = height * row_bytes
    if height > (MAX_GPU_BYTES // (row_bytes + 1)):
        return 0
    var src_size = height * (row_bytes + 1)
    if not valid_buffers(
        src_addr, src_len, src_size, dst_addr, dst_len, dst_size
    ):
        return 0
    var src = U8Ptr(unsafe_from_address=src_addr)
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    var all_none = True
    for y in range(height):
        var src_offset = y * (row_bytes + 1)
        var filter_type = Int(src[src_offset])
        if filter_type < 0 or filter_type > 4:
            return 0
        if filter_type != 0:
            all_none = False

    if all_none:
        if height * row_bytes >= PARALLEL_COPY_BYTES and height > 1:
            initialize_runtime()
            var workers = min(height, num_physical_cores())
            var rows_per_worker = (height + workers - 1) // workers

            @parameter
            @__copy_capture(src, dst, height, row_bytes, rows_per_worker)
            def copy_rows(worker: Int):
                var first = worker * rows_per_worker
                var last = min(first + rows_per_worker, height)
                for y in range(first, last):
                    copy_none_row(src, dst, y, row_bytes)

            parallelize[copy_rows](workers)
        else:
            for y in range(height):
                copy_none_row(src, dst, y, row_bytes)
        return 1

    for y in range(height):
        var src_offset = y * (row_bytes + 1)
        var row_offset = y * row_bytes
        var filter_type = Int(src[src_offset])
        if filter_type == 0:
            copy_none_row(src, dst, y, row_bytes)
            continue
        if filter_type == 2:
            var vector_end = (row_bytes // W) * W
            var x = 0
            while x < vector_end:
                var filtered = src.load[width=W](src_offset + 1 + x)
                var up = SIMD[DType.uint8, W](0)
                if y > 0:
                    up = dst.load[width=W](row_offset - row_bytes + x)
                dst.store(row_offset + x, filtered + up)
                x += W
            while x < row_bytes:
                var up = 0
                if y > 0:
                    up = Int(dst[row_offset - row_bytes + x])
                dst[row_offset + x] = UInt8(
                    (Int(src[src_offset + 1 + x]) + up) & 255
                )
                x += 1
            continue
        for x in range(row_bytes):
            var filtered = Int(src[src_offset + 1 + x])
            var left = 0
            var up = 0
            var upper_left = 0
            if x >= bpp:
                left = Int(dst[row_offset + x - bpp])
            if y > 0:
                up = Int(dst[row_offset - row_bytes + x])
                if x >= bpp:
                    upper_left = Int(dst[row_offset - row_bytes + x - bpp])
            var predictor = 0
            if filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                predictor = paeth(left, up, upper_left)
            dst[row_offset + x] = UInt8((filtered + predictor) & 255)
    return 1


@export("mpp_pack_bits")
def mpp_pack_bits(src_addr: Int, src_len: Int, dst_addr: Int, dst_len: Int,
                  height: Int, samples_per_row: Int, bitdepth: Int) abi("C") -> Int:
    if bitdepth != 1 and bitdepth != 2 and bitdepth != 4:
        return 0
    if height <= 0 or samples_per_row <= 0 or samples_per_row > MAX_GPU_BYTES:
        return 0
    var samples_per_byte = 8 // bitdepth
    var row_bytes = (samples_per_row * bitdepth + 7) // 8
    if height > (MAX_GPU_BYTES // samples_per_row):
        return 0
    var src_size = height * samples_per_row
    if height > (MAX_GPU_BYTES // row_bytes):
        return 0
    var dst_size = height * row_bytes
    if not valid_buffers(
        src_addr, src_len, src_size, dst_addr, dst_len, dst_size
    ):
        return 0
    var src = U8Ptr(unsafe_from_address=src_addr)
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    var mask = (1 << bitdepth) - 1
    for y in range(height):
        for byte_index in range(row_bytes):
            var packed = 0
            for lane in range(samples_per_byte):
                var sample_index = byte_index * samples_per_byte + lane
                var sample = 0
                if sample_index < samples_per_row:
                    sample = Int(src[y * samples_per_row + sample_index])
                    if sample > mask:
                        return 0
                packed |= sample << (8 - bitdepth * (lane + 1))
            dst[y * row_bytes + byte_index] = UInt8(packed)
    return 1


@export("mpp_unpack_bits")
def mpp_unpack_bits(src_addr: Int, src_len: Int, dst_addr: Int, dst_len: Int,
                    height: Int, samples_per_row: Int, bitdepth: Int) abi("C") -> Int:
    if bitdepth != 1 and bitdepth != 2 and bitdepth != 4:
        return 0
    if height <= 0 or samples_per_row <= 0 or samples_per_row > MAX_GPU_BYTES:
        return 0
    var row_bytes = (samples_per_row * bitdepth + 7) // 8
    if height > (MAX_GPU_BYTES // row_bytes):
        return 0
    var src_size = height * row_bytes
    if height > (MAX_GPU_BYTES // samples_per_row):
        return 0
    var dst_size = height * samples_per_row
    if not valid_buffers(
        src_addr, src_len, src_size, dst_addr, dst_len, dst_size
    ):
        return 0
    var src = U8Ptr(unsafe_from_address=src_addr)
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    var mask = (1 << bitdepth) - 1
    for y in range(height):
        for x in range(samples_per_row):
            var bit_offset = x * bitdepth
            var byte_index = bit_offset // 8
            var shift = 8 - bitdepth - (bit_offset & 7)
            dst[y * samples_per_row + x] = UInt8(
                (Int(src[y * row_bytes + byte_index]) >> shift) & mask
            )
    return 1


@export("mpp_pack_u16be")
def mpp_pack_u16be(src_addr: Int, src_len: Int, dst_addr: Int, dst_len: Int,
                   count: Int) abi("C") -> Int:
    if count <= 0 or count > (MAX_GPU_BYTES // 2):
        return 0
    if not valid_buffers(
        src_addr, src_len, count * 2, dst_addr, dst_len, count * 2
    ):
        return 0
    var src = U16Ptr(unsafe_from_address=src_addr)
    var dst = U8Ptr(unsafe_from_address=dst_addr)
    for i in range(count):
        var value = Int(src[i])
        dst[2 * i] = UInt8(value >> 8)
        dst[2 * i + 1] = UInt8(value & 255)
    return 1


@export("mpp_unpack_u16be")
def mpp_unpack_u16be(src_addr: Int, src_len: Int, dst_addr: Int, dst_len: Int,
                     count: Int) abi("C") -> Int:
    if count <= 0 or count > (MAX_GPU_BYTES // 2):
        return 0
    if not valid_buffers(
        src_addr, src_len, count * 2, dst_addr, dst_len, count * 2
    ):
        return 0
    var src = U8Ptr(unsafe_from_address=src_addr)
    var dst = U16Ptr(unsafe_from_address=dst_addr)
    for i in range(count):
        dst[i] = UInt16((Int(src[2 * i]) << 8) | Int(src[2 * i + 1]))
    return 1
