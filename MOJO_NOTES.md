# Mojo dialect notes (verified by probe against the pinned compiler, not by docs)

Toolchain these notes describe: `mojo ==1.1.0.dev2026081105`.
Every claim below was checked by compiling it. `bin/probe-confirm.py` and
`bin/probe-hints.py` in the factory regenerate the list; re-run them after any
toolchain bump rather than trusting this file.

## 0. Two things that break everything if you forget them

- **`MODULAR_HOME` must point at the env's `share/max`.** Without it every import
  fails with `unable to locate module 'std'`, which looks like a missing package
  but is not. `pixi run` sets it for you. A bare `mojo build` does not:
  ```bash
  export MODULAR_HOME="$(dirname "$(dirname "$(which mojo)")")/share/max"
  ```
- **`mojo build -o` takes an output FILE path**, not a directory. A directory there
  fails late, at link time, with `ld: cannot open output file ... Is a directory`.

## 1. Language changes from the 0.x/1.0 dialect

- **`fn` has been removed. Use `def` for everything.** `fn` is a hard error:
  `'fn' has been removed; use 'def' instead`.
- `UnsafePointer` is deprecated in favour of **`Pointer`**. It still compiles, with
  a warning, so old code builds — but write `Pointer[T, AnyOrigin[mut=True]]`.
- `int(x)` / `float(x)` are **not builtins**. Use the type as a constructor:
  `Int(x)`, `Float64(x)`. There is no `.to_int()`, `.int()`, or `round(x).to_int()`.
- `simdwidthof[DType.float64]()` is gone. It is now
  `from std.sys import simd_width_of` then `simd_width_of[DType.float64]()`.
  A hardcoded `comptime W = 4` is always valid and is often the better choice.
- `SIMD` has no `.min()` / `.max()` methods. The free `min(a, b)` / `max(a, b)`
  work on both scalars and `SIMD` values.

## 2. Export / FFI

- `@export("symbol_name")` sits on the line above the def. The ABI is an *effect*
  before the arrow: `def f(a: Int) abi("C") -> Float64:`.
- **`@export` now REQUIRES an explicit `abi()`.** Omitting it is an error:
  `@export requires an explicit 'abi()' effect on the function`. This is stricter
  than 1.0, where it only warned.
- An `abi("C")` function **may not be `raises`**. Put the fallible work in a
  `try:` / `except:` inside the body instead.
- `@export` rejects parametric functions, including an inferred pointer origin
  (`Pointer[Float64, _]`). Annotate the origin explicitly.
- Buffers cross the C ABI as **`Int` addresses**, rebuilt inside the wrapper:
  ```mojo
  var q = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=addr)
  ```
- Pointers are NON-NULLABLE: constructing one from address 0 fails a compile-time
  constraint. Take `Int` and construct inside the branch that uses it.
- `AnyOrigin[mut=True]` is the only usable mutable origin name. `MutableAnyOrigin`
  and friends do not exist.

## 3. Memory and SIMD (all verified working)

```mojo
comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()

var i = 0
while i + W <= n:                      # vector body
    p.unsafe_store(i, p.unsafe_load[width=W](i) * 2.0)
    i += W
while i < n:                           # scalar tail
    p.unsafe_store(i, p.unsafe_load(i) * 2.0)
    i += 1
```
- `p.unsafe_load[width=W](i)` / `p.unsafe_store(i, v)` / `v.reduce_add()` all work.
- `p.unsafe_load(i)` for a scalar load; `p.unsafe_offset(i)` also works.
- `load` / `store` / positional `p[i]` / pointer `+` arithmetic all still
  compile but warn. Use the `unsafe_*` names above in new code.

## 4. Parallelism — moved to `max.algorithm`, not deleted

`parallelize` is gone from `std.algorithm` (which no longer even exports
`sort`) and there is no `std.parallelism` / `std.threading`. It was **moved,
not removed**: `from max.algorithm import parallelize` compiles and runs. The
compiler gives no "did you mean" hint, so this looks exactly like a removal.

It really is concurrent — 16 workers each sleeping 0.5 s return in ~0.6 s, not
~8 s. `initialize_runtime()` is required first; calling `parallelize` from a
`--emit shared-lib` function that skipped it segfaults. The existing
`@parameter` / `@__copy_capture` worker closures port over unchanged.

Consequence: a CPU kernel that was parallelised on the host can stay
parallelised, importing from `max.algorithm`. Do NOT write
`from std.algorithm import parallelize` and assume it works.

## 5. GPU — host API moved to `max.gpu.host`

- `from std.gpu import thread_idx` and `from std.gpu import global_idx` **work**.
- `from std.memory import stack_allocation` resolves, but its signature does not
  match the old `stack_allocation[T](n)` form — check it by compiling.
- **`DeviceContext` is not in `std`**: not in `std.gpu.host`, not in `std.gpu`,
  and the compiler suggests nothing. It was **moved, not deleted**:
  `from max.gpu.host import DeviceContext` works, along with
  `enqueue_create_buffer[DType.uint8](n)`, `enqueue_copy` in both directions,
  `enqueue_function[k](..., grid_dim=, block_dim=)` and `synchronize()`. The
  sources live under `site-packages/max/gpu/host/`; there is no `max/gpu`
  directory listing because the package is compiled.
- **Device kernels must take `DevicePassable` arguments.** An `Int` parameter
  fails deep in the pass manager with `Int and UInt do not conform to
  DevicePassable`; use `Int32` / `Int64` and widen with `Int(...)` in the body.
  `Int` arithmetic on locals inside the kernel is fine.
- `global_idx.x` does not compare cleanly against an `Int32` parameter in every
  direction; writing `var y = Int(global_idx.x)` and comparing against
  `Int(height)` is the form that compiles.

Practical rule: a GPU path is only worth writing if you can compile and run it.
Otherwise state in the README that the port is CPU-only and why. The GPU is shared
with production workloads — see the memory limits in `prompts/accel.md`.

## 6. Build

- Build cost is ~2-5s and essentially FIXED regardless of function count. Batch
  many functions into ONE compilation unit rather than compiling files separately.
- `mojo run` JITs in ~1.2s per invocation; a built shared lib + ctypes call is ~0.9us.
- `mojo build --emit shared-lib` errors if the file defines `main`.

## 7. Diagnosing a moved symbol

Do not guess module paths. A bare unknown name makes the compiler name the module
it expected:

```mojo
def _p() -> Int:
    return simd_width_of[DType.float64]()   # -> "did you mean to import it from 'std.sys'?"
```
No hint means the symbol is gone, not relocated. `bin/probe-hints.py` automates this.
