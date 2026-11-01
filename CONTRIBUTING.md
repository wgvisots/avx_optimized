# Contributing to avx_optimized

## Source layout

```
include/avx_optimized/   public headers (the API reference lives in their comments)
src/kernels.h            the kernel table each instruction set implements, and the rules for ISA files
src/kernels_scalar.cpp   portable reference kernels
src/kernels_avx2.cpp     AVX2 + FMA kernels      (compiled with -mavx2 -mfma)
src/kernels_avx512.cpp   AVX-512 kernels         (compiled with -mavx512f/bw/dq/vl)
src/cpu.cpp              CPU detection and dispatch
src/vectorized.cpp       public functions that forward to the active kernel table
src/complex_fir_filter.cpp
test/                    GoogleTest suite, run against every supported instruction set
bench/                   benchmark program
examples/                standalone downstream project
cmake/                   dependency lookup, package config, the ISA symbol check
```

A call goes **public function → `activeKernels()` → kernel table entry → kernel**. Kernel table
entries are named after the public functions they implement, so `avx_optimized::magnitude()` lands
in `KernelTable::magnitude`, implemented by a `magnitude()` in each `kernels_*.cpp`.
`kernels_avx2.cpp` and `kernels_avx512.cpp` share the same structure (masked tails, loop skeletons,
element-wise helpers, then the kernels), so they can be read side by side.

## Adding a function

1. Declare it in the right public header with a doc comment: what it computes, units of `count`,
   aliasing rules, special values, and whether it is exact. Declare its span form too, with the
   size rule between its arrays. The docs build fails on any undocumented public declaration.
2. Add a slot to `KernelTable` in `src/kernels.h`.
3. Implement it in `kernels_scalar.cpp`, `kernels_avx2.cpp` and `kernels_avx512.cpp`, and add it to
   each file's table.
4. Forward to it from the public function (usually `src/vectorized.cpp`), and implement the span
   form there as a size check followed by a call to the pointer form.
5. Add a test that compares it with a double-precision reference over `test_helpers::testSizes()`,
   with misaligned `GuardedBuffer`s. Tests derived from `test_helpers::IsaTest` automatically run
   against every instruction set the machine supports. In `test/test_spans.cpp`, check that the span
   form matches the pointer form and that mismatched sizes throw.
6. Add it to the README's quick reference, and to `bench/benchmark.cpp`.

## Rules for the ISA kernel files

These are spelled out at the top of `src/kernels.h`. In short: in `kernels_avx2.cpp` and
`kernels_avx512.cpp`, use only intrinsics, plain arithmetic and helpers in an anonymous namespace,
and no `std::` functions or templates.

An inline function compiled with AVX-512 can otherwise be merged by the linker with a baseline copy
of itself and crash CPUs without AVX-512. The `avx_optimized_isa_object_symbols` test inspects the
compiled objects and fails if that becomes possible.

Kernels must handle every `count` (including 0) and must never form a pointer outside their arrays.
The tests check every tail length with guard zones around each buffer, and CI runs them under
AddressSanitizer.

## Style

- **Formatting:** run `clang-format -i` (version 18 or newer) on changed files. CI checks it.
  `.clang-format` lays code out as tables: consecutive declarations, assignments and trailing
  comments line up in columns, and lists that wrap put one item per line. A blank line or a comment
  line ends a table, so use blank lines to group related lines. Use `// clang-format off` / `on`
  only for layouts it can't express, such as a matrix of constants.
- **Names in full:** no `using namespace`, `using` declarations or namespace aliases. Write
  `avx_optimized::`, `std::` and so on every time, so every name says where it comes from. Inside a
  namespace's own code, names from that same namespace may stay unqualified.
- **Naming:** `camelCase` for functions and variables, `PascalCase` for types, `kPascalCase` for
  constants.

## Testing

```sh
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
cmake --preset sanitize && cmake --build --preset sanitize && ctest --preset sanitize
```

**Testing AVX-512 without AVX-512 hardware:** run the tests under the
[Intel Software Development Emulator](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html):

```sh
sde64 -spr -- build/release/test/avx_optimized_tests    # Sapphire Rapids: AVX-512
sde64 -hsw -- build/release/test/avx_optimized_tests    # Haswell: AVX2 only
sde64 -snb -- build/release/test/avx_optimized_tests    # Sandy Bridge: no AVX2, so scalar
```

CI runs the tests on several emulated CPUs for every push.

**Comparing instruction sets by hand:** `AVX_OPTIMIZED_ISA=scalar ./your_program` (or `avx2`, `avx512`).

## API reference

Configure with `-DAVX_OPTIMIZED_BUILD_DOCS=ON` (requires Doxygen), then build the
`avx_optimized_docs` target. The HTML ends up in `<build>/docs/html/index.html`.
