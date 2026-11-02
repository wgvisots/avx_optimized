# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue. Use GitHub's private reporting:
on the repository's **Security** tab, choose **Report a vulnerability**.

Please include:
- the version or commit you tested,
- the function or class involved, with the arguments or input that trigger the problem,
- the CPU or instruction set it happens on (`avx_optimized::isaName(avx_optimized::activeIsa())`,
  or try `AVX_OPTIMIZED_ISA=scalar`, `avx2` or `avx512`), and
- what happens: a crash, an out-of-bounds read or write, a wrong result, and so on.

You can expect a reply within two weeks. Once a fix is ready, it is released and the report is
published as a GitHub security advisory, crediting you unless you'd rather not be named.

## Supported versions

Fixes are made on the latest release only.

## What counts as a vulnerability

- Reading or writing outside the arrays given to a function, with arguments that follow its
  documented rules.
- Undefined behavior, crashes or hangs caused by input **values**, such as NaN, infinity,
  subnormals or extreme numbers.
- Instructions the CPU doesn't support being executed (a crash with SIGILL) after dispatch has
  checked the CPU.

Not vulnerabilities, because the documentation makes them the caller's responsibility:
- Pointer-form calls with a `count` larger than the arrays. Use the span forms when sizes come
  from untrusted input; they check them.
- Overlapping arrays where the documentation forbids it, or using one filter object from several
  threads at once.
