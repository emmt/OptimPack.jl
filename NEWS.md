# User visible changes in `OptimPack`

## Unreleased

# Breaking changes

- `fzero`, `fmin`, and `fmax` return a 5-tuple: `(x, fx, lo, hi, nf)` with `x` the
  (approximate) solution, `fx = f(x)`, `lo` and `hi` the lower and upper bounds for the
  solution, and `nf` the number of calls to `f`.

# Added

- `fminbrkt` and `fmaxbrkt` to search for a local extremum in a bracketed interval.

- `OptimPack` now depends on the artifact `OptimPack_jll`. As a result, installation is
  easier and following the evolution of the C library should be transparent.

- API for Powell's methods (`bobyqa`, `cobyla`, and `newuoa`) have been unified with most
  parameters specified by keywords. The returned status implements properties like
  `status.reason` to retrieve error message in case of failure. The new API makes use of
  context to save allocations.

- Optimization algorithms may use a context storing all settings and work-spaces
  to avoid allocations (and thus garbage collection) when running the same algorithm
  several times.


## Version 1.1.0

- Optional argument `mem` is now a keyword in methods `vmlmb` and `vmlmb!`.
- Method `fmin_global` has been deprecated in favor of `OptimPack.BraDi.minimize`.
- By default, installation is done with precompiled libraries.
- Compatible with OptimPack 3.1 whose functionalities have been split in 3 libraries.

## Version 1.0.0

- Add compatibility with Julia ≥ 0.7.


## Version 0.3.0

- Add compatibility with C library OptimPack 3.0.
