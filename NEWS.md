# User visible changes in `OptimPack`

## Unreleased

The main changes in this major release of `OptimPack` are:

- Most algorithms are implemented in pure Julia (except Powell's methods) for the following
  benefits:

  - Algorithms in Julia for large scale problems should run faster and can deal with
    variables stored in GPU.

# Breaking changes

- `fzero`, `fmin`, and `fmax` return a 5-tuple: `(x, fx, lo, hi, nf)` where `x` is the
  (approximate) solution, `fx = f(x)`, `lo` and `hi` are lower and upper bounds for the
  solution, and `nf` is the number of calls to `f`.

- The API of Powell's methods `cobyla`, `bobyqa`, and `newuoa` has changed to be more
  similar to other methods of the package. These methods require to load `OptimPack_jll` (so
  that `OptimPack` only weakly depends on this artifact).

# Added

- `simplex(f, x0, siz)` to optimize `f(x)` by Nelder-Mead *Simplex* method. The method is
  derivative-free and can use a context with all required resources to avoid further
  allocations when solving many similar problems.

- `fminbrkt` and `fmaxbrkt` to search for a local extremum in a bracketed interval.

- Aliases `Brent.maximize(f, a, b)` and `Brent.minimize(f, a, b)` to `fmin` and `fmax`.

- `BraDi.maximize(f, x)` and `BraDi.minimize(f, x)` can takes the sample numbers `x` as a
  tuple or as a variable length list of arguments. Hence, `BraDi.maximize(f, a, b)` and
  `BraDi.minimize(f, a, b)` are the same as `Brent.fmax(f, a, b)` and `Brent.fmin(f, a, b)`
  respectively except that the former evaluate the function at the endpoints `a` and `b`
  while the latter only evaluate `f` on the open interval `(a,b)`.

- `BraDi.maximize(f, x)` and `BraDi.minimize(f, x)` can deal with quantities (i.e. numbers
  with units).

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
