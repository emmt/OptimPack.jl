# User visible changes in `OptimPack`

## Unreleased

The main changes in this major release of `OptimPack` are:

- Most algorithms are implemented in pure Julia (except Powell's methods) for the following
  benefits:

  - Algorithms in Julia for large scale problems should run faster and can deal with
    variables stored in GPU.

  - Algorithms in Julia can deal with numbers with units for the variables and the objective
    function.

- A few new algorithms are provided, notably the STEP method for global optimization of a
  univariate function and a version of the Nelder-Mead (*"simplex"*) algorithm.

- Algorithms for searching a zero, a minimum, or a maximum of a univariate function now
  return an interval of confidence for the solution.

- The APIs of the algorithms have changed to be more similar and to provide more information
  on return.

- Most optimization algorithms implement a common API (public but not exported), based on
  the allocation of a context that depends on the method and on two functions:
  `OptimPack.solve!` to solve the optimization problem and `OptimPack.configure!` to modify
  the settings. The purpose of these changes is to let one solve several similar problems
  (although one problem at a time) with as few additional allocations as possible.

- Optimization algorithms can minimize or maximize the objective function.

# Breaking changes

- `fzero`, `fmin`, and `fmax` return a 5-tuple: `(x, fx, lo, hi, nf)` where `x` is the
  (approximate) solution, `fx = f(x)`, `lo` and `hi` are lower and upper bounds for the
  solution, and `nf` is the number of calls to `f`.

- In `spg2` and `spg2!` have been renamed `spg` and `spg!` to follow other implementations.

- In `spg` and `spg!` the function `prj!` implementing the projection onto the convex
  feasible set takes a single argument `x`. On entry of `prj!`, `x` contains the
  unconstrained variables. On return of `prj!`, `x` is overwritten by the variables
  projected onto the convex set.

- The API of Powell's methods `cobyla`, `bobyqa`, and `newuoa` has changed to be more
  similar to other methods of the package. These methods require to load artifact
  `OptimPack_jll` (so that `OptimPack` only weakly depends on this artifact).

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

- Variables and constraints may be multi-dimensional arrays in Powell's methods `cobyla`,
  `bobyqa`, and `newuoa`.

- API for Powell's methods (`bobyqa`, `cobyla`, and `newuoa`) have been unified with most
  parameters specified by keywords. The returned status implements properties like
  `status.reason` to retrieve error message in case of failure. The new API makes use of
  context to save allocations.

- Many algorithms can be simply tested with [`CUTEst`](https://github.com/JuliaSmoothOptimizers/CUTEst.jl).

## Version 1.1.0

- Optional argument `mem` is now a keyword in methods `vmlmb` and `vmlmb!`.
- Method `fmin_global` has been deprecated in favor of `OptimPack.BraDi.minimize`.
- By default, installation is done with precompiled libraries.
- Compatible with OptimPack 3.1 whose functionalities have been split in 3 libraries.

## Version 1.0.0

- Add compatibility with Julia ≥ 0.7.


## Version 0.3.0

- Add compatibility with C library OptimPack 3.0.
