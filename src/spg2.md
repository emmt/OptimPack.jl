## Spectral Projected Gradient Method

The function, `spg2` implements the Spectral Projected Gradient Method
(Version 2: "*continuous projected gradient direction*") to find the local
minimizers of a given function with convex constraints, described in
references [1] and [2] below.  The calling sequence is:
````{.jl}
ws = spg2(fg!, prj!, x0, m)
````
The user must supply the functions `fg!` and `prj!` to evaluate the
objective function (and its gradient) and to project an arbitrary point
onto the feasible region.  These functions must be defined as:
````{.jl}
function fg!(x, g)
  g[...] = gradient_at(x)
  return function_value_at(x)
end
function prj!(xp, x)
  xp[...] = projection_of(x)
end
````
Argument `x0` is the initial solution and argument `m` is the number of
previous function values to be considered in the nonmonotone line search.
If `m <= 1`, then a monotone line search with Armijo-like stopping criterion
will be used.

For efficiency reasons (avoiding memory allocate/copy/free), it is important
that the gradient and the projected variables be stored in the provided
arrays (`g` and `xp` respectively).  The projector must be able to perform
*in-place* operation (*i.e.*, with `xp` and `x` being the same array).

The following keywords are available:

* `eps1` - Algorithm stops when the infinite-norm of the projected gradient
           is less or equal `eps1`.
* `eps2` - Algorithm stops when the Euclidean norm of the projected
           gradient is less or equal `eps2`.
* `eta`  - Scaling parameter for the gradient, the projected gradient is
           computed as `(x - prj(x - eta*g))/eta` (with `g` the gradient at
           `x`) instead of `x - prj(x - g)` which corresponds to the
           default behavior (same as if `eta=1`) and is usually used in
           methodological publications although it does not scale correctly
           (for instance, if you make a change of variables or simply
           multiply the function by some factor.
* `maxit` - Maximum number of iterations.
* `maxfc` - Maximum number of function evaluations.
* `verb` - If true, print some information at each iteration.
* `printer` - If specified, a function to print or display some information
           at each iteration.  This subroutine will be called with a single
           argument which is a the same as the returned result except that
           it member `x` is set with the variables at the current iteration
           instead of the final one.

The result `ws` has the following members:

* `ws.x`      - The current or final variables.
* `ws.f`      - The function value at `x`.
* `ws.pginfn` - The infinite norm of the gradient at `x`.
* `ws.pgtwon` - The Euclidean norm of the gradient at `x`.
* `ws.xbest`  - The approximation to the local minimizer.
* `ws.fbest`  - The function value at `xbest`.
* `ws.iter`   - The number of iterations.
* `ws.fcnt`   - The number of function (and gradient) evaluations.
* `ws.pcnt`   - The number of projections.
* `ws.status` - Termination parameter:
  * `OptimPack.SPG2_CONVERGENCE_WITH_INFNORM` = convergence with projected gradient infinite-norm,
  * `OptimPack.SPG2_CONVERGENCE_WITH_TWONORM` = convergence with projected gradient Euclidean norm,
  * `OptimPack.SPG2_TOO_MANY_ITERATIONS` = too many iterations,
  * `OptimPack.SPG2_TOO_MANY_EVALUATIONS` = too many function evaluations.

Note that the final iteration may not be the best one.


# REFERENCES:

1. E. G. Birgin, J. M. Martinez, and M. Raydan, "*Nonmonotone spectral
   projected gradient methods on convex sets*", SIAM Journal on
   Optimization **10**, pp. 1196-1211 (2000).

2. E. G. Birgin, J. M. Martinez, and M. Raydan, "*SPG: software for
   convex-constrained optimization*", ACM Transactions on Mathematical
   Software (TOMS) **27**, pp. 340-349 (2001).
