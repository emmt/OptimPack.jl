# OptimPack.jl

[![Build Status](https://travis-ci.org/emmt/OptimPack.jl.svg?branch=master)](https://travis-ci.org/emmt/OptimPack.jl)

OptimPack.jl is the Julia interface to
[OptimPack](https://github.com/emmt/OptimPack), a library for solving
large scale optimization problems.


## Installation

From a Julia session, type the following commands:
```julia
Pkg.add("BinDeps")
Pkg.clone("https://github.com/emmt/OptimPack.jl.git")
Pkg.build("OptimPack")
```


## Unconstrained Minimization of a Nonlinear Smooth Function

There are two methods in OptimPack to minimize a nonlinear smooth
multivariate function without constraints: non-linear conjugate gradient
(NLCG) and limited memory variable metric method (VMLM).

The easiest way to use these minimizers is to provide a Julia function, say
`fg!`, which is in charge of computing the value of the function and its
gradient for given variables.  This function must have the form:
```julia
function fg!(x, g)
   g[...] = ... # store the gradient of the function
   f = ...      # compute the function value
   return f     # return the function value
end
```
where the arguments `x` and `g` are Julia arrays (same types and
dimensions) with, on entry, `x` storing the variables and, on exit, `g`
storing the gradient.  The user defined function shall return the function
value.


## Nonlinear Conjugate Gradient (NLCG)

The solution `x` can be computed by one of the implemented nonlinear
conjugate gradient methods with:
```julia
x = nlcg(fg!, x0, method)
```
where `x0` gives the initial value of the variables (as well as the data
type and dimensions of the solution).  `x0` is a Julia dense array with any
dimensions and with elements of type `Float64` or `Float32`.  Argument
`method` is optional and can be used to choose among the different implemented
methods (see below).

The keyword `verb` can be set true to print information at each iteration.
Other keywords are described in the following sub-sections.


### Method Settings

The different nonlinear conjugate gradient methods mainly differ by the way
they compute the search direction.  The conjugate gradient iteration
writes:
```julia
x_{k+1} = x_{k} + alpha_{k} * d_{k}
```
with `alpha_{k}` the step length and where the search direction `d_{k}` is
derived from the gradient `g(x_{k})` of the objective function at the
current point `x_{k}` and from the previous search direction `d_{k-1}` by
an *update rule* which depends on the specific method.  Typically:
```julia
d_{k} = -g(x_{k}) + beta_{k} * d_{k-1}
```
where `beta_{k}` is computed following different recipes.  To choose which
recipe to use, the value of the `method` argument can be set to one of the
following values:

- `OptimPack.NLCG_FLETCHER_REEVES` for Fletcher & Reeve method;
- `OptimPack.NLCG_HESTENES_STIEFEL` for Hestenes & Stiefel method;
- `OptimPack.NLCG_POLAK_RIBIERE_POLYAK` for Polak, Ribière & Polyak method;
- `OptimPack.NLCG_FLETCHER` for Fletcher "*Conjugate Descent*" method;
- `OptimPack.NLCG_LIU_STOREY` for Liu & Storey method;
- `OptimPack.NLCG_DAI_YUAN` for Dai & Yuan method;
- `OptimPack.NLCG_PERRY_SHANNO` for Perry & Shanno update rule;
- `OptimPack.NLCG_HAGER_ZHANG` for Hager & Zhang method.

The above values can be bitwise or'ed with the following bits:

- `OptimPack.NLCG_POWELL` to force parameter `beta` to be nonnegative;
- `OptimPack.NLCG_SHANNO_PHUA` to guess the step length following the
  prescription of Shanno & Phua.

For instance:
```julia
method = OptimPack.NLCG_POLAK_RIBIERE_POLYAK | OptimPack.NLCG_POWELL
```
merely corresponds to PRP+ algorithm by Polak, Ribière & Polyak; while:
```julia
method = OptimPack.NLCG_PERRY_SHANNO | OptimPack.NLCG_SHANNO_PHUA
```
merely corresponds to the nonlinear conjugate gradient method implemented
in CONMIN (Shanno & Phua, 1980).

The default settings for nonlinear conjugate gradient is:
```julia
const OptimPack.NLCG_DEFAULT  = (OptimPack.NLCG_HAGER_ZHANG | OptimPack.NLCG_SHANNO_PHUA)
```


### Stopping Criteria

The nonlinear conjugate gradient methods are iterative algorithms, the
convergence is assumed to be achieved when the Euclidean norm of the
gradient is smaller than a threshold.  In pseudo-code, the criterion is:
```julia
||g(x)|| <= max(0, gatol, grtol*||g(x0)||)
```
where `||g(x)||` is the Euclidean norm of the gradient at the current
solution `x`, `||g(x0)||` is the Euclidean norm of the initial gradient at
`x0`, `gatol` is an absolute threshold parameter and `grtol` is a relative
threshold parameter.  The keywords `gatol` and `grtol` can be used to
specify other values for these parameters than the default ones which are
`gatol = 0.0` and `grtol = 1E-6`.

It may be desirable to limit the time spent by the algorithm.  To that end,
the keywords `maxiter` and `maxeval` are available to specify the maximum
number of iterations and evaluations of the algorithm respectively.  Their
default values is `-1` which means that there are no restrictions.  For now,
the algorithm can only be safely stopped at an acceptable iterate, thus the
maximum number of allowed function evaluations may slightly exceed the
value of `maxeval`.


### Line Search Settings

The keyword `lnsrch` can be used to specify another line search method than
the default one:
```julia
x = nlcg(fg!, x0, method, lnsrch=ls)
```
where `ls` is one of the implemented line search methods:
```julia
ls = OptimPack.ArmijoLineSearch(ftol=...)
ls = OptimPack.MoreThuenteLineSearch(ftol=..., gtol=..., xtol=...)
ls = OptimPack.NonmonotoneLineSearch(mem=..., ftol=..., amin=..., amax=...)
```
with `ftol` the tolerance on the function reduction for the Armijo or first
Wolfe condition, `gtol` the tolerance on the gradient for the second
(strong) Wolfe condition, `xtol` the relative precision for the step length
(set to the machine relative precision by default), `mem` the number of
previous steps to remember for the nonmonotone line search, keywords `amin`
and `amax` set the lower steplength bound and the upper steplength relative
bound to trigger bissection in nonmonotone line search.  By default, the
values used in SPG2 are used for the nonmonotone line search: `mem = 10`,
`ftol = 1E-4`, `amin = 0.1` and `amax = 0.9`.

The line search is safeguarded by imposing lower and upper bounds on the
step.  In `nlcg` and `vmlm`, keywords `stpmin` and `stpmax` can be used to
specify the step bounds relatively to the size of the first step for each
line search.  Their default values are: `stpmin = 1E-20` and `stpmax =
1E+20`; if specified, they must be such that: `0 <= stpmin < stpmax`.


## Variable Metric with Limited Memory (VMLM)

Alternatively, the solution `x` can be computed by a limited memory version
of the variable metric method (implementing BFGS updates) with:
```julia
x = vmlm(fg!, x0, m)
```
where the optional argument `m` is the number of previous steps to memorize
(by default `m = 3`) while other arguments have the same meaning as for
`nlcg`.

Keywords `verb`, `gatol`, `grtol`, `lnsrch`, `stpmin` and `stpmax` can also
be specified for `vmlm` and have the same meaning as for `nlcg`.

In addition to these keywords, you can specify how to scale the inverse
Hessian in variable metric method via the `scaling` keyword:
```julia
scaling = OptimPack.SCALING_NONE             # to use a unit scaling (no scaling)
scaling = OptimPack.SCALING_OREN_SPEDICATO   # to scale by: gamma1 = <s,y>/<y,y>
scaling = OptimPack.SCALING_BARZILAI_BORWEIN # to scale by: gamma2 = <s,s>/<s,y>
```
where `<s,y>` denotes the inner product between the previous step `s` and
gradient difference `y`.


## Spectral Projected Gradient Method

The spectral projected gradient (SPG2) method of Birgin, Martinez & Raydan
can be used for solving large constrained optimization problems.  The usage
of the SPG2 method is documented [here](doc/spg2.md).


## Low-level Interface

### Operations on Vectors

To create a vector space for vectors of dimensions `dims` and element type
`T`:
```julia
space = OptimPack.DenseVectorSpace(T, dims)
```
where `T` is `Float32` or `Float64` (or any type alias of these,
e.g. `Cfloat` or `Cdouble`) and `dims` is a tuple of the dimensions.

It is also possible to *wrap* a vector around a specific Julia array:
```julia
vect = OptimPack.wrap(space, arr)
```
where `space` is an OptimPack *shaped* vector space and `arr` is a Julia
array.  The element type and dimension list of the array must match those
of the vector space.  A method is available to change the contents of such
a vector:
```julia
OptimPack.wrap!(vect, arr2)
```
where `arr2` is another Julia array (the same constraints on the element
type and dimensions apply).

Note that `arr` must be a **dense array** (of type `DenseArray`) as the
elements of *shaped* vectors are supposed to be stored contiguously in
memory.  OptimPack offers the possibility to create custom vector spaces
and this will be exploited in a near futur to allow for other flavors of
Julia arrays.


### Error Management

Run-time errors throw Julia exception.

