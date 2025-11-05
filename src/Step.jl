"""
    OptimPack.Step

Implement the STEP (*"Select The Easiest Point"*) method for global univariate optimization
described in:

* S. Swarzberg, G. Seront & H. Bersini, *"S.T.E.P.: the easiest way to optimize a
  function"*, in IEEE World Congress on Computational Intelligence, Proceedings of the First
  IEEE Conference on Evolutionary Computation, vol. 1, pp. 519-524 (1994).

"""
module Step

using TypeUtils: @public
@public maximize, minimize, search

using ..Brent: Undef, fmin, concrete_precision, promote_typeof, tolerances, throw_bad_argument

using Printf, TypeUtils, Neutrals

const PlusOrMinusOne = Union{typeof(ONE),typeof(-ONE)}

"""
    OptimPack.Step.minimize([T,] f, a, b; kwds...) -> (xm, fm, lo, hi, nf)

Attempt to find a global minimum of `f(x)` in the interval `[a,b]`. Return the approximate
solution `xm`, `fm = f(xm)`, the bounds `lo` and `hi` of the interval bracketing the
solution, and the number `nf` of function evaluations. Optional argument `T` is the
floating-point type for computations, which is inferred from the numeric precision of `a`
and `b` by default.

See [`OptimPack.Step.search`](@ref) for a description of the algorithm and for allowed
keywords. See [`OptimPack.Step.maximize`](@ref) for searching for a global maximum.

""" minimize

"""
    OptimPack.Step.maximize([T,] f, a, b; kwds...) -> (xm, fm, lo, hi, nf)

Attempt to find a global maximum of `f(x)` in the interval `[a,b]`. Return the approximate
solution its position `xm`, `fm = f(xm)`, the bounds `lo` and `hi` of the interval
bracketing the solution, and the number `nf` of function evaluations. Optional argument `T`
is the floating-point type for computations, which is inferred from the numeric precision of
`a` and `b` by default.

See [`OptimPack.Step.search`](@ref) for a description of the algorithm and for allowed
keywords. See [`OptimPack.Step.minimize`](@ref) for searching for a global minimum.

""" maximize

for (func, sgn) in (:minimize => +ONE,
                    :maximize => -ONE)
    @eval begin
        $func(args...; kwds...) = search($sgn, args...; kwds...)
        $func(::Type{T}, args...; kwds...) where {T} =
            search(T, $sgn, args...; kwds...)
    end
end

"""
    OptimPack.Step.search([T,] sgn, f, a, b; kwds...) -> (xm, fm, lo, hi, nf)

Attempt to find a global extremum of the univariate function `f(x)` in the interval `[a,b]`.
Argument `sgn` can be `+𝟙` (resp. `-𝟙`) to search for a global minimum (resp. maximum).
Optional argument `T` is the floating-point type for computations, which is inferred from
the numeric precision of `a` and `b` by default.

The result is a 5-tuple with `xm` the position of the extremum, `fm = f(xm)` the
corresponding function value, `lo` and `hi` lower and upper bound for the solution, and `nf`
the number of function evaluations.

The algorithm is based on the STEP method described in:

* S. Swarzberg, G. Seront & H. Bersini, *"S.T.E.P.: the easiest way to optimize a
  function"*, in IEEE World Congress on Computational Intelligence, Proceedings of the First
  IEEE Conference on Evolutionary Computation, vol. 1, pp. 519-524 (1994).

The following optional keywords can be specified:

* `atol` and `rtol` are absolute and relative tolerances for the accuracy of the solution.
  The algorithm stops when:

      max(xm - lo, hi - xm) ≤ max(atol, rtol*abs(xm))

  By default, the absolute tolerance `atol` is set to the smallest positive normal number
  representable by the floating-point type of `x`, while the relative tolerance `rtol` is
  set to the square root of the relative precision of the floating-point type of `x`.

* `aboost` and `rboost` are absolute and relative boost parameters for the function value
  that the algorithm is trying to reach and which is defined as:

      fboost = fm ± max(aboost, rboost*abs(fm))

  where `fm` is the best function value so far and `±` is `+` when seeking for a maximum and
  `-` when seeking for a minimum. By default, these parameters are both set to zero as
  (hence `fboost = fm`) because these settings are the most efficient in practice (unlike
  what is suggested in the paper describing the S.T.E.P. method).

* `printer` can be set with a user defined function to print iteration information, its
  signature is:

      printer(io::IO, iter::Int, eval::Int, xm, fm, lo, hi)

  with `io` the output stream, `iter` the iteration number (`iter = 0` for the starting
  point), `eval` the number of calls to `f`, `xm` the best solution found so far, `fm =
  f(xm)` the corresponding function value, and `lo` and `hi` lower and upper bounds on the
  solution. If the variable `x` and/or the function value `f(x)` have units, these are
  discarded by the default printer (`xm`, `lo`, and `hi` however always have the same
  units).

* `output` specifies the output stream for printing information (`stdout` is used by
  default).

See also [`OptimPack.Step.minimize`](@ref) and [`OptimPack.Step.maximize`](@ref).

"""
search(sgn::Integer, f, a::Number, b::Number; kwds...) =
    search(concrete_precision(typeof(a), typeof(b)), sgn, f, a, b; kwds...)

function search(::Type{T}, sgn::Integer, f, a::Number, b::Number;
                kwds...) where {T<:AbstractFloat}
    if sgn == one(sgn)
        return search(T, +ONE, f, a, b; kwds...)
    elseif sgn isa Signed && sgn == -one(sgn)
        return search(T, -ONE, f, a, b; kwds...)
    else
        throw_bad_argument("`sgn` must be `±1`")
    end
end

function search(::Type{T}, sgn::PlusOrMinusOne, f, a::Number, b::Number;
                atol::Union{Number,Undef} = undef,
                rtol::Union{Number,Undef} = undef,
                aboost::Union{Number,Undef} = undef,
                rboost::Union{Number,Undef} = undef,
                kwds...) where {T<:AbstractFloat}
    # Convert ends of initial interval to chosen floating-point type.
    Tx = convert_real_type(T, promote_typeof(a, b))
    a = convert(Tx, a)
    b = convert(Tx, b)

    # Absolute and relative tolerances for the solution are the same as for Brent's `fmin`
    # algorithm.
    atol, rtol = tolerances(fmin, a, b; atol=atol, rtol=rtol)

    # Evaluate function at the ends of the initial interval.
    fa = sgn*convert_real_type(T, f(a))
    Tf = typeof(fa)
    fb, eval = if a == b
        fa, 1
    else
        sgn*convert(Tf, f(b)), 2
    end

    # Determine function boost parameters.
    if aboost isa Number
        aboost ≥ zero(aboost) || throw_bad_argument("parameter `aboost` must be non-negative")
        aboost = convert(Tf, aboost)::Tf
    else
        aboost = zero(Tf)
    end
    if rboost isa Number
        rboost ≥ zero(rboost) || throw_bad_argument("parameter `rboost` must be non-negative")
        rboost = convert(T, rboost)::T
    else
        rboost = zero(T)
    end

    # Apply STEP algorithm.
    return _search(sgn, f, a, fa, b, fb, eval, atol, rtol, aboost, rboost; kwds...)
end

function _search(sgn::PlusOrMinusOne, f, a::Tx, fa::Tf, b::Tx, fb::Tf, eval::Int,
                 atol::Tx, rtol::T, aboost::Tf, rboost::T;
                 verb::Bool = false, printer = default_printer,
                 output::IO = stdout) where {T<:AbstractFloat,Tx,Tf}

    # Order ends of initial search interval.
    if a > b
        a, fa, b, fb = b, fb, a, fa
    end

    # Initial best solution.
    xbest, fbest = fa < fb ? (a, fa) : (b, fb)
    iter = 0 # initial number of iterations
    verb && printer(output, iter, eval, xbest, sgn*fbest, a, b)
    if has_converged(xbest, a, b, atol, rtol)
        return (xbest, sgn*fbest, a, b, eval)
    end

    # If the initial interval is larger than the required precision,
    # iteratively split the "easiest" sub-interval until the precision
    # criterion is satisfied.

    # Create initial chained list of nodes.
    qnan = sqrt(zero(Tf))/zero(Tx) # so that any comparison with this value is false
    list = Node(a, fa, qnan)
    append!(list, b, fb, qnan)

    # Iteration number of the last full update of the "difficulties" of the
    # sub-intervals.
    last_rehash = iter - 1 # force rehash
    fboost = fbest # declare local variable with correct type, its value will
                   # be computed at next rehash

    # Iterate until convergence.
    split = list # manage to split the initial interval
    while true
        # Split the chosen interval.
        c = (a + b)/2
        fc = sgn*convert(Tf, f(c))
        eval += 1
        if fc < fbest
            # Solution has improved. Memorize it and check for convergence.
            xbest, fbest = c, fc
            iter += 1
            verb && printer(output, iter, eval, xbest, sgn*fbest, a, b)
            if has_converged(xbest, a, b, atol, rtol)
                return (xbest, sgn*fbest, a, b, eval)
            end
        end

        # Insert node c in chained list, update "difficulties", and find next
        # sub-interval to split.
        append!(split, c, fc, qnan)
        qmin = typemax(qnan)
        if last_rehash < iter
            # Do not rehash until next iteration.
            last_rehash = iter

            # Recompute all "difficulties".
            fboost = fbest - max(aboost, rboost*abs(fbest))
            this = list
            w = sqrt(this.fx - fboost)
            while true
                next = this.next
                this.x < next.x || break
                w′, w = w, sqrt(next.fx - fboost)
                q = (w + w′)/abs(next.x - this.x)
                if q < qmin
                    qmin = q
                    split = this
                end
                this.q = q
                this = next
            end
        else
            # Compute the "difficulties" of the two new sub-intervals.
            wa = sqrt(fa - fboost)
            wb = sqrt(fb - fboost)
            wc = sqrt(fc - fboost)
            split.q      = (wa + wc)/abs(c - a)
            split.next.q = (wc + wb)/abs(b - c)

            # Find next interval to split.
            this = list
            while true
                next = this.next
                this.x < next.x || break
                if this.q < qmin
                    qmin = this.q
                    split = this
                end
                this = next
            end
        end

        # Get ends of the chosen interval and corresponding function values.
        next = split.next
        a = split.x
        b = next.x
        fa = split.fx
        fb = next.fx
    end
end

function default_printer(io::IO, iter::Int, eval::Int,
                         xm::Tx, fm::Tf, a::Tx, b::Tx) where {Tx,Tf}
    if eval < 3
        @printf(io, "# %s%s\n# %s%s\n",
                "ITERS    EVALS              X         ",
                "             F(X)               PREC",
                "--------------------------------------",
                "--------------------------------------")
    end
    @printf(io, "%7d  %7d  %23.15e  %23.15e  %10.2e\n", iter, eval,
            xm/oneunit(Tx), fm/oneunit(Tf), abs(b - a)/(2*oneunit(Tx)))
end

# Yields whether search has converged withing tolerances. x ∈ [a, b] is the current
# estimate, a ≤ b are the ends of the current bracketing interval, atol and rtol are
# absolute and relative tolerances.
function has_converged(x::Tx, a::Tx, b::Tx, atol::Tx, rtol::AbstractFloat) where {Tx}
    return max(abs(x - a), abs(x - b)) ≤ max(atol, rtol*abs(x))
end

"""
    OptimPack.Step.Node(x, fx, q[, next]) -> node

Build a new node for the S.T.E.P. method at coordinate `x`, function value `fx = f(x)`, and
*"difficulty"* `q`. The nodes form a cyclic chained list, the next node of the new node may
optionally be specified.

"""
mutable struct Node{Tx,Tf,Tq}
    x::Tx  # position
    fx::Tf # function value
    q::Tq  # "quality" factor of the interval
    next::Node{Tx,Tf,Tq}
    function Node(x::Tx, fx::Tf, q::Tq, next::Node{Tx,Tf,Tq}) where {Tx,Tf,Tq}
        return new{Tx,Tf,Tq}(x, fx, q, next)
    end
    function Node(x::Tx, fx::Tf, q::Tq) where {Tx,Tf,Tq}
        node = new{Tx,Tf,Tq}(x, fx, q)
        node.next = node
        return node
    end
end

"""
    append!(node, x, fx, q) -> next

Append a new node with parameters `x`, `fx = f(x)`, and `q` to `node` and return it.

"""
function Base.append!(node::Node, x, fx, q)
    next = Node(x, fx, q, node.next)
    node.next = next
    return next
end

# Simple parabola.  To be minimized over [-1,2].
testParabola(x) = x*x

# Brent's 5th function.  To be minimized over [-10,10].
testBrent5(x) = (x - sin(x))*exp(-x*x)

# Michalewicz's 1st function.  To be minimized over [-1,2].
testMichalewicz1(x) = x*sin(10.0*x)

# Michalewicz's 2nd function.  To be maximized over [0,pi].
function testMichalewicz2(x)
    s = 0.0
    a = sin(x)
    b = x*x/pi
    for i in 1:10
        s += a*(sin(b*i)^20)
    end
    return s
end

function runtests()
    println("\n# Simple parabola:")
    (xbest, fbest, lo, hi, n) = minimize(testParabola, -1, 2, verb=true,
                                           atol=1e-12, rtol=0)
    println("x = $xbest ± $((hi - lo)/2), f(x) = $fbest, ncalls = $n")

    println("\n# Brent's 5th function:")
    (xbest, fbest, lo, hi, n) = minimize(testBrent5, -10, 10, verb=true)
    println("x = $xbest ± $((hi - lo)/2), f(x) = $fbest, ncalls = $n")

    println("\n# Michalewicz's 1st function:")
    (xbest, fbest, lo, hi, n) = minimize(testMichalewicz1, -1, 2, verb=true)
    println("x = $xbest ± $((hi - lo)/2), f(x) = $fbest, ncalls = $n")

    println("\n# Michalewicz's 2nd function:")
    (xbest, fbest, lo, hi, n) = maximize(testMichalewicz2, 0, pi, verb=true,
                                         atol=1e-9, rtol=0)
    println("x = $xbest ± $((hi - lo)/2), f(x) = $fbest, ncalls = $n")
end

end # module
