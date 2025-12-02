#---------------------------------------------------------------------- API for algorithms -

# TODO documentation
function solve! end
function configure! end
function restart! end
function iterate! end

#--------------------------------------------------------------------- API for convex sets -

"""
    OptimPack.has_constraints(Ω) -> bool

Return whether convex set `Ω` of the feasible variables imposes some constraints.

See also [`OptimPack.project_variables!`](@ref).

"""
has_constraints(Ω::ConvexSet) = false

"""
    OptimPack.project_variables!([ls,] x, Ω) -> x

Project variables `x` onto the convex set `Ω`.

See also [`OptimPack.project_direction!`](@ref) and [`OptimPack.line_search_limits`](@ref).

"""
function project_variables!(x::AbstractArray, Ω::ConvexSet)
    return project_variables!(LoopStyle(x, Ω), x, Ω)
end

"""
    OptimPack.project_direction!([ls,] p, x, ±, d, Ω) -> p

Overwrite `p` with the projected direction such that, for some `ε > 0`, the following
property holds:

    ∀ α ∈ [0,ε], proj(x ± α⋅d) = x ± α⋅p

with `±` being either `+` or `-`, `proj` the projection onto the feasible convex set `Ω ⊆
ℝⁿ`, `x ∈ Ω` the variables, and `d ∈ ℝⁿ` a search direction. Hence, `±p` is a feasible
search direction starting at `x ∈ Ω`.

!!! note
    `x` must be feasible, that is `x ∈ Ω` must hold; this is not verified for efficiency
    reasons.

See also [`OptimPack.unblocked_variables!`](@ref), [`OptimPack.project_variables!`](@ref)
and [`OptimPack.line_search_limits`](@ref).

"""
function project_direction!(p::AbstractArray, x::AbstractArray, pm::PlusMinus,
                            d::AbstractArray, Ω::ConvexSet)
    return project_direction!(LoopStyle(p, x, d, Ω), p, x, pm, d, Ω)
end

"""
    OptimPack.unblocked_variables!([ls,] u, x, ±, d, Ω) -> u

Overwrite `u` with zeros and ones depending whether variables are blocked or not by the
constraints `x ∈ Ω ⊆ ℝⁿ` in the line-search in the direction `±d` starting at `x`. `Ω` is
the convex set of feasible variables and the element-wise multiplication of `u` and `d`
yields the projected direction.

!!! note
    `x` must be feasible, that is `x ∈ Ω` must hold; this is not verified for efficiency
    reasons.

See also [`OptimPack.projected_direction!`](@ref), [`OptimPack.project_variables!`](@ref)
and [`OptimPack.line_search_limits`](@ref).

"""
function unblocked_variables!(u::AbstractArray, x::AbstractArray, pm::PlusMinus,
                              d::AbstractArray, Ω::ConvexSet)
    return unblocked_variables!(LoopStyle(u, x, d, Ω), u, x, pm, d, Ω)
end

"""
    OptimPack.line_search_limits([ls,] x0, ±, d, Ω) -> (αₘᵢₙ, αₘₐₓ)

Return the limits `αₘᵢₙ ≥ 0` and `αₘₐₓ ≥ 0` for the step length `α` in a line-search where
iterates `x` are given by:

    x = proj(x0 ± α*d)

with `proj(x)` the orthogonal projection on the convex set `Ω ⊆ ℝⁿ` of feasible variables.

The limit `αₘᵢₙ` is the largest nonnegative step length such that:

    0 ≤ α ≤ αₘᵢₙ    ==>    proj(x0 ± α*d) = x0 ± α*d

The limit `αₘₐₓ` is the least nonnegative step length such that:

    α ≥ αₘₐₓ    ==>    proj(x0 ± α*d) = proj(x0 ± αₘₐₓ*d)

In other words, no bounds are overcome if `0 ≤ α ≤ αₘᵢₙ` and the projected variables are all
the same for any `α` such that `α ≥ αₘₐₓ`.

!!! note
    `x0` must be feasible, that is `x0 ∈ Ω` must hold; this is not verified for efficiency
    reasons.

See also: [`OptimPack.line_search_step_max`](@ref), [`OptimPack.project_variables!`](@ref),
and [`OptimPack.project_direction!`](@ref).

"""
function line_search_limits(x0::AbstractArray, pm::PlusMinus, d::AbstractArray,
                            Ω::ConvexSet)
    return line_search_limits(LoopStyle(x0, d, Ω), x0, pm, d, Ω)
end

"""
    OptimPack.line_search_step_max([ls,] x0, ±, d, Ω) -> αₘₐₓ

Return the limit `αₘₐₓ ≥ 0` for the step length `α` in a line-search where iterates `x` are
given by:

    x = proj(x0 ± α*d)

with `proj(x)` the orthogonal projection on the convex set`Ω ⊆ ℝⁿ` of feasible variables.

The limit `αₘₐₓ` is the least nonnegative step length such that:

    α ≥ αₘₐₓ    ==>    proj(x0 ± α*d) = proj(x0 ± αₘₐₓ*d)

In other words, the projected variables are all the same for any `α` such that `α ≥ αₘₐₓ`.

!!! note
    `x0` must be feasible, that is `x0 ∈ Ω` must hold; this is not verified for efficiency
    reasons.

See also: [`OptimPack.line_search_limits`](@ref), [`OptimPack.project_variables!`](@ref),
and [`OptimPack.project_direction!`](@ref).

"""
function line_search_step_max(x0::AbstractArray, pm::PlusMinus, d::AbstractArray,
                              Ω::ConvexSet)
    return line_search_step_max(LoopStyle(x0, d, Ω), x0, pm, d, Ω)
end
