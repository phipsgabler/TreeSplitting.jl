
abstract type Tree{N} end

struct Leaf{N} <: Tree{N}
    label::Int
    
    Leaf{N}(n) where N = (1 <= n <= N) ? new{N}(n) : error("Illegal construction")
end

struct Branch{N} <: Tree{N}
    left::Tree{N}
    right::Tree{N}
end


import Base: show

function show(io::IO, l::Leaf, indent = 0)
    print(io, " " ^ indent, "Leaf($(l.label))")
end

function show(io::IO, b::Branch, indent = 0)
    print(io, " " ^ indent)
    print(io, "Branch(\n")
    show(io, b.left, indent + 2)
    print(io, ",\n")
    show(io, b.right, indent + 2)
    print(io, ")")
end


import Base: rand
using Random: AbstractRNG, Sampler, SamplerType

rand(rng::AbstractRNG, ::SamplerType{Leaf{N}}) where {N} = Leaf{N}(rand(rng, 1:N))

"""
    BoltzmannSampler{N}(m, n)
Configuration for Boltzmann sampling `DecisionTree{N}`s with minimum size `m` and maximum size `n`.
"""
struct BoltzmannSampler{N} <: Sampler{Tree{N}}
    minsize::Int
    maxsize::Int
end

function boltzmann_ub(rng::AbstractRNG, sampler::BoltzmannSampler{N}, cursize) where {N}
    minsize = sampler.minsize
    maxsize = sampler.maxsize

    if cursize ≥ maxsize        # maximum bound exceeded
        return nothing, cursize
    elseif rand(rng) ≤ 0.5      # generate random leaf
        return rand(rng, Leaf{N}), cursize
    else                        # try generating a branch -- if its size is small enough
        # try generating children, or immediately fail
        left_children, cursize = boltzmann_ub(rng, sampler, cursize + 1)
        left_children === nothing && return nothing, cursize

        right_children, cursize = boltzmann_ub(rng, sampler, cursize + 1)
        right_children === nothing && return nothing, cursize

        # `cursize` is now the size of the whole thing, and less than maxsize
        return Branch{N}(left_children, right_children), cursize        
    end
end

function rand(rng::AbstractRNG, sampler::BoltzmannSampler{N}) where {N}
    candidate, size = boltzmann_ub(rng, sampler, 0)
    while candidate === nothing || size ≤ sampler.minsize
        candidate, size = boltzmann_ub(rng, sampler, 0)
    end

    return candidate
end


function randomsplitnaive_impl(l::Leaf, n, context, continuation)
    n += 1
    if rand() ≤ 1/n
        continuation = k -> (context(k(l)), l)
    end

    return continuation, n
end

function randomsplitnaive_impl(b::Branch, n, context, continuation)
    continuation, n = randomsplitnaive_impl(b.left, 
                                            n,
                                            t -> context(Branch(t, b.right)),
                                            continuation)
    continuation, n = randomsplitnaive_impl(b.right, 
                                            n,
                                            t -> context(Branch(b.left, t)),
                                            continuation)

    n += 1
    if rand() ≤ 1/n
        continuation = k -> (context(k(b)), b)
    end
    
    return continuation, n
end


"""
    randomsplitnaive(action, t::Tree)
Select uniformly at random a subtree `s` of `t`, then replace `s` by `action(s)`. 
Returns both the new tree and the replaced subtree.
"""
function randomsplitnaive(action, t::Tree)
    # The continuation argument will be assigned a default with probability 1 on the first leaf,
    # so passing `nothing` is safe here.
    cont, _ = randomsplitnaive_impl(t, 0, identity, nothing)
    cont(action)
end


randomchildnaive(t::Tree) = randomsplitnaive(identity, t)[2]


sumtree(t::Leaf) = t.label
sumtree(t::Branch) = sumtree(t.left) + sumtree(t.right)

function testvalues_native(t, N)
    zd = [sumtree(randomchildnaive(t)) for _ in 1:N]
    [(i, sum(zd .== i)) for i in unique(zd)]
end


# Context types:
abstract type Context{N} end

struct LeftContext{N} <: Context{N}
    right::Tree{N}
    parent::Context{N}
end

struct RightContext{N} <: Context{N}
    left::Tree{N}
    parent::Context{N}
end

# This will be used instead of `nothing` in the top-level call
struct NoContext{N} <: Context{N} end


# Continuation type:
struct Cont{N}
    context::Context{N}
    chunk::Tree{N}
end



function (context::RightContext{N})(t::Tree{N}) where N
    context.parent(Branch(context.left, t))
end

function (context::LeftContext{N})(t::Tree{N}) where N
    context.parent(Branch(t, context.right))
end

function (context::NoContext{N})(t::Tree{N}) where N
    t
end

function (continuation::Cont{N})(k::Function) where N
    newchunk = k(continuation.chunk)::Tree{N}
    tree = continuation.context(newchunk)::Tree{N}
    return tree, continuation.chunk
end


function randomsplit_impl(t::Leaf{N}, n::Int, context::Context{N}, continuation::Cont{N}) where {N}
    n += 1
    if rand() ≤ 1/n
        return Cont{N}(context, t), n
    else
        return continuation, n
    end
end

function randomsplit_impl(t::Branch{N}, n::Int, context::Context{N}, continuation::Cont{N}) where {N}
    c1, n = randomsplit_impl(t.left, n, LeftContext{N}(t.right, context), continuation)
    c2, n = randomsplit_impl(t.right, n, RightContext{N}(t.left, context), c1)
    
    n += 1
    if rand() ≤ 1/n
        return Cont{N}(context, t), n
    else
        return c2, n
    end
end

function randomsplit(action, t::Tree{N}) where {N}
    default_context = NoContext{N}()
    cont, _ = randomsplit_impl(t, 0, default_context, Cont{N}(default_context, t))
    cont(action)
end

randomchild(t::Tree{N}) where {N} = randomsplit(identity, t)[2]


function testvalues(t, N)
    zd = [sumtree(randomchild(t)) for _ in 1:N]
    [(i, sum(zd .== i)) for i in unique(zd)]
end

