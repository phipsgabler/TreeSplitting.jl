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
