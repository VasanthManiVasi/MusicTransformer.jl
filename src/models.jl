using Flux: @functor
using MacroTools: @forward

using Transformers.Basic
using Transformers.Basic: AbstractTransformer
using Transformers.Stacks

struct BaselineMusicTransformer{T<:Stack} <: AbstractTransformer
    ts::T
end

@functor BaselineMusicTransformer

@forward BaselineMusicTransformer.ts Base.getindex, Base.length

function BaselineMusicTransformer(size::Int, heads::Int, ps::Int, layers::Int)
    rem(size, heads) != 0 && error("size not divisible by heads")
    BaselineMusicTransformer(size, heads, div(size, heads), ps, layers)
end

function BaselineMusicTransformer(size::Int, head::Int, hs::Int, ps::Int, layers::Int)
    BaselineMusicTransformer(
        Stack(
            @nntopo_str("indices => e:e => pe:(e, pe) => input => $layers"),
            Embed(size, 310),
            PositionEmbedding(size, 2048),
            (e, pe) -> (e .+ pe),
            [
                Transformer(size, head, hs, ps; future=false, pdrop=0)
                for i = 1:layers
            ]...,
        )
    )
end