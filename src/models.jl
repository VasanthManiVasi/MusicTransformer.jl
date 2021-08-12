export UnconditionalMusicTransformer

using Flux: unsqueeze, @functor
using MacroTools: @forward

using Transformers.Basic: AbstractTransformer, Embed
using Transformers.Stacks

abstract type MusicTransformerModel <: AbstractTransformer end

struct UnconditionalMusicTransformer{T<:Stack} <: MusicTransformerModel
    ts::T
end

@functor UnconditionalMusicTransformer

@forward UnconditionalMusicTransformer.ts Base.getindex, Base.length

function UnconditionalMusicTransformer(size::Int, heads::Int, ps::Int, layers::Int)
    rem(size, heads) != 0 && error("size not divisible by heads")
    UnconditionalMusicTransformer(size, heads, div(size, heads), ps, layers)
end

function UnconditionalMusicTransformer(size::Int, head::Int, hs::Int, ps::Int, layers::Int)
    UnconditionalMusicTransformer(
        Stack(
            @nntopo_str("indices => e:e => pe:(e, pe) => input => $layers => norm => logits"),
            Embed(size, 310, scale=sqrt(size)),
            PositionEncoding(size, 4096),
            # Perform bottom transformation and add position embedding
            (e, pe) -> (e .+ pe),
            [
                MusicTransformerBlock(size, head, hs, ps; future=false, pdrop=0)
                for i = 1:layers
            ]...,
            LayerNorm(size),
            Dense(size, 310),
        )
    )
end

function (mt::UnconditionalMusicTransformer)(indices::T) where T
    mt.ts(indices)
end

function Base.show(io::IO, mt::UnconditionalMusicTransformer)
    layer_1 = 3 + 1 # index of layer 1 is after the first 3 embedding layers
    hs = div(size(mt.ts[layer_1].mh.iqproj.W)[1], mt.ts[layer_1].mh.head)
    h, ps = size(mt.ts[layer_1].pw.dout.W)
    num_layers = length(mt.ts) - 3 - 2 # Ignore embedding and output layers
    print(io, "UnconditionalMusicTransformer(")
    print(io, "layers=$num_layers, ")
    print(io, "head=$(mt.ts[layer_1].mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h))")
end