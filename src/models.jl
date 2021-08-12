export UnconditionalMusicTransformer, MelodyConditionedMusicTransformer

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

struct MelodyConditionedMusicTransformer{EE<:Stack, E<:Stack, DE<:Stack, D<:Stack} <: MusicTransformerModel
    encoder_embedding::EE
    encoder::E
    decoder_embedding::DE
    decoder::D
end

@functor MelodyConditionedMusicTransformer

@forward MelodyConditionedMusicTransformer.ts Base.getindex, Base.length

function MelodyConditionedMusicTransformer(size::Int, heads::Int, ps::Int,
                                     encoder_layers::Int, decoder_layers::Int)

    rem(size, heads) != 0 && error("size not divisible by heads")
    MelodyConditionedMusicTransformer(size, heads, div(size, heads), ps, encoder_layers, decoder_layers)
end

function MelodyConditionedMusicTransformer(size::Int, head::Int, hs::Int, ps::Int,
                                     encoder_layers::Int, decoder_layers::Int)

    MelodyConditionedMusicTransformer(
        Stack(
            @nntopo_str(
                "(indices, tid):indices => e:tid => te:(e, te) => out => pe:(out, pe) => enc_input"),
            Embed(size, 92, scale=sqrt(size)),
            Embed(size, 32),
            # Add target space embedding
            (e, te) -> Float32.(e .+ te),
            PositionEncoding(size, 4096),
            # Add position embedding
            (e, pe) -> (e .+ pe)
        ),
        Stack(
            @nntopo_str("input => $encoder_layers => output"),
            [
                MusicTransformerBlock(size, head, hs, ps; future=true, pdrop=0)
                for i = 1:encoder_layers
            ]...,
            LayerNorm(size),
        ),
        Stack(
            @nntopo_str("indices => e => pe:(e, pe) => dec_input"),
            Embed(size, 310, scale=sqrt(size)),
            PositionEncoding(size, 4096),
            # Add position embedding
            (e, pe) -> (e .+ pe)
        ),
        Stack(
            @nntopo_str(
                "((input, m) => input:(input, m)) => $decoder_layers:input => norm => output"),
            [
                MusicTransformerDecoder(size, head, hs, ps; pdrop=0)
                for i = 1:decoder_layers
            ]...,
            LayerNorm(size),
            Dense(size, 310), # softmax embedding layer
        )
    )
end

function (mt::MelodyConditionedMusicTransformer)(inputs::Vector{Int}, targets::Vector{Int})
    target_space_id = [1]
    encoder_inputs = mt.encoder_embedding(inputs, target_space_id)
    encoder_context = mt.encoder(encoder_inputs)
    decoder_inputs = mt.decoder_embedding(targets)
    logits = mt.decoder(decoder_inputs, encoder_context)

    logits, encoder_context
end

function Base.show(io::IO, mt::MelodyConditionedMusicTransformer)
    hs = div(size(mt.encoder[1].mh.iqproj.W)[1], mt.encoder[1].mh.head)
    h, ps = size(mt.encoder[1].pw.dout.W)
    encoder_layers = length(mt.encoder) - 1 # Ignore postprocess layernorm
    decoder_layers = length(mt.decoder) - 2 # Ignore postprocess layernorm and softmax embedding layer
    print(io, "MelodyConditionedMusicTransformer(")
    print(io, "encoder_layers=$encoder_layers, ")
    print(io, "decoder_layers=$decoder_layers, ")
    print(io, "head=$(mt.encoder[1].mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h))")
end
