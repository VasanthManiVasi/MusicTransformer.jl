using Flux: unsqueeze, @functor
using MacroTools: @forward

using Transformers.Basic
using Transformers.Basic: AbstractTransformer
using Transformers.Stacks

using NoteSequences
using StatsBase: wsample

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
            @nntopo_str("indices => e:e => pe:(e, pe) => input => $layers => logits"),
            Embed(size, 310),
            PositionEmbedding(size, 2048),
            (e, pe) -> (e .+ pe),
            [
                Transformer(size, head, hs, ps; future=false, pdrop=0)
                for i = 1:layers
            ]...,
            Dense(size, 310),
        )
    )
end

function PositionEmbedding(size::Int, max_len::Int = 2048)
    # Follows the tensor2tensor implementation - which is used by the unconditional 16L Music Transformer
    num_timescales = size / 2
    positions = Float32.(collect(0.0:max_len-1))
    log_timescale_increment = log(1e4) / (num_timescales-1)
    inv_timescales = Float32.(exp.(collect(0.0:(num_timescales-1)) .* -log_timescale_increment))
    scaled_time = unsqueeze(positions, 2) * unsqueeze(inv_timescales, 1)
    embedding = hcat(sin.(scaled_time), cos.(scaled_time))
    Transformers.Basic.PositionEmbedding(false, collect(embedding'))
end

function (mt::BaselineMusicTransformer)(embeds::T) where T
    mt.ts(embeds)
end

function (t::Transformer)(x::A, mask=nothing) where {T, N, A<:AbstractArray{T, N}}
    dropout = t.drop

    # Layer norm is a preprocess for the unconditional 16L Music Transformer,
    # Dropout and addition (residual connection) are postprocesses

    # Add x in residual connection
    # Attention layer
    x_normed = t.mhn(x) # LayerNorm
    a = t.mh(x_normed, x_normed, x_normed; mask=mask) # MultiheadAttention
    # a = dropout(a) # Dropout
    res_a = x + a # Addition (residual)

    # Feed-forward layer
    res_a_normed = t.pwn(res_a) # LayerNorm
    pwffn = t.pw(res_a_normed) # Pointwise feed-forward
    # pwffn = dropout(pwffn) # Dropout
    res_pwffn = res_a + pwffn # Addition
    res_pwffn

    #=
    # Add x_norm in residual connection
    # Attention layer
    x = t.mhn(x) # LayerNorm
    a = t.mh(x, x, x; mask=mask) # MultiheadAttention
    # a = dropout(a) # Dropout
    res_a = x + a # Addition (residual)

    # Feed-forward layer
    res_a = t.pwn(res_a) # LayerNorm
    pwffn = t.pw(res_a) # Pointwise feed-forward
    # pwffn = dropout(pwffn) # Dropout
    res_pwffn = res_a + pwffn # Addition
    res_pwffn
    =#
end

function sample_and_append!(logits, performance, inputs)
    prediction = wsample(performance.labels, softmax(logits[:, end]))
    # Add prediction to the model inputs
    push!(inputs, prediction)
    # Add predicted event to performance
    push!(performance, decodeindex(prediction, performance))

    prediction
end

function default_performance()
    performance = Performance(100, velocity_bins = 32)
    MIN_PITCH = 21
    MAX_PITCH = 108
    performance.event_ranges = [
        (NOTE_ON, MIN_PITCH, MAX_PITCH),
        (NOTE_OFF, MIN_PITCH, MAX_PITCH),
        (TIME_SHIFT, 1, NoteSequences.DEFAULT_MAX_SHIFT_STEPS),
        (VELOCITY, 1, 32) # 32 velocity bins
    ]
    performance.num_classes = 308 + 2 # Add PAD, EOS
end

function generate(model::BaselineMusicTransformer;
                  primer::Vector{PerformanceEvent}=[PerformanceEvent(TIME_SHIFT, 100)],
                  raw = false)

    performance = default_performance()
    performance.events = deepcopy(primer)

    # Take account of PAD and EOS by adding 2 to encodeindex
    inputs = map(event -> encodeindex(event, performance) + 2, performance)

    logits = model(inputs)
    pred = sample_and_append!(logits, performance, inputs)

    # Sample till end of sequence is encountered (EOS = 2)
    while pred != 2
        logits = model(inputs)
        pred = sample_and_append!(logits, performance, inputs)
        print(pred)
    end

    raw == true && return performance
    getnotesequence(performance)
end

function Base.show(io::IO, mt::BaselineMusicTransformer)
    layer_1 = 3 + 1 # index of layer 1 is after the first 3 embedding layers
    hs = div(size(mt.ts[layer_1].mh.iqproj.W)[1], mt.ts[layer_1].mh.head)
    h, ps = size(mt.ts[layer_1].pw.dout.W)
    num_layers = length(mt.ts) - 3 - 1 # Ignore embedding and output layers
    print(io, "BaselineMusicTransformer(")
    print(io, "layers=$num_layers, ")
    print(io, "head=$(mt.ts[layer_1].mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h))")
end