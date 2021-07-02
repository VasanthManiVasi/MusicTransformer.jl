using Flux: @functor
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

function (mt::BaselineMusicTransformer)(embeds::T) where T
    mt.ts(embeds)
end

function generate(model::MT;
    primer::Vector{PerformanceEvent}=[PerformanceEvent(TIME_SHIFT, 100)],
    raw = false)

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
    performance.events = deepcopy(primer)

    # Take account of PAD and EOS by adding 2 to encodeindex
    inputs = map(event -> encodeindex(event, performance) + 2, performance)

    logits = model(inputs)
    prediction = wsample(performance.labels, softmax(logits[:, end]))
    push!(inputs, prediction)
    push!(performance, decodeindex(prediction, performance))

    # Sample till end of sequence is encountered (EOS = 2)
    while pred != 2
        logits = model(inputs)
        prediction = wsample(performance.labels, softmax(logits[:, end]))
        push!(inputs, prediction)
        push!(performance, decodeindex(prediction, performance))
        print(prediction)
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