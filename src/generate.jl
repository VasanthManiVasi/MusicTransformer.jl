export generate

using NoteSequences
using NoteSequences.PerformanceRepr
using StatsBase: wsample

function default_performance_encoder()
    MIN_PITCH = 21
    MAX_PITCH = 108
    perfencoder = PerformanceOneHotEncoding(minpitch=MIN_PITCH,
                                            maxpitch=MAX_PITCH,
                                            num_velocitybins=32)

    perfencoder
end

function sample_and_append!(logits::Array,
                            performance::Performance,
                            perfencoder::PerformanceOneHotEncoding,
                            inputs::Vector{Int})
    prediction = wsample(perfencoder.labels, softmax(logits[:, end]))
    # Add prediction to the model inputs
    push!(inputs, prediction)
    # Add predicted event to performance
    push!(performance, decodeindex(prediction, perfencoder))

    prediction
end

function generate_events!(model::MusicTransformerModel,
                          performance::Performance,
                          perfencoder::PerformanceOneHotEncoding,
                          numsteps::Int)
    # Take account of PAD and EOS by adding 2 to encodeindex
    inputs = map(event -> encodeindex(event, perfencoder) + 2, performance)

    logits = model(inputs)
    pred = sample_and_append!(logits, performance, perfencoder, inputs)

    # Sample till end of sequence is encountered (EOS = 2) or numsteps is reached
    while pred != 2 && performance.numsteps <= numsteps
        logits = model(inputs)
        pred = sample_and_append!(logits, performance, perfencoder, inputs)
    end
end

function generate(model::MusicTransformerModel;
                  primer::Performance=Performance(100, velocity_bins=32),
                  numsteps::Int=3000,
                  as_notesequence::Bool=false)

    perfencoder = default_performance_encoder()
    performance = deepcopy(primer)

    if isempty(performance)
        push!(performance, perfencoder.defaultevent)
    end

    @info "Generating..."
    generate_events!(model, performance, perfencoder, numsteps)

    ns = getnotesequence(performance)
    as_notesequence == true && return ns
    notesequence_to_midi(ns)
end