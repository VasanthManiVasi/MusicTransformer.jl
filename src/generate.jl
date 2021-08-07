export generate

using NoteSequences
using NoteSequences: DEFAULT_STEPS_PER_SECOND

using StatsBase: wsample

function default_performance_encoder()
    MIN_PITCH = 21
    MAX_PITCH = 108
    steps_per_second = NoteSequences.DEFAULT_STEPS_PER_SECOND
    num_velocitybins = 32
    add_eos = false
    encoder = MidiPerformanceEncoder(steps_per_second, num_velocitybins,
                                     MIN_PITCH, MAX_PITCH, add_eos)

    encoder
end

function sample_and_append!(logits::Array,
                            performance::Performance,
                            encoder::MidiPerformanceEncoder,
                            inputs::Vector{Int})

    prediction = wsample(1:encoder.vocab_size, softmax(logits[:, end]))

    # Add prediction to the model inputs
    push!(inputs, prediction)

    # Add predicted event to performance
    push!(performance, decodeindex(prediction, encoder))

    prediction
end

function generate_events!(model::MusicTransformerModel,
                          performance::Performance,
                          encoder::MidiPerformanceEncoder,
                          numsteps::Int)

    inputs = map(event -> encodeindex(event, encoder), performance)

    logits = model(inputs)
    pred = sample_and_append!(logits, performance, encoder, inputs)

    # Sample till end of sequence is encountered or numsteps is reached
    while pred != EOS_TOKEN && performance.numsteps <= numsteps
        logits = model(inputs)
        pred = sample_and_append!(logits, performance, encoder, inputs)
    end
end

function generate(model::MusicTransformerModel;
                  primer::Performance=Performance(DEFAULT_STEPS_PER_SECOND, velocity_bins=32),
                  numsteps::Int=3000,
                  as_notesequence::Bool=false,
                  encoder::MidiPerformanceEncoder=default_performance_encoder())

    performance = deepcopy(primer)

    if isempty(performance)
        push!(performance, encoder.default_event)
    end

    @info "Generating..."
    generate_events!(model, performance, encoder, numsteps)

    ns = getnotesequence(performance)
    as_notesequence == true && return ns
    notesequence_to_midi(ns)
end