export generate, generate_accompaniment

using NoteSequences
using NoteSequences: DEFAULT_STEPS_PER_SECOND

using StatsBase: wsample

const MIN_PITCH = 21
const MAX_PITCH = 108
const VELOCITY_BINS = 32

"""
    default_melody_encoder()

Return a TextMelodyEncoder with the following parameters.
`minpitch::Int`=21
`maxpitch::Int`=109
`steps_per_second::Int`=100.
"""
default_melody_encoder() = TextMelodyEncoder(MIN_PITCH, MAX_PITCH + 1, DEFAULT_STEPS_PER_SECOND)

"""
    default_performance_encoder(add_eos::Bool=true)

Return a MidiPerformanceEncoder with the following parameters.
`steps_per_second::Int`=100
`velocity_bins::Int`=32
`minpitch::Int`=21
`maxpitch::Int`=109

If `add_eos` is true, `EOS_TOKEN` is appended at the end of the sequence.
"""
function default_performance_encoder(add_eos::Bool=true)
    encoder = MidiPerformanceEncoder(DEFAULT_STEPS_PER_SECOND, VELOCITY_BINS,
                                     MIN_PITCH, MAX_PITCH, add_eos)

    encoder
end

"""
    sample!(performance::Performance,
            targets::Vector{Int},
            logits::Array{T, N},
            encoder::MidiPerformanceEncoder)

Sample from `logits` and append the output to the `targets`.
The decoded performance event from the output is appended to the performance.
"""
function sample!(performance::Performance,
                 targets::Vector{Int},
                 logits::Array{T, N},
                 encoder::MidiPerformanceEncoder) where {T, N}

    prediction = wsample(1:encoder.vocab_size, softmax(logits[:, end]))

    # Add prediction to the model targets
    push!(targets, prediction)

    # Add predicted event to performance
    push!(performance, decodeindex(prediction, encoder))

    prediction
end

"""
    generate_events!(model::UnconditionalMusicTransformer,
                     performance::Performance,
                     encoder::MidiPerformanceEncoder,
                     numsteps::Int)

Generate performance events from the model using the given performance input.
`numsteps` determines the amount of steps to generate for, and the generated events
are appended to the input performance.
"""
function generate_events!(model::UnconditionalMusicTransformer,
                          performance::Performance,
                          encoder::MidiPerformanceEncoder,
                          numsteps::Int)

    targets = map(event -> encodeindex(event, encoder), performance)

    logits = model(targets)
    pred = sample!(performance, targets, logits, encoder)

    # Sample till end of sequence is encountered or numsteps is reached
    while pred != EOS_TOKEN && performance.numsteps <= numsteps
        logits = model(targets)
        pred = sample!(performance, targets, logits, encoder)
    end
end

"""
    generate(model::UnconditionalMusicTransformer;
             primer::Performance=Performance(DEFAULT_STEPS_PER_SECOND, velocity_bins=32),
             numsteps::Int=3000,
             as_notesequence::Bool=false,
             encoder::MidiPerformanceEncoder=default_performance_encoder())

Generate a musical performance from the `UnconditionalMusicTransformer` with the given
`primer`. If the `primer` is empty, a default event of `PerformanceEvent(TIME_SHIFT, 100)` will be
appended. The performance will be converted to and from one-hot indices using the `encoder`.

`numsteps` determines the amount of steps to generate for.
A step is 10 milliseconds or 0.01 seconds at the default steps per second rate of 100.
If `as_notesequence` is true, a `NoteSequence` is returned.
Otherwise, a midifile is returned.
"""
function generate(model::UnconditionalMusicTransformer;
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

"""
    generate_accompaniment(model::MelodyConditionedMusicTransformer,
                           melody::Vector{Int};
                           numsteps::Int=3000,
                           as_notesequence::Bool=false,
                           target_encoder::MidiPerformanceEncoder=default_performance_encoder())

Generate an accompaniment for the given melody using the `MelodyConditionedMusicTransformer`.

`numsteps` determines the amount of steps to generate for.
A step is 10 milliseconds or 0.01 seconds at the default steps per second rate of 100.
If `as_notesequence` is true, a `NoteSequence` is returned.
Otherwise, a midifile is returned.
"""
function generate_accompaniment(model::MelodyConditionedMusicTransformer,
                                melody::Vector{Int};
                                numsteps::Int=3000,
                                as_notesequence::Bool=false,
                                target_encoder::MidiPerformanceEncoder=default_performance_encoder())

    if isempty(melody)
        throw(error("Melody must not be empty"))
    end

    targets = [encodeindex(target_encoder.default_event, target_encoder)]

    @info "Generating..."
    performance = Performance(DEFAULT_STEPS_PER_SECOND, velocity_bins=32)

    logits, encoder_context = model(melody, targets)
    pred = sample!(performance, targets, logits, target_encoder)

    while pred != EOS_TOKEN && performance.numsteps <= numsteps
        embedded_targets = model.decoder_embedding(targets)
        logits = model.decoder(embedded_targets, encoder_context)
        pred = sample!(performance, targets, logits, target_encoder)
    end

    ns = getnotesequence(performance)
    as_notesequence == true && return ns
    notesequence_to_midi(ns)
end