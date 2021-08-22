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
`minpitch::Int=21`
`maxpitch::Int=109`
`steps_per_second::Int=100`
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
    MidiPerformanceEncoder(DEFAULT_STEPS_PER_SECOND, VELOCITY_BINS,
                           MIN_PITCH, MAX_PITCH, add_eos)
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
                  primer::Performance=Performance(DEFAULT_STEPS_PER_SECOND, velocity_bins=VELOCITY_BINS),
                  numsteps::Int=3000,
                  as_notesequence::Bool=false,
                  encoder::MidiPerformanceEncoder=default_performance_encoder())

    performance = deepcopy(primer)

    targets = map(event -> encode_event(event, encoder), performance)
    pred = 0
    @info "Generating..."

    # Sample till EOS is encountered or numsteps is reached
    while pred != EOS_TOKEN && performance.numsteps <= numsteps
        t = targets |> device
        logits = model(t)
        probabilities = softmax(logits[:, end]) |> cpu
        pred = wsample(1:encoder.vocab_size, probabilities)

        # Add prediction to the model targets
        push!(targets, pred)

        # Add predicted event to performance
        push!(performance, decode_event(pred, encoder))
    end

    ns = getnotesequence(performance)
    as_notesequence == true && return ns
    midifile(ns)
end

"""
Generate a musical performance from the `UnconditionalMusicTransformer` with the given
`target_indices`. `target_indices` is the one-hot indices of performance events. If it is empty,
a zero matrix will be passed to the model (default token).

`decode_len` determines the length of the sequence to decode.
The generated performance will be decoded from the predicted indices using the `encoder`.
If `as_notesequence` is true, a `NoteSequence` is returned.
Otherwise, a midifile is returned.
"""
function generate(model::UnconditionalMusicTransformer,
                  target_indices::Vector{Int};
                  decode_len::Int=1024,
                  as_notesequence::Bool=false,
                  encoder::MidiPerformanceEncoder=default_performance_encoder())

    targets = deepcopy(target_indices)
    pred = 0
    @info "Generating..."

    # Sample till EOS is encountered or decoding length is reached
    while pred != EOS_TOKEN && length(targets) <= decode_len
        t = targets |> device
        logits = model(t)
        probabilities = softmax(logits[:, end]) |> cpu
        pred = wsample(1:encoder.vocab_size, probabilities)
        push!(targets, pred)
    end


    ns = decode_to_notesequence(targets, encoder)
    as_notesequence == true && return ns
    midifile(ns)
end

"""
    generate_accompaniment(model::MelodyConditionedMusicTransformer,
                           melody::Vector{Int};
                           decode_len::Int=2048,
                           as_notesequence::Bool=false,
                           target_encoder::MidiPerformanceEncoder=default_performance_encoder())

Generate an accompaniment for the given melody using the `MelodyConditionedMusicTransformer`.

`decode_len` determines the length of the sequence to decode.
The generated performance will be decoded from the predicted indices using `target_encoder`.
If `as_notesequence` is true, a `NoteSequence` is returned.
Otherwise, a midifile is returned.
"""
function generate_accompaniment(model::MelodyConditionedMusicTransformer,
                                melody::Vector{Int};
                                decode_len::Int=(4096 - length(inputs)),
                                as_notesequence::Bool=false,
                                target_encoder::MidiPerformanceEncoder=default_performance_encoder())

    if isempty(melody)
        throw(error("Melody must not be empty"))
    end

    targets = Vector{Int}()
    @info "Generating..."

    model = model |> device

    melody, targets = device.([melody, targets])
    logits, encoder_context = model(melody, targets)
    pred = wsample(1:target_encoder.vocab_size, softmax(logits[:, end]) |> cpu)
    push!(targets, pred)

    while pred != EOS_TOKEN && length(targets) <= decode_len
        t = targets |> device
        logits = model(t, encoder_context)
        probabilities = softmax(logits[:, end]) |> cpu
        pred = wsample(1:target_encoder.vocab_size, probabilities)
        push!(targets, pred)
    end

    ns = decode_to_notesequence(targets, target_encoder)
    as_notesequence == true && return ns
    midifile(ns)
end