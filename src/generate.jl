using NoteSequences
using StatsBase: wsample

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
    performance
end

function sample_and_append!(logits::Array, performance::Performance, inputs::Vector{Int})
    prediction = wsample(performance.labels, softmax(logits[:, end]))
    # Add prediction to the model inputs
    push!(inputs, prediction)
    # Add predicted event to performance
    push!(performance, decodeindex(prediction, performance))

    prediction
end

function generate_events!(model::MusicTransformerModel, performance::Performance, numsteps::Int)
    # Take account of PAD and EOS by adding 2 to encodeindex
    inputs = map(event -> encodeindex(event, performance) + 2, performance)

    logits = model(inputs)
    pred = sample_and_append!(logits, performance, inputs)

    # Sample till end of sequence is encountered (EOS = 2) or numsteps is reached
    while pred != 2 && performance.numsteps <= numsteps
        logits = model(inputs)
        pred = sample_and_append!(logits, performance, inputs)
    end
end

function generate(model::MusicTransformerModel;
                  primer::Vector{PerformanceEvent}=[PerformanceEvent(TIME_SHIFT, 100)],
                  numsteps::Int = 3000)

    performance = default_performance()
    performance.events = deepcopy(primer)
    generate_events!(model, performance, numsteps)
    getnotesequence(performance)
end