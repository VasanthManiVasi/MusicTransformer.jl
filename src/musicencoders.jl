export MidiPerformanceEncoder
export encode_notesequence, decode_to_notesequence

using NoteSequences
using NoteSequences.PerformanceRepr
import NoteSequences.PerformanceRepr: encodeindex, decodeindex

const PAD_TOKEN = 1
const EOS_TOKEN = 2

"""
    MidiPerformanceEncoder <: Any

Encoder to convert between performance event indices and `NoteSequences`.

`MidiPerformanceEncoder` accounts for padding and EOS tokens, while
`PerformanceOneHotEncoding` does not.

## Fields
* `encoding::PerformanceOneHotEncoding`: Performance encoder object that is used for encoding.
* `steps_per_second::Int`: Amount of quantization steps per second.
* `num_velocitybins::Int`: Number of velocity bins.
* `add_eos::Bool`: If true, add the EOS token at the end while encoding.
"""
struct MidiPerformanceEncoder
    encoding::PerformanceOneHotEncoding
    steps_per_second::Int
    num_velocitybins::Int
    add_eos::Bool

    function MidiPerformanceEncoder(steps_per_second::Int,
                                    num_velocitybins::Int,
                                    minpitch::Int,
                                    maxpitch::Int,
                                    add_eos::Bool)

        encoding = PerformanceOneHotEncoding(num_velocitybins=num_velocitybins,
                                             max_shift_steps=steps_per_second,
                                             minpitch=minpitch,
                                             maxpitch=maxpitch)

        new(encoding, steps_per_second, num_velocitybins, add_eos)
    end
end

function Base.getproperty(encoder::MidiPerformanceEncoder, sym::Symbol)
    if sym === :num_reserved_tokens
        return 2 # PAD, EOS
    elseif sym === :vocab_size
        return encoder.encoding.num_classes + encoder.num_reserved_tokens
    else
        getfield(encoder, sym)
    end
end

"""
    encode_notesequence(ns::NoteSequence, encoder::MidiPerformanceEncoder)

Convert the given `NoteSequence` into performance event indices.
If `encoder.add_eos` is true, `EOS_TOKEN` will be added at the end.
"""
function encode_notesequence(ns::NoteSequence, encoder::MidiPerformanceEncoder)
    NoteSequences.absolutequantize!(ns, encoder.steps_per_second)
    performance = Performance(ns, velocity_bins=encoder.num_velocitybins)
    indices = [encodeindex(event, encoder) for event in performance]
    encoder.add_eos && push!(indices, EOS_TOKEN)

    indices
end

"""
    decode_to_notesequence(indices::Vector{Int}, encoder::MidiPerformanceEncoder)

Convert the given performance event indices into a `NoteSequence`.
"""
function decode_to_notesequence(indices::Vector{Int}, encoder::MidiPerformanceEncoder)
    performance = Performance(encoder.steps_per_second, velocity_bins=encoder.num_velocitybins)
    performance_events = [decodeindex(index, encoder) for index in indices]
    append!(performance, performance_events)

    getnotesequence(performance)
end

"""
    encodeindex(event::PerformanceEvent, encoder::MidiPerformanceEncoder)

Encode a `PerformanceEvent` into it's one-hot index using a `MidiPerformanceEncoder`.
"""
function encodeindex(event::PerformanceEvent, encoder::MidiPerformanceEncoder)
    index = encodeindex(event, encoder.encoding)
    index + encoder.num_reserved_tokens
end

"""
    decodeindex(index::Int, encoder::MidiPerformanceEncoder)

Decode a one-hot index into it's `PerformanceEvent` using a `MidiPerformanceEncoder`.
"""
function decodeindex(index::Int, encoder::MidiPerformanceEncoder)
    index -= encoder.num_reserved_tokens
    decodeindex(index, encoder.encoding)
end