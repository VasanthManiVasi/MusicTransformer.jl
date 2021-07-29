using CSV
using JLD2
using FileIO
using DataStructures

using NoteSequences
using NoteSequences: DEFAULT_STEPS_PER_SECOND

using MusicTransformer

const MIN_PITCH = 21
const MAX_PITCH = 108
const VELOCITY_BINS = 32

# Data augmentation constants
const TRANSPOSE_AMOUNTS = [-3, -2, -1, 0, 1, 2, 3]
const STRETCH_FACTORS = [0.95, 0.975, 1.0, 1.025, 1.05]

"""
    filter_invalid_notes!(ns::NoteSequence, minpitch, maxpitch)

Filter pitches that are out of range in the `NoteSequence`.
"""
function filter_invalid_notes!(ns::NoteSequence, minpitch, maxpitch)
    ns.notes = [note for note in ns.notes if minpitch <= note.pitch <= maxpitch]
    ns
end

"""
    augment_notesequence(ns::NoteSequence,
                         stretchfactor::Float64, transposeamount::Int,
                         minpitch::Int, maxpitch::Int)

Augment the `NoteSequence` by temporal stretching and transposition.
"""
function augment_notesequence(ns::NoteSequence,
                              stretchfactor::Float64, transposeamount::Int,
                              minpitch::Int, maxpitch::Int)

    augmented_ns = deepcopy(ns)
    NoteSequences.temporalstretch!(augmented_ns, stretchfactor)
    NoteSequences.transpose(augmented_ns, transposeamount, minpitch, maxpitch)
end

"""
    extract_examples(ns::NoteSequence,
                     augment_values::Matrix{Tuple{Float64, Int64}},
                     encoder::MidiPerformanceEncoder,
                     minpitch::Int,
                     maxpitch::Int)

Extract training examples from the given notesequence after performing augmentation.

If there were any deleted notes after transposition due to the pitches going out of range,
that specific augmented example is not included in the training set.
"""
function extract_examples(ns::NoteSequence,
                          augment_values::Matrix{Tuple{Float64, Int64}},
                          encoder::MidiPerformanceEncoder,
                          minpitch::Int,
                          maxpitch::Int)

    examples = Vector{Tuple{Int, Vector{Int}}}()
    l = Threads.SpinLock()

    Threads.@threads for (i, augment_value_pair) in collect(enumerate(augment_values))
        stretchfactor, transposeamount = augment_value_pair
        augmented_ns, num_deleted = augment_notesequence(ns, stretchfactor, transposeamount,
                                                         minpitch, maxpitch)
        if num_deleted == 0
            indices = encode_notesequence(augmented_ns, encoder)
            lock(l)
            try
                push!(examples, (i, indices))
            finally
                unlock(l)
            end
        end
    end
    sort!(examples, by=ex->ex[1])

    # Extracting only the actual examples
    [example[2] for example in examples]
end

"""
    generate_examples(maestro_data_dir::String,
                      output_dir::String,
                      transposeamounts::Vector{Int}=TRANSPOSE_AMOUNTS,
                      stretchfactors::Vector{Float64}=STRETCH_FACTORS,
                      steps_per_second::Int=DEFAULT_STEPS_PER_SECOND,
                      num_velocitybins::Int=VELOCITY_BINS,
                      minpitch::Int=MIN_PITCH,
                      maxpitch::Int=MAX_PITCH,
                      add_eos::Bool=false)

Transform raw midi files from the MAESTRO Dataset to one-hot indices
after performing augmentation.

The midifiles are split into train/test/val splits based on the metadata
given in maestro-v3.0.0.csv. The generated data will be stored in their respective split files
with numbers as their identifiers.
"""
function generate_examples(maestro_data_dir::String,
                           output_dir::String,
                           transposeamounts::Vector{Int}=TRANSPOSE_AMOUNTS,
                           stretchfactors::Vector{Float64}=STRETCH_FACTORS,
                           steps_per_second::Int=DEFAULT_STEPS_PER_SECOND,
                           num_velocitybins::Int=VELOCITY_BINS,
                           minpitch::Int=MIN_PITCH,
                           maxpitch::Int=MAX_PITCH,
                           add_eos::Bool=false)

    "maestro-v3.0.0.csv" ∉ readdir(maestro_data_dir) &&
        throw(error("MAESTRO metadata csv file is not found."))

    !isdir(output_dir) && mkdir(output_dir)
    maestro_csv = CSV.File(joinpath(maestro_data_dir, "maestro-v3.0.0.csv"))
    augment_values = permutedims(collect(Iterators.product(stretchfactors, transposeamounts)), (2, 1))
    encoder = MidiPerformanceEncoder(steps_per_second, num_velocitybins, minpitch, maxpitch, add_eos)

    # Keep track of the number of examples in each mode (train, test, validation)
    num_examples = DefaultDict{String, Int}(0)

    for (rootdir, _, filenames) in collect(walkdir(maestro_data_dir))
        for filename in filenames
            if ismidifile(filename)
                year = rootdir[end-3:end]
                filepath = joinpath(rootdir, filename)

                midi = load(filepath)
                ns = midi_to_notesequence(midi)
                NoteSequences.applysustainchanges!(ns)
                filter_invalid_notes!(ns, minpitch, maxpitch)

                split_mode = maestro_split_mode(year, filename, maestro_csv)
                if split_mode == "train"
                    # Extract training examples after performing augmentation
                    examples = extract_examples(ns, augment_values, encoder, minpitch, maxpitch)
                else
                    # No need to perform augmentation for validation and test examples
                    examples = [encode_notesequence(ns, encoder)]
                end

                save_examples(examples, output_dir, split_mode, num_examples[split_mode])
                num_examples[split_mode] += length(examples)
                println("Converted $filename")
            end
        end
    end

    for mode in ["train", "validation", "test"]
        num_examples[mode] == 0 && continue
        filename = joinpath(output_dir, mode * ".jld2")
        jldopen(filename, "a+") do file
            file["num_examples"] = num_examples[mode]
        end
    end
end

ismidifile(filename::String) = query(filename) isa File{format"MIDI"}

"""
    maestro_split_mode(year::String, midi_filename::String, maestro_csv::CSV.File)

Return the MAESTRO dataset split mode from the MAESTRO metadata csv for the given midifile.
"""
function maestro_split_mode(year::String, midi_filename::String, maestro_csv::CSV.File)
    midi_filename = joinpath(year, midi_filename)
    index = findfirst(name -> name == midi_filename, maestro_csv.midi_filename)
    maestro_csv[index].split
end

"""
    save_examples(examples::Vector{Vector{Int}}, output_dir::String, mode::String, num_examples::Int)

Save the examples in the `output_dir` with a filename of `mode`.jld2.
The identifier for each example is a number, starting from `num_examples` + 1.
"""
function save_examples(examples::Vector{Vector{Int}}, output_dir::String, mode::String, num_examples::Int)
    filename = joinpath(output_dir, mode * ".jld2")
    jldopen(filename, "a+") do file
        for i = 1:length(examples)
            file[string(num_examples + i)] = examples[i]
        end
    end
end

# data_dir = "/home/user/Downloads/midis/maestro-v3.0.0/"
# generate_examples(data_dir, "/tmp/maestro_data")
