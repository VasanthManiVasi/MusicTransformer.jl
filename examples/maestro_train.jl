using Flux
using Flux: onehotbatch, label_smoothing, update!
using Zygote

using Random
using FileIO
using Dates
using JLD2

using MusicTransformer
using Transformers
using Transformers.Basic, Transformers.Datasets
import Transformers.Datasets: dataset, datafile, reader, Mode

struct MaestroDataset <: Dataset end

Transformers.Datasets.testfile(::MaestroDataset) = "/home/user/Downloads/maestro_test.jld2"
Transformers.Datasets.trainfile(::MaestroDataset) = "/home/user/Downloads/maestro_train.jld2"
Transformers.Datasets.devfile(::MaestroDataset) = "/home/user/Downloads/maestro_validation.jld2"

function reader(::Type{M}, ds::MaestroDataset; batchsize=32, shuffle=false) where {M <: Mode}
    @assert batchsize >= 1

    chan = Channel{Vector{Vector{Int}}}()
    filename = datafile(M, ds)
    num_examples = load(filename, "num_examples")
    load_order = shuffle ? randperm(num_examples) : collect(1:num_examples)

    loader = @async begin
        for batch_load_order in Iterators.partition(load_order, batchsize)
            length(batch_load_order) != batchsize && continue
            data = collect(load(filename, string.(batch_load_order)...))
            put!(chan, ifelse(batchsize == 1, [data], data))
        end
    end

    bind(chan, loader)
    chan
end

Base.@kwdef struct HParams
    num_classes::Int=310
    lr::Float64=1e-6
    epochs::Int=10
    crop_len::Int=2048
    batchsize::Int=8
    shuffle::Bool=true
    α::Float64=0.1
    log_at_steps::Int=25
    checkpoint_at_steps::Int=1000
end

"""
    preprocess(data::Vector{Vector{Int}},
                crop_len::Int=2048,
                num_classes::Int=1,
                α::Float64=0.1)

Return X and Y batches of the data after preprocessing by taking random crops.
Label smoothing is applied to the targets.
"""
function preprocess(data::Vector{Vector{Int}},
                    crop_len::Int=2048,
                    num_classes::Int=1,
                    α::Float64=0.1)

    batchsize = length(data)
    batch = Matrix{Int}(undef, batchsize, crop_len+1)

    for i = 1:batchsize
        offset = length(data[i]) - crop_len
        crop_begin = rand(1:offset)
        crop_end = crop_begin + crop_len
        batch[i, :] = @view data[i][crop_begin:crop_end]
    end

    X = batch[:, 1:end-1]
    # Shift inputs right and one-hot encode them
    Y = onehotbatch(batch[:, 2:end], 1:num_classes)

    X, label_smoothing(Y, α)
end

"""
    train!(model; hparams::HParams=HParams(), checkpoint_dir="/tmp/maestro_model")

Train the model with the given hyperparameters.
The model weights are saved after training.
"""
function train!(model; hparams::HParams=HParams(), checkpoint_dir="/tmp/maestro_model")
    !isdir(checkpoint_dir) && mkdir(checkpoint_dir)
    ps = params(model)
    opt = ADAM(hparams.lr)

    @info "Training..."
    for i = 1:hparams.epochs
        trainingstep = 1
        datareader = reader(Train, MaestroDataset(),
                        batchsize=hparams.batchsize, shuffle=hparams.shuffle)

        for data in datareader
            X, Y = preprocess(data, hparams.crop_len, hparams.num_classes, hparams.α)

            loss, back = Zygote.pullback(ps) do
                logits = model(X)
                logcrossentropy(Y, logsoftmax(logits))
            end
            grads = back(1f0)

            if trainingstep % hparams.log_at_steps == 0
                println("Epoch: $i   Step: $trainingstep   Loss: $loss")
            end

            if trainingstep % hparams.checkpoint_at_steps == 0
                save_path = joinpath(checkpoint_dir, "model_checkpoint_$(now()).jld2")
                JLD2.@save save_path weights=ps
            end

            update!(opt, ps, grads)
            trainingstep += 1
        end
    end

    save_path = joinpath(checkpoint_dir, "model_checkpoint_$(now()).jld2")
    JLD2.@save save_path weights=ps

    @info "Saved model weights at $save_path"
end

# Usage:
# model = BaselineMusicTransformer(512, 8, 2048, 6)
# hparams = HParams(epochs=15, crop_len=1028, log_at_steps=10)
# train!(model, hparams=hparams)
