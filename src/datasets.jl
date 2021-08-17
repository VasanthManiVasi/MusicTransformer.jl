export MaestroDataset, reader, @dataset_str

using Pkg.TOML
using DataDeps
using Random

using Transformers.Datasets
import Transformers.Datasets: trainfile, testfile, devfile, Mode

const dataset_configs = open(TOML.parse, joinpath(@__DIR__, "datasets.toml"))

# For clarity
macro dataset_str(name::String)
    :(@datadep_str($name))
end

struct MaestroDataset <: Dataset end

trainfile(::MaestroDataset) = dataset"Maestro_LM/train.jld2"
devfile(::MaestroDataset) = dataset"Maestro_LM/validation.jld2"
testfile(::MaestroDataset) = dataset"Maestro_LM/test.jld2"

function reader(::Type{M}, ds::MaestroDataset; batchsize=32, shuffle=true) where {M <: Mode}
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

function register_datasets(configs)
    for (dataset, config) in pairs(configs)
        description = config["description"]
        checksum = config["checksum"]
        url = config["url"]
        if startswith(url, "https://docs.google.com")
            fetch_method = Transformers.Datasets.download_gdrive
        else
            fetch_method = DataDeps.fetch_http
        end
        dep = DataDep(dataset, description, url, checksum;
                      fetch_method=fetch_method, post_fetch_method=DataDeps.unpack)
        DataDeps.register(dep)
    end
end

function list_datasets()
    println.(keys(dataset_configs))
    return
end
