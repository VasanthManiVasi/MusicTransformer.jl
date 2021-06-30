using JLD2
using Flux: loadparams!
using Requires

"""     readckpt(path)
Load weights from a tensorflow checkpoint file into a Dict.
"""
readckpt(path) = error("readckpt require TensorFlow.jl installed. run `Pkg.add(\"TensorFlow\"); using TensorFlow`")

@init @require TensorFlow="1d978283-2c37-5f34-9a8e-e9c0ece82495" begin
    import .TensorFlow
    function readckpt(path)
        weights = Dict{String, Array}()
        TensorFlow.init()
        ckpt = TensorFlow.pywrap_tensorflow.x.NewCheckpointReader(path)
        shapes = ckpt.get_variable_to_shape_map()

        for (name, shape) ∈ shapes
            # Ignore training related data - e.g. learning rates
            # Also ignore scalars and other variables that aren't stored correctly in the checkpoint (shape == Any[])
            (occursin("training", name) || shape == Any[]) && continue
            weight = ckpt.get_tensor(name)
            if length(shape) == 2
                weight = collect(weight')
            end
            weights[name] = weight
        end

        weights
    end
end

"""     ckpt2_to_jld2(ckptpath::String, ckptname::String, savepath::String)
Loads the pre-trained model weights from a tensorflow checkpoint and saves to JLD2
"""
function ckpt_to_jld2(ckptpath::String, ckptname::String; savepath::String="./")
    files = readdir(ckptpath)
    ckptname ∉ files && error("The checkpoint file $ckptname is not found")
    ckptname*".meta" ∉ files && error("The checkpoint meta file is not found")
    weights = readckpt(joinpath(ckptpath, ckptname))
    jld2name = normpath(joinpath(savepath, ckptname[1:end-5]*".jld2"))
    @info "Saving the model weights to $jld2name"
    JLD2.@save jld2name weights
end