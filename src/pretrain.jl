using JLD2, Requires, Pkg.TOML, DataDeps
using Flux: loadparams!

export list_pretrains, load_pretrain, @pretrain_str

const configs = open(TOML.parse, joinpath(@__DIR__, "pretrains.toml"))

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
Loads the pre-trained model weights from a TensorFlow checkpoint and saves to JLD2
"""
function ckpt_to_jld2(ckptpath::String, ckptname::String; savepath::String="./")
    weights = readckpt(joinpath(ckptpath, ckptname))
    jld2name = normpath(joinpath(savepath, ckptname[1:end-5]*".jld2"))
    @info "Saving the model weights to $jld2name"
    JLD2.@save jld2name weights
end

function register_configs(configs)
    for (model_name, config) in pairs(configs)
        model_desc = Transformers.Pretrain.description(config["description"], config["host"], config["link"])
        checksum = config["checksum"]
        url = config["url"]
        dep = DataDep(model_name, model_desc, url, checksum;
                      fetch_method=Transformers.Datasets.download_gdrive)
        DataDeps.register(dep)
    end
end

"""     list_pretrains()
List all the available pre-trained models.
"""
function list_pretrains()
    println.(keys(configs))
    return
end

"""     load_pretrain(path)
Loads a pre-trained Music Transformer model.
"""
function load_pretrain(model_name::String)
    if model_name ∉ keys(configs)
        error("""Invalid model.
               Please try list_pretrains() to check the available pre-trained models""")
    end

    model_config = configs[model_name]
    loader = loading_method(Val(Symbol(model_config["model_type"])))

    model_path = @datadep_str("$model_name/$model_name.jld2")
    if !endswith(model_path, ".jld2")
        error("""Invalid file. A .jld2 file is required to load the model.
                 If this is a tensorflow checkpoint file, run ckpt_to_jld2 to convert it.""")
    end

    JLD2.@load model_path weights
    loader(weights, model_config)
end

loading_method(::Val{:unconditional}) = load_unconditional_musictransformer

# Macro from Transformers.jl
macro pretrain_str(name)
    :(load_pretrain($(esc(name))))
end

function load_unconditional_musictransformer(weights, config)
    num_layers = config["num_layers"]
    heads = config["heads"]
    depth = config["depth"]
    ffn_depth = config["ffn_depth"]

    mt = BaselineMusicTransformer(depth, heads, ffn_depth, num_layers)

    layers = keys(weights)

    for i = 1:num_layers
        # Get all layers in a block
        block = filter(l->occursin("layer_$(i-1)/", l), layers)
        # Index of the block in Transformer body - after the embedding layers
        block_index = i + 3
        for k ∈ block
            if occursin("self_attention", k)
                if occursin("q/kernel", k)
                    loadparams!(mt[block_index].mh.iqproj.W, [weights[k]])
                elseif occursin("k/kernel", k)
                    loadparams!(mt[block_index].mh.ikproj.W, [weights[k]])
                elseif occursin("v/kernel", k)
                    loadparams!(mt[block_index].mh.ivproj.W, [weights[k]])
                elseif occursin("output_transform/kernel", k)
                    loadparams!(mt[block_index].mh.oproj.W, [weights[k]])
                elseif occursin("layer_norm_scale", k)
                    loadparams!(mt[block_index].mhn.diag.α, [weights[k]])
                elseif occursin("layer_norm_bias", k)
                    loadparams!(mt[block_index].mhn.diag.β, [weights[k]])
                else
                    @warn "Unknown variable: $k"
                end
            elseif occursin("ffn", k)
                if occursin("conv1/kernel", k)
                    loadparams!(mt[block_index].pw.din.W, [weights[k]])
                elseif occursin("conv1/bias", k)
                    loadparams!(mt[block_index].pw.din.b, [weights[k]])
                elseif occursin("conv2/kernel", k)
                    loadparams!(mt[block_index].pw.dout.W, [weights[k]])
                elseif occursin("conv2/bias", k)
                    loadparams!(mt[block_index].pw.dout.b, [weights[k]])
                elseif occursin("layer_norm_scale", k)
                    loadparams!(mt[block_index].pwn.diag.α, [weights[k]])
                elseif occursin("layer_norm_bias", k)
                    loadparams!(mt[block_index].pwn.diag.β, [weights[k]])
                else
                    @warn "Unknown variable: $k"
                end
            else
                @warn "Unknown variable: $k"
            end
        end
    end

    output_norm = filter(l->occursin("transformer/body/decoder/layer_prepostprocess/layer_norm", l), layers)
    block_index = 3 + num_layers + 1
    for k ∈ output_norm
        if occursin("layer_norm_scale", k)
            loadparams!(mt[block_index].diag.α, [weights[k]])
        elseif occursin("layer_norm_bias", k)
            loadparams!(mt[block_index].diag.β, [weights[k]])
        end
    end

    # Load embedding
    embedding = weights["transformer/symbol_modality_310_512/shared/weights_0"]
    loadparams!(mt.ts[1], [embedding])

    # This pre-trained Music Transformer shares embedding and softmax weights
    # Base.lastindex is not defined on Stack, manually write the last index for now
    # 3 embedding layers, 16 transformer body blocks, 1 output layer norm, + 1 is the last embedding layer
    last_layer = 3 + num_layers + 1 + 1
    loadparams!(mt.ts[last_layer].W, [embedding'])

    mt
end