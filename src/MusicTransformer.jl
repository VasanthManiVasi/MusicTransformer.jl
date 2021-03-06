module MusicTransformer

using CUDA
using Flux
using Transformers

include("layers.jl")
include("models.jl")
include("musicencoders.jl")
include("generate.jl")
include("pretrain.jl")
include("datasets.jl")

device = Flux.cpu

if CUDA.has_cuda()
    device = Flux.gpu
    println("GPU is enabled.")
    CUDA.allowscalar(false)
end

function call_registers()
    register_configs(pretrained_configs)
    register_datasets(dataset_configs)
end

@init call_registers()

end
