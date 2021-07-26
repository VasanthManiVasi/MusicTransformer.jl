module MusicTransformer

using Transformers, Flux

include("layers.jl")
include("models.jl")
include("generate.jl")
include("pretrain.jl")
include("musicencoders.jl")

@init register_configs(configs)

end
