module MusicTransformer

using Flux
using Transformers

include("layers.jl")
include("models.jl")
include("generate.jl")
include("pretrain.jl")
include("musicencoders.jl")

@init register_configs(configs)

end
