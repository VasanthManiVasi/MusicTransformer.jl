module MusicTransformer

using Transformers, Flux

include("layers.jl")
include("models.jl")
include("pretrain.jl")

end
