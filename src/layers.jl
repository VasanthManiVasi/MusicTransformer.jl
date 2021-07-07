using Flux: @functor
using Transformers.Basic
using Transformers.Basic: AbstractTransformer

struct MusicTransformerBlock{MA<:MultiheadAttention, LA<:LayerNorm, P<:PwFFN, LP<:LayerNorm, DP<:Dropout} <: AbstractTransformer
    mh::MA
    mhn::LA
    pw::P
    pwn::LP
    drop::DP
end

@functor MusicTransformerBlock

function MusicTransformerBlock(size::Int, head::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    MusicTransformerBlock(size, head, div(size, head), ps;future=future, act=act, pdrop=pdrop)
end

MusicTransformerBlock(size::Int, head::Int, hs::Int, ps::Int; future::Bool = true, act = relu, pdrop = 0.1) = MusicTransformerBlock(
    MultiheadAttention(head, size, hs, size; future=future, pdrop=pdrop),
    LayerNorm(size),
    PwFFN(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (t::MusicTransformerBlock)(x::A, mask=nothing) where {T, N, A<:AbstractArray{T, N}}
    dropout = t.drop

    # Layer norm is a preprocess for the unconditional 16L Music Transformer,
    # Dropout and addition (residual connection) are postprocesses

    # Attention layer
    x_normed = t.mhn(x) # LayerNorm
    a = t.mh(x_normed, x_normed, x_normed; mask=mask) # MultiheadAttention
    a = dropout(a) # Dropout
    res_a = x + a # Addition (residual)

    # Feed-forward layer
    res_a_normed = t.pwn(res_a) # LayerNorm
    pwffn = t.pw(res_a_normed) # Pointwise feed-forward
    pwffn = dropout(pwffn) # Dropout
    res_pwffn = res_a + pwffn # Addition
    res_pwffn
end

function Base.show(io::IO, t::MusicTransformerBlock)
    hs = div(size(t.mh.iqproj.W)[1], t.mh.head)
    h, ps = size(t.pw.dout.W)

    print(io, "MusicTransformerBlock(")
    print(io, "head=$(t.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h)")
    if Flux.istraining()
        print(io, ", dropout=$(t.drop.p))")
    else
        print(io, ")")
    end
end

function PositionEmbeddingT2T(size::Int, max_len::Int = 2048)
    # Follows the tensor2tensor implementation - which is used by the unconditional 16L Music Transformer
    num_timescales = size / 2
    positions = Float32.(collect(0.0:max_len-1))
    log_timescale_increment = log(1e4) / (num_timescales-1)
    inv_timescales = Float32.(exp.(collect(0.0:(num_timescales-1)) .* -log_timescale_increment))
    scaled_time = unsqueeze(positions, 2) * unsqueeze(inv_timescales, 1)
    timing_signals = hcat(sin.(scaled_time), cos.(scaled_time))
    PositionEmbedding(false, collect(timing_signals'))
end