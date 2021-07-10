export MusicTransformerBlock, PositionEmbeddingT2T

using Flux: @functor
using Tullio

using Transformers.Basic
using Transformers.Basic: AbstractTransformer, AbstractAttention, MultiheadAttention, PwFFN
using Transformers: Abstract3DTensor, batchedmul

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

struct MultiheadRelativeAttention{R<:AbstractArray, Q<:Dense, K<:Dense, V<:Dense, O<:Dense, DP<:Dropout} <: AbstractAttention
    head::Int
    future::Bool
    relative_embedding::R
    iqproj::Q
    ikproj::K
    ivproj::V
    oproj::O
    drop::DP
end

function Flux.functor(mh::MultiheadRelativeAttention)
    (mh.relative_embedding, mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj), m -> MultiheadRelativeAttention(
                                                                                    mh.head,
                                                                                    mh.future,
                                                                                    m...,
                                                                                    mh.drop)
end

function MultiheadRelativeAttention(head::Int,
                                    is::Int,
                                    hs::Int,
                                    os::Int,
                                    max_relative_position::Int;
                                    future::Bool=true,
                                    share_relative_embedding::Bool=true,
                                    pdrop = 0.1)

    if share_relative_embedding
        relative_embedding = randn(hs, max_relative_position)
    else
        relative_embedding = randn(hs, max_relative_position, heads)
    end

    MultiheadRelativeAttention(head, future, relative_embedding,
                              Dense(is, hs*head), Dense(is, hs*head), Dense(is, hs*head), Dense(hs*head, os),
                              Dropout(pdrop))
end

function Base.show(io::IO, mh::MultiheadRelativeAttention)
    hs = div(size(mh.iqproj.W)[1], mh.head)
    is = size(mh.iqproj.W)[end]
    os = size(mh.oproj.W)[1]

    print(io, "MultiheadRelativeAttention(")
    print(io, "head=$(mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "$(is)=>$(os)")

    if Flux.istraining()
        print(io, ", dropout=$(mh.drop.p))")
    else
        print(io, ")")
    end
end

function (mh::MultiheadRelativeAttention)(query::A1,
                                          key::A2,
                                          value::A3;
                                          mask=nothing) where {T,
                                                               A1 <: AbstractMatrix{T},
                                                               A2 <: AbstractMatrix{T},
                                                               A3 <: AbstractMatrix{T}}

    # size(query) == (dims, seq_len)
    ipq = mh.iqproj(query)
    ipk = mh.ikproj(key)
    ipv = mh.ivproj(value)

    h = size(ipq)[1] #h == hs * head
    hs = div(h, mh.head)

    #size(hq) == (hs, seq_len, head)
    hq = permutedims(reshape(ipq, hs, mh.head, :), [1, 3, 2])
    hk = permutedims(reshape(ipk, hs, mh.head, :), [1, 3, 2])
    hv = permutedims(reshape(ipv, hs, mh.head, :), [1, 3, 2])

    atten = relative_attention(hq, hk, hv, mh.relative_embedding,
                               mask, mh.future, mh.drop)

    # size(atten) == (head*hs, seq_len)
    atten = reshape(permutedims(atten, [1, 3, 2]), h, :)

    mh.oproj(atten)
end

function relative_attention(query::A1,
                            key::A2,
                            value::A3,
                            relative_embedding::A4,
                            mask, future::Bool,
                            dropout) where {T,
                                            A1 <: Abstract3DTensor{T},
                                            A2 <: Abstract3DTensor{T},
                                            A3 <: Abstract3DTensor{T},
                                            A4 <: Union{Abstract3DTensor{T}, AbstractMatrix{T}}}

    #size(query) == (dims, {q,k}_seq_len, batch) == size(key) == size(value)
    #size(score) == (k_seq_len, q_seq_len, batch)
    dk = size(key, 1)
    score = batchedmul(key, query; transA = true)

    # if share_relative_embedding: size(rel_embed) == (dims, q_seq_len)
    # else:                        size(rel_embed) == (dims, q_seq_len, heads)
    seq_len = size(key, 2)
    rel_embed = get_relative_embedding(relative_embedding, seq_len)

    # size(rel_score) == (k_seq_len, q_seq_len, batch)
    rel_score = mul_relative_keys(query, rel_embed)
    rel_score = _relative_to_absolute_position(rel_score)

    # Add relative positional information to attention score
    score += rel_score

    score = score ./ convert(T, sqrt(dk))

    score = Transformers.Basic.apply_mask(score, mask, future)
    score = softmax(score; dims=1)
    dropout !== nothing && (score = dropout(score))
    batchedmul(value, score) #size(return) == (dims, q_seq_len, batch)
end

get_relative_embedding(r::AbstractMatrix{T}, seq_len::Int) where {T} = r[:, end-seq_len+1:end]
get_relative_embedding(r::Abstract3DTensor{T}, seq_len::Int) where {T} = r[:, end-seq_len+1:end, :]

function mul_relative_keys(x::Abstract3DTensor{T}, y::AbstractMatrix{T}) where {T}
    # Heads share relative embedding
    @tullio z[m, l, h] := x[d, l, h] * y[d, m]
end

function mul_relative_keys(x::Abstract3DTensor{T}, y::Abstract3DTensor{T}) where {T}
    # Heads don't share relative embedding
    @tullio z[m, l, h] := x[d, l, h] * y[d, m, h]
end

function _relative_to_absolute_position(x::Abstract3DTensor{T}) where {T}
    # size(x) == (seq_len, seq_len, heads)
    _, seq_len, heads = size(x)

    # Padding
    x = vcat(zeros(T, 1, seq_len, heads), x)

    # Reshape
    x = reshape(x, (seq_len, seq_len+1, :))
    x = permutedims(x, (2, 1, 3))

    # Slice
    x = x[2:end, :, :]

    x
end