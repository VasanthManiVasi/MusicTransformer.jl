export MusicTransformerBlock, PositionEncoding

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

"""
    MusicTransformerBlock(size::Int, head::Int, ps::Int;
                          future::Bool = true, act = relu, pdrop = 0.1)
    MusicTransformerBlock(size::Int, head::Int, hs::Int, ps::Int;
                          future::Bool = true, act = relu, pdrop = 0.1)

Return a Pre-LayerNorm Transformer Encoder.
"""
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

    # Layer norm is a preprocess for the unconditional Music Transformer,
    # Dropout and residual connection are postprocesses

    # Self-Attention layer
    x_normed = t.mhn(x)
    a = t.mh(x_normed, x_normed, x_normed; mask=mask)
    a = dropout(a)
    res_a = x + a

    # Feed-forward layer
    res_a_normed = t.pwn(res_a)
    pwffn = t.pw(res_a_normed)
    pwffn = dropout(pwffn)
    res_pwffn = res_a + pwffn
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

struct MusicTransformerDecoder{MA<:MultiheadAttention, LA<:LayerNorm,
                               IMA<:MultiheadAttention, ILA<:LayerNorm,
                               P<:PwFFN, LP<:LayerNorm, DP<:Dropout} <: AbstractTransformer
    mh::MA
    mhn::LA
    imh::IMA
    imhn::ILA
    pw::P
    pwn::LP
    drop::DP
end

@functor MusicTransformerDecoder

"""
    MusicTransformerDecoder(size::Int, head::Int, ps::Int;
                            act = relu, pdrop = 0.1)
    MusicTransformerDecoder(size::Int, head::Int, hs::Int, ps::Int;
                            act = relu, pdrop = 0.1)

Return a Pre-LayerNorm Transformer Decoder.
"""
function MusicTransformerDecoder(size::Int, head::Int, ps::Int; act = relu, pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    MusicTransformerDecoder(size, head, div(size, head), ps; act=act, pdrop=pdrop)
end

MusicTransformerDecoder(size::Int, head::Int, hs::Int, ps::Int; act = relu, pdrop = 0.1) = MusicTransformerDecoder(
    MultiheadAttention(head, size, hs, size; future=false, pdrop=pdrop),
    LayerNorm(size),
    MultiheadAttention(head, size, hs, size; future=true, pdrop=pdrop),
    LayerNorm(size),
    PwFFN(size, ps, act),
    LayerNorm(size),
    Dropout(pdrop),
)

function (td::MusicTransformerDecoder)(x::AbstractArray{T,N}, m, mask=nothing) where {T,N}
    dropout = td.drop

    # Self-Attention layer
    x_normed = td.mhn(x)
    a = td.mh(x_normed, x_normed, x_normed)
    a = dropout(a)
    res_a = x + a

    # Encoder-Decoder attention
    res_a_normed = td.imhn(res_a)
    ia = td.imh(res_a_normed, m, m, mask=mask)
    ia = dropout(ia)
    res_ia = res_a + ia

    # Feed-forward layer
    res_ia_normed = td.pwn(res_ia)
    pwffn = td.pw(res_ia_normed)
    pwffn = dropout(pwffn)
    res_pwffn = res_ia + pwffn
    res_pwffn
end

function Base.show(io::IO, td::MusicTransformerDecoder)
    hs = div(size(td.imh.iqproj.W)[1], td.imh.head)
    h, ps = size(td.pw.dout.W)

    print(io, "MusicTransformerDecoder(")
    print(io, "head=$(td.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h)")
    if Flux.istraining()
        print(io, ", dropout=$(td.drop.p))")
    else
        print(io, ")")
    end
end

function PositionEncoding(size::Int, max_len::Int = 2048)
    # Follows the tensor2tensor implementation - which is used by the Music Transformer
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
        relative_embedding = randn(Float32, hs, max_relative_position)
    else
        relative_embedding = randn(Float32, hs, max_relative_position, heads)
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
                                                       A1 <: Abstract3DTensor{T},
                                                       A2 <: Abstract3DTensor{T},
                                                       A3 <: Abstract3DTensor{T}}
    qs = size(query)
    ks = size(key)
    vs = size(value)

      #size(ipq) == (h, q_seq_len, batch)
    ipq = @toNd mh.iqproj(query)
    ipk = @toNd mh.ikproj(key)
    ipv = @toNd mh.ivproj(value)

    h = size(ipq, 1)
    hs = div(h, mh.head)

    #size(ipq) == (hs, q_seq_len, head, batch)
    ipq = permutedims(reshape(ipq, hs, mh.head, qs[2], qs[3]), [1, 3, 2, 4])
    ipk = permutedims(reshape(ipk, hs, mh.head, ks[2], ks[3]), [1, 3, 2, 4])
    ipv = permutedims(reshape(ipv, hs, mh.head, vs[2], vs[3]), [1, 3, 2, 4])

    #size(ipq) == (hs, q_seq_len, head * batch)
    ipq = reshape(ipq, hs, qs[2], :)
    ipk = reshape(ipk, hs, ks[2], :)
    ipv = reshape(ipv, hs, vs[2], :)

    atten = relative_attention(ipq,ipk,ipv, mh.relative_embedding,
                               mask, mh.future, mh.drop)

    atten = permutedims(reshape(atten, hs, qs[2], mh.head, qs[3]), [1, 3, 2, 4]) #size(atten) == (hs, head, ql, b)
    atten = reshape(atten, h, qs[2], qs[3]) #size(atten) == (h, ql, b)

    out = @toNd mh.oproj(atten)
    out #size(out) == (h, q_seq_len, batch)
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

struct TransformerRelative{MA<:MultiheadRelativeAttention, LA<:LayerNorm, P<:PwFFN, LP<:LayerNorm, DP<:Dropout} <: AbstractTransformer
    mh::MA
    mhn::LA
    pw::P
    pwn::LP
    drop::DP
end

@functor TransformerRelative

"""

"""
function TransformerRelative(size::Int, head::Int, ps::Int, max_relative_position::Int;
                            future::Bool = true, act = relu, pdrop = 0.1)

    rem(size, head) != 0 && error("size not divisible by head")
    TransformerRelative(size, head, div(size, head), ps, max_relative_position;
                        future=future, act=act, pdrop=pdrop)
end

function TransformerRelative(size::Int, head::Int, hs::Int, ps::Int, max_relative_position::Int;
                             future::Bool = true, act = relu, pdrop = 0.1)

    TransformerRelative(
        MultiheadRelativeAttention(head, size, hs, size, max_relative_position; future=future, pdrop=pdrop),
        LayerNorm(size),
        PwFFN(size, ps, act),
        LayerNorm(size),
        Dropout(pdrop),
    )
end

function (t::TransformerRelative)(x::A, mask=nothing) where {T, N, A<:AbstractArray{T, N}}
    dropout = t.drop

    # Attention layer
    a = t.mh(x, x, x; mask=mask)
    a = dropout(a)
    res_a = x + a
    res_a = t.mhn(x)

    # Feed-forward layer
    pwffn = t.pw(res_a)
    pwffn = dropout(pwffn)
    res_pwffn = res_a + pwffn
    res_pwffn = t.pwn(res_a)
    res_pwffn
end

function Base.show(io::IO, t::TransformerRelative)
    hs = div(size(t.mh.iqproj.W)[1], t.mh.head)
    h, ps = size(t.pw.dout.W)
    m = size(t.mh.relative_embedding, 2)

    print(io, "TransformerRelative(")
    print(io, "head=$(t.mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h), ")
    print(io, "max_relative_position=$(m)")
    if Flux.istraining()
        print(io, ", dropout=$(t.drop.p))")
    else
        print(io, ")")
    end
end