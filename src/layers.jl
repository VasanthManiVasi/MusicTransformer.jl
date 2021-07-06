using Transformers.Basic: PositionEmbedding

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