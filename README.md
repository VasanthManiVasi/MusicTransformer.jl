# MusicTransformer

A Julia-based implementation of [Music Transformer: Generating Music with Long-Term Structure](https://arxiv.org/abs/1809.04281).

The Music Transformer is an attention-based neural network that can generate music with improved long-term coherence. Using the event-based [Performance](https://arxiv.org/abs/1808.03715) representation, the Music Transformer generates expressive performances directly without first generating a score. A Julia-based implementation of the Performance representation can be found at [NoteSequences.jl](https://github.com/VasanthManiVasi/NoteSequences.jl)

## Installation

```julia
] add https://github.com/VasanthManiVasi/MusicTransformer.jl
```

## References
1. [**Music Transformer: Generating Music with Long-Term Structure** - _arxiv.org_](https://arxiv.org/abs/1809.04281)
2. [**Music Transformer** - _Google_](https://magenta.tensorflow.org/music-transformer)
3. [**Tensor2Tensor** - _Google_](https://github.com/tensorflow/tensor2tensor)
