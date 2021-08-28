# MusicTransformer

A Julia-based implementation of [**Music Transformer: Generating Music with Long-Term Structure**](https://arxiv.org/abs/1809.04281).

The Music Transformer is an attention-based neural network that can generate music with improved long-term coherence. Using the event-based [Performance](https://arxiv.org/abs/1808.03715) representation, the Music Transformer generates expressive performances directly without first generating a score. A Julia-based implementation of the Performance representation can be found at [NoteSequences.jl](https://github.com/VasanthManiVasi/NoteSequences.jl)

## Installation

MusicTransformer.jl uses [NoteSequences.jl](https://github.com/VasanthManiVasi/NoteSequences.jl) internally. Please install both of them to use the package.

```julia
] add https://github.com/VasanthManiVasi/NoteSequences.jl
] add https://github.com/VasanthManiVasi/MusicTransformer.jl
```

## Example

```julia
using FileIO
using MusicTransformer
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

music_transformer = pretrained"unconditional_model_16"

midi = generate(music_transformer, numsteps=1500)

save("generated.mid", midi)
```

See `examples/Melody Conditioning` for an example of using the Melody-Conditioned Music Transformer to generate an accompaniment given a melody.

## Training

To train a new music transformer or to fine-tune one of the available models on your own collection of midi files, the midi files must first be converted to one-hot indices, which can then be fed to the model as inputs. Please refer to `examples/MAESTRO Language Modelling/maestro_datagen.jl` for an example script on converting midifiles to model inputs. The example provided performs data generation for training a piano performance language model on the MAESTRO dataset.
Finally, see `examples/MAESTRO Language Modelling/maestro_train.jl` for a training script to train a language model on the processed MAESTRO dataset we obtained from data generation. The examples can be modified for training on your own data.

## References

1. [**Music Transformer: Generating Music with Long-Term Structure** - _arxiv.org_](https://arxiv.org/abs/1809.04281)
2. [**Music Transformer** - _Google_](https://magenta.tensorflow.org/music-transformer)
3. [**Tensor2Tensor** - _Google_](https://github.com/tensorflow/tensor2tensor)
