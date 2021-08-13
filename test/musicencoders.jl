using NoteSequences: SeqNote

@testset "MidiPerformanceEncoder" begin
    @testset "Encoding/Decoding" begin
        encoder = MidiPerformanceEncoder(100, 32, 21, 108, true)

        ns = NoteSequence()
        append!(ns.notes, [
            SeqNote(21, 127, 0, 220, 1, 0),
            SeqNote(50, 1, 110, 220, 1, 0),
            SeqNote(108, 100, 220, 440, 1, 0),
            SeqNote(80, 100, 330, 440, 1, 0),
        ])

        expected = [
            310,
            3,
            203,
            279,
            32,
            203,
            91,
            120,
            303,
            90,
            203,
            62,
            203,
            178,
            150,
            2
        ]

        @test expected == encode_notesequence(ns, encoder)
    end

    @testset "Misc" begin
        encoder = MidiPerformanceEncoder(100, 32, 21, 108, true)

        @test encode_notesequence(NoteSequence(), encoder) == [2]
        @test encoder.vocab_size == 310
        @test encoder.num_reserved_tokens == 2
    end
end

@testset "TextMelodyEncoder" begin
    @testset "Encoding/Decoding" begin
        encoder = TextMelodyEncoder(21, 108, 10)

        ns = NoteSequence()
        append!(ns.notes, [
            SeqNote(21, 127, 0, 220, 1, 0),
            SeqNote(107, 100, 220, 440, 1, 0),
            SeqNote(80, 100, 440, 660, 1, 0),
        ])

        expected = [
            5,
            3,
            3,
            3,
            4,
            91,
            3,
            3,
            3,
            4,
            64,
            3,
            3,
            3,
            3
        ]

        @test expected == encode_notesequence(ns, encoder)
    end

    @testset "Misc" begin
        encoder = TextMelodyEncoder(21, 109, 100)

        @test encode_notesequence(NoteSequence(), encoder) == Int[]
        @test encoder.vocab_size == 92
        @test encoder.num_reserved_tokens == 2
    end
end