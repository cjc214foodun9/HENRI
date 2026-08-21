from __future__ import annotations

import ast

from wave_ast_decoder import WaveASTDecoder


class _Codec:
    d_model = 256
    k_bins = 256


def test_binary_collection_tracer_generates_pairwise_sum_program():
    decoder = WaveASTDecoder(_Codec(), device="cpu")
    bodies = decoder._instantiate("pairwise_sum", ["left", "right"])
    sources = {
        "def pairwise_sum(left, right):\n" + body
        for body in bodies
    }

    expected = (
        "def pairwise_sum(left, right):\n"
        "    return [x + y for x, y in zip(left, right)]"
    )
    assert expected in sources
    ast.parse(expected)


def test_binary_collection_tracer_generates_mapping_and_dictionary_forms():
    decoder = WaveASTDecoder(_Codec(), device="cpu")
    bodies = decoder._instantiate("pairwise", ["keys", "values"])
    joined = "\n".join(bodies)

    assert "dict(zip(keys, values))" in joined
    assert "[x - y for x, y in zip(keys, values)]" in joined
