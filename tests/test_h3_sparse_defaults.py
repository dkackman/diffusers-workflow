"""The MiniMax-H3 templates default to sparse attention (#931).

Sparse is the default because it measured 2.2-4.4x faster per forward on a
3090; off CUDA it falls back to dense with a warning, so the default is safe
everywhere. Its routing defaults to top-k 0.15 (#932): faster than tau 0.5 at
equal or better faces and skin on the ramp's L7 and L5, judged by Don on three
seeds. Each template that pins a backend must also pass `sparse_tau` and
`sparse_topk` beside it, or the variables would be declared and never reach
the transformer.
"""

import json
from glob import glob

import pytest


def _read(path):
    with open(path) as f:
        return f.read()


# selected by the H3 transformer, not by any attention_backend: a Music 3
# template that pins a backend is no H3 default
H3 = sorted(
    p for p in glob("workflows/templates/minimax/*.json") if "MiniMaxH3" in _read(p)
)


def test_nineteen_h3_templates():
    assert len(H3) == 19


@pytest.mark.parametrize("path", H3)
def test_h3_defaults_to_sparse(path):
    d = json.loads(_read(path))
    assert d["variables"]["attention_backend"] == "sparse"
    assert d["variables"]["sparse_tau"] == 0.5
    assert d["variables"]["sparse_topk"] == 0.15
    text = json.dumps(d["steps"])
    pins = text.count('"attention_backend": "variable:attention_backend"')
    assert text.count('"sparse_tau": "variable:sparse_tau"') == pins
    assert text.count('"sparse_topk": "variable:sparse_topk"') == pins
    assert text.count('"attention_backend": "variable:attention_backend"') >= 1
    assert "attention_backend" in d["cost_drivers"]
