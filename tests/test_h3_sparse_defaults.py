"""The MiniMax-H3 templates default to sparse attention (#931).

Sparse is the default because it measured 2.2-4.4x faster per forward on a
3090; off CUDA it falls back to dense with a warning, so the default is safe
everywhere. Each template that pins a backend must also pass `sparse_tau`
beside it, or the variable would be declared and never reach the transformer.
"""

import json
from glob import glob

import pytest

H3 = sorted(
    p
    for p in glob("workflows/templates/minimax/*.json")
    if '"attention_backend"' in open(p).read()
)


def test_nineteen_h3_templates():
    assert len(H3) == 19


@pytest.mark.parametrize("path", H3)
def test_h3_defaults_to_sparse(path):
    d = json.load(open(path))
    assert d["variables"]["attention_backend"] == "sparse"
    assert d["variables"]["sparse_tau"] == 0.5
    text = json.dumps(d["steps"])
    assert text.count('"sparse_tau": "variable:sparse_tau"') == text.count(
        '"attention_backend": "variable:attention_backend"'
    )
    assert text.count('"attention_backend": "variable:attention_backend"') >= 1
    assert "attention_backend" in d["cost_drivers"]
