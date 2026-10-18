import pathlib
import tomllib


def test_comfy_kitchen_is_a_core_dependency():
    data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    # Pre-1.0 kernel API: the upper bound keeps a minor bump from landing unseen
    assert "comfy-kitchen>=0.2.38,<0.3" in [d.replace(" ", "") for d in deps]
