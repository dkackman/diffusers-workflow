import pathlib
import tomllib


def test_comfy_kitchen_is_a_core_dependency():
    data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("comfy-kitchen>=0.2.38") for d in deps)
