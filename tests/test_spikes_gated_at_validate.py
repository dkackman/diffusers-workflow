"""With DW_EXPERIMENTAL_REFINE_WINDOWS off, every long-form spike the flag
still gates is refused by the server's validate route, naming the flag:
#1376 moved only the windowed refine out of it. Each case edits the shipped
chained-segments template inline and posts it to /api/validate, the route
`validate_workflow` answers through."""

import copy
import json
import pathlib

import pytest

FLAG = "DW_EXPERIMENTAL_REFINE_WINDOWS"
TEMPLATE = json.loads(
    (
        pathlib.Path(__file__).parent.parent
        / "workflows/templates/minimax/chained-segments.json"
    ).read_text()
)


def _chain(**keys):
    def edit(step):
        step["pipeline"].setdefault("chain", {}).update(keys)
        return "chain"

    return edit


def _argument(name, value):
    def edit(step):
        step["pipeline"].setdefault("arguments", {})[name] = value
        return "arguments"

    return edit


# (edit of step 0, the key the refusal is placed at)
CASES = {
    "capstone": (_chain(experimental_capstone={"enter_below_sigma": 0.5}), None),
    "latent_carry": (
        _chain(continuity="guide", experimental_latent_carry=True),
        None,
    ),
    "keystone_latents": (
        _chain(continuity="guide", keystone="segment_1", keystone_slot="latent"),
        "keystone_slot",
    ),
    "anchor_every": (_chain(keystone="segment_1", anchor_every=2), None),
    "sigma_gate": (
        _chain(
            continuity="guide",
            keystone="segment_1",
            keystone_slot={"frame": 34, "enter_below_sigma": 0.5},
        ),
        "keystone_slot",
    ),
    "selflift": (_argument("experimental_selflift", {}), None),
    "tst": (_argument("experimental_tst", {}), None),
    "keystone_attention": (_argument("experimental_keystone_attention", {}), None),
}

KEYS = {
    "capstone": "experimental_capstone",
    "latent_carry": "experimental_latent_carry",
    "anchor_every": "anchor_every",
    "selflift": "experimental_selflift",
    "tst": "experimental_tst",
    "keystone_attention": "experimental_keystone_attention",
}


def _validate(client, name):
    edit, key = CASES[name]
    definition = copy.deepcopy(TEMPLATE)
    where = edit(definition["steps"][0])
    key = key or KEYS[name]
    path = f"steps[0].pipeline.{where}.{key}"
    response = client.post("/api/validate?sizes=false", json={"workflow": definition})
    assert response.status_code == 200, response.text
    body = response.json()
    return body, [e["message"] for e in body.get("errors", []) if e["path"] == path]


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_spike_is_refused_at_validate_with_the_flag_off(
    gallery_client, monkeypatch, name
):
    monkeypatch.delenv(FLAG, raising=False)
    body, messages = _validate(gallery_client, name)
    assert body["valid"] is False
    assert any(FLAG in message for message in messages), body


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_flag_on_lifts_the_flag_refusal(gallery_client, monkeypatch, name):
    monkeypatch.setenv(FLAG, "1")
    _, messages = _validate(gallery_client, name)
    assert not any(FLAG in message for message in messages), messages
