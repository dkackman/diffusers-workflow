"""The attention backend applied to a component is visible to the consumer."""

from types import SimpleNamespace

from dw.pipeline_processors import components


def test_component_backend_emits_log(monkeypatch):
    logged = []
    monkeypatch.setattr(
        components, "emit_log", lambda msg, **kw: logged.append((msg, kw))
    )
    applied = []
    monkeypatch.setattr(
        components,
        "set_component_attention_backend",
        lambda component, backend: applied.append(backend),
    )
    transformer = SimpleNamespace()
    pipeline = SimpleNamespace(transformer=transformer)
    configuration = {"components": {"transformer": {"attention_backend": "sage_hub"}}}

    components.configure_components(pipeline, configuration, "cpu")

    assert applied == ["sage_hub"]
    assert len(logged) == 1
    message, data = logged[0]
    assert "transformer" in message and "sage_hub" in message
    assert data == {"component": "transformer", "attention_backend": "sage_hub"}
