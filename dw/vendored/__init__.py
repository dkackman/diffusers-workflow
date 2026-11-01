"""Upstream model code carried into dw as one module per port, each pinned to
the upstream commit or Hub revision its docstring names, so a port can be
diffed against its source and re-pinned as a unit. A port reaches dw only for
its validated media loaders (dw/tasks/audio_utils.py, dw/dsp.py),
and only a task module imports from here: dw/tasks/speech_generation.py
(audio8_tts_model) and dw/tasks/sound_effects.py (the mmaudio_* modules).

Like dw/community_pipelines/, this package is outside the architecture
ratchet (scripts/arch_metrics.py EXCLUDED): a port keeps upstream's shape and
size, which the module and function ceilings are not written for.
"""
