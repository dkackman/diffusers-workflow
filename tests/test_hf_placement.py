"""One place decides how a transformers pipeline is placed on a device."""

from dw.tasks.model_cache import hf_pipeline_placement


def test_mps_loads_on_the_cpu_and_moves_afterwards():
    # A device_map has the loading threads cast shards straight onto the
    # device, which races inside torch's Metal shader cache
    assert hf_pipeline_placement("mps") == {"device": "mps"}


def test_other_backends_use_a_device_map():
    assert hf_pipeline_placement("cuda:1") == {"device_map": "cuda:1"}
    assert hf_pipeline_placement("cpu") == {"device_map": "cpu"}
