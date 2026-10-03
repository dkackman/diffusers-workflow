"""dw/lora_hub.py against a fake HfApi: exact-base search, safetensors-only,
format classification from the header, warnings rather than crashes."""

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dw.lora_hub import classify_format, hub_candidates, search_hub

OLD = datetime(2025, 1, 1, tzinfo=timezone.utc)
NEW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def sibling(name):
    return SimpleNamespace(rfilename=name)


def info(repo, files=("lora.safetensors",), downloads=10, card=None, gated=False, modified=NEW):
    return SimpleNamespace(
        id=repo, downloads=downloads, likes=1, last_modified=modified, gated=gated,
        sha=f"{repo}-sha", siblings=[sibling(f) for f in files], card_data=card,
    )


class FakeApi:
    def __init__(self, listings, headers=None, base_modified=OLD, fail_header=()):
        self.listings = listings  # {(filter, search): [info]}
        self.headers = headers or {}
        self.base_modified = base_modified
        self.fail_header = set(fail_header)
        self.searches = []

    def list_models(self, *, filter, search=None, sort=None, limit=None, expand=None):
        self.searches.append((filter, search))
        return list(self.listings.get((filter, search), []))

    def model_info(self, repo):
        return SimpleNamespace(last_modified=self.base_modified)

    def parse_safetensors_file_metadata(self, repo, filename, *, revision=None, timeout=None):
        if repo in self.fail_header:
            raise RuntimeError("range read refused")
        return SimpleNamespace(tensors=dict.fromkeys(self.headers.get(repo, ["x.lora_A.weight"])))


F = "base_model:adapter:Qwen/Qwen-Image-2.1"


class TestFormat:
    def test_diffusers_peft_keys(self):
        assert classify_format(["transformer.blocks.0.attn.to_q.lora_A.weight"]) == "diffusers"

    def test_kohya_keys(self):
        assert classify_format(["lora_unet_blocks_0_attn.lora_down.weight"]) == "kohya"

    def test_full_weight_keys_win(self):
        assert classify_format(["a.lora_A.weight", "b.diff", "c.diff_b"]) == "full_weight"

    def test_anything_else(self):
        assert classify_format(["model.weight"]) == "unknown"


class TestCandidates:
    def test_searches_the_exact_base_per_word_and_merges(self):
        api = FakeApi({(F, "voxel style"): [info("a/voxel")], (F, "voxel"): [info("a/voxel"), info("b/voxel2", downloads=99)]})
        results = hub_candidates(["Qwen/Qwen-Image-2.1"], "voxel style", ["voxel"], 8, {}, api)
        assert (F, "voxel style") in api.searches and (F, "voxel") in api.searches
        assert [r["model_name"] for r in results] == ["b/voxel2", "a/voxel"]

    def test_a_stop_word_query_searches_unfiltered(self):
        api = FakeApi({(F, None): [info("a/x")]})
        results = hub_candidates(["Qwen/Qwen-Image-2.1"], "style", [], 8, {}, api)
        assert api.searches == [(F, None)]
        assert results[0]["model_name"] == "a/x"

    def test_a_pickle_only_repo_is_dropped(self):
        api = FakeApi({(F, None): [info("a/bin", files=("lora.bin",))]})
        assert hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api) == []

    def test_a_candidate_carries_a_step_ready_lora(self):
        api = FakeApi({(F, None): [info("a/x")]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert result["source"] == "hub" and result["status"] == "candidate"
        assert result["as_lora"] == {"model_name": "a/x", "weight_name": "lora.safetensors", "revision": "a/x-sha", "scale": 1.0}
        assert result["format"] == "diffusers"

    def test_multiple_weights_leave_weight_name_unset(self):
        api = FakeApi({(F, None): [info("a/x", files=("one.safetensors", "two.safetensors"))]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "as_lora" not in result
        assert result["weights"] == ["one.safetensors", "two.safetensors"]
        assert "multiple_weights" in result["warnings"]

    def test_full_weight_keys_warn_will_not_load(self):
        api = FakeApi({(F, None): [info("a/x")]}, headers={"a/x": ["b.diff"]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "will_not_load" in result["warnings"]

    def test_no_card_is_warnings_not_a_crash(self):
        api = FakeApi({(F, None): [info("a/x", card=None, modified=None)]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "no_license" in result["warnings"]
        assert result.get("trigger") is None

    def test_a_list_instance_prompt_takes_its_first(self):
        card = {"license": "apache-2.0", "instance_prompt": ["Voxel Style", "other"]}
        api = FakeApi({(F, None): [info("a/x", card=card)]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert result["trigger"] == "Voxel Style"
        assert result["license"] == "apache-2.0"

    def test_older_than_its_base_is_stale(self):
        api = FakeApi({(F, None): [info("a/x", modified=OLD)]}, base_modified=NEW)
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "stale" in result["warnings"]

    def test_gated_is_flagged(self):
        api = FakeApi({(F, None): [info("a/x", gated="manual")]})
        assert "gated" in hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)[0]["warnings"]

    def test_an_unreadable_header_is_a_warning(self):
        api = FakeApi({(F, None): [info("a/x")]}, fail_header=["a/x"])
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api)
        assert "header_unreadable" in result["warnings"]

    def test_a_catalog_rejected_repo_comes_back_rejected(self):
        api = FakeApi({(F, None): [info("drozbay/FastH3")]})
        [result] = hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {"drozbay/FastH3": ".diff keys"}, api)
        assert result == {"source": "hub", "status": "rejected", "model_name": "drozbay/FastH3", "reason": ".diff keys"}

    def test_a_malformed_repo_id_from_the_hub_is_ignored(self):
        api = FakeApi({(F, None): [info("../evil")]})
        assert hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api) == []

    def test_limit_caps_the_candidates(self):
        api = FakeApi({(F, None): [info(f"a/x{i}", downloads=i) for i in range(5)]})
        assert len(hub_candidates(["Qwen/Qwen-Image-2.1"], "", [], 2, {}, api)) == 2


class TestFailure:
    def test_a_hub_error_is_returned_not_raised(self):
        class Down(FakeApi):
            def list_models(self, **kwargs):
                raise ConnectionError("hub unreachable")
        results, error = search_hub(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api=Down({}))
        assert results == [] and "hub unreachable" in error

    def test_a_slow_hub_times_out(self):
        class Slow(FakeApi):
            def list_models(self, **kwargs):
                time.sleep(2)
                return []
        results, error = search_hub(["Qwen/Qwen-Image-2.1"], "", [], 8, {}, api=Slow({}), timeout=0.2)
        assert results == [] and "timed out" in error
