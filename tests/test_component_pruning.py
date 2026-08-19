"""
Unit tests for component pruning - truncate_layers and remove_modules.

Covers the mechanics (in-place ModuleList truncation, Identity replacement,
dotted-path resolution and its errors) and the numeric contract the feature
exists for: hidden_states[k] of a truncated stack is bit-identical to the
full model's, because index k of the returned tuple is recorded before
layer k runs.
"""

import pytest
import torch
from torch import nn

from dw.pipeline_processors.pipeline import (
    replace_modules_with_identity,
    truncate_module_lists,
)


class Inner(nn.Module):
    def __init__(self, num_layers=6, width=8):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(num_layers))


class Outer(nn.Module):
    def __init__(self, num_layers=6, width=8):
        super().__init__()
        self.inner = Inner(num_layers, width)
        self.head = nn.Linear(width, width)


class TestTruncateModuleLists:
    def test_truncates_in_place(self):
        component = Outer(num_layers=6)
        kept = [component.inner.layers[i] for i in range(3)]

        truncate_module_lists(
            component, "outer", {"truncate_layers": {"inner.layers": 3}}
        )

        assert len(component.inner.layers) == 3
        # The surviving layers are the same objects, not copies
        assert all(component.inner.layers[i] is kept[i] for i in range(3))

    def test_keep_at_least_length_is_a_noop(self):
        component = Outer(num_layers=4)
        truncate_module_lists(
            component, "outer", {"truncate_layers": {"inner.layers": 4}}
        )
        assert len(component.inner.layers) == 4

        truncate_module_lists(
            component, "outer", {"truncate_layers": {"inner.layers": 9}}
        )
        assert len(component.inner.layers) == 4

    def test_keep_below_one_raises(self):
        component = Outer()
        with pytest.raises(ValueError, match="at least one layer"):
            truncate_module_lists(
                component, "outer", {"truncate_layers": {"inner.layers": 0}}
            )

    def test_bad_path_raises(self):
        component = Outer()
        with pytest.raises(ValueError, match="no attribute 'missing'"):
            truncate_module_lists(
                component, "outer", {"truncate_layers": {"inner.missing": 2}}
            )

    def test_non_module_list_raises(self):
        component = Outer()
        with pytest.raises(ValueError, match="not a ModuleList"):
            truncate_module_lists(component, "outer", {"truncate_layers": {"head": 2}})

    def test_absent_configuration_is_a_noop(self):
        component = Outer(num_layers=5)
        truncate_module_lists(component, "outer", {})
        assert len(component.inner.layers) == 5


class TestReplaceModulesWithIdentity:
    def test_replaces_with_identity(self):
        component = Outer()
        replace_modules_with_identity(component, "outer", {"remove_modules": ["head"]})
        assert isinstance(component.head, nn.Identity)
        # The parameters are actually gone from the module tree
        assert all("head" not in name for name, _ in component.named_parameters())

    def test_replaces_nested_module(self):
        component = Outer()
        replace_modules_with_identity(
            component, "outer", {"remove_modules": ["inner.layers"]}
        )
        assert isinstance(component.inner.layers, nn.Identity)

    def test_missing_module_raises(self):
        component = Outer()
        with pytest.raises(ValueError, match="no attribute 'missing'"):
            replace_modules_with_identity(
                component, "outer", {"remove_modules": ["missing"]}
            )

    def test_absent_configuration_is_a_noop(self):
        component = Outer()
        replace_modules_with_identity(component, "outer", {})
        assert isinstance(component.head, nn.Linear)


class TestHiddenStateEquivalence:
    """hidden_states[k] survives truncation to k+1 layers bit-for-bit."""

    def test_truncated_hidden_state_is_bit_identical(self):
        # A tiny causal LM built from config - no downloads. The final norm is
        # what makes keeping exactly k layers wrong, so the model must have one,
        # which every Llama-family model does.
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=64,
        )
        model = LlamaForCausalLM(config).eval()
        input_ids = torch.randint(0, 64, (1, 10))

        k = 3
        with torch.no_grad():
            full = model.model(input_ids=input_ids, output_hidden_states=True)
        reference = full.hidden_states[k]

        truncate_module_lists(
            model, "text_encoder", {"truncate_layers": {"model.layers": k + 1}}
        )
        with torch.no_grad():
            truncated = model.model(input_ids=input_ids, output_hidden_states=True)

        assert len(truncated.hidden_states) == k + 2
        assert torch.equal(truncated.hidden_states[k], reference)

    def test_keeping_exactly_k_layers_is_not_identical(self):
        # The guard case: with only k layers, index k of the tuple is the
        # final-norm output - a different tensor. This is why the H3 config
        # keeps 51 layers for hidden_states[50], not 50.
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=64,
        )
        model = LlamaForCausalLM(config).eval()
        input_ids = torch.randint(0, 64, (1, 10))

        k = 3
        with torch.no_grad():
            full = model.model(input_ids=input_ids, output_hidden_states=True)
        reference = full.hidden_states[k]

        truncate_module_lists(
            model, "text_encoder", {"truncate_layers": {"model.layers": k}}
        )
        with torch.no_grad():
            truncated = model.model(input_ids=input_ids, output_hidden_states=True)

        assert not torch.equal(truncated.hidden_states[k], reference)
