"""Tests for text_generation task."""

import json as _json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from dw.tasks.text_generation import (
    generate_text,
    _DEFAULT_MODEL,
    DEFAULT_VISION_MODEL,
    _VISION_REPETITION_PENALTY,
    resolve_system_prompt,
)


class TestTextGeneration(unittest.TestCase):
    """Tests for the generate_text function."""

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_returns_generated_string(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "an expanded detailed prompt"}]
        mock_pipeline.return_value = pipe

        result = generate_text("a cat", device="cpu")

        self.assertEqual(result, "an expanded detailed prompt")
        mock_pipeline.assert_called_once()

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_uses_default_model(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text("test", device="cpu")

        call_kwargs = mock_pipeline.call_args
        self.assertEqual(call_kwargs[1]["model"], _DEFAULT_MODEL)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_custom_model_name(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "test", device="cpu", model_name="meta-llama/Llama-3.2-1B-Instruct"
        )

        call_kwargs = mock_pipeline.call_args
        self.assertEqual(call_kwargs[1]["model"], "meta-llama/Llama-3.2-1B-Instruct")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_system_prompt_included(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "a cat",
            device="cpu",
            system_prompt="You expand prompts for image generation.",
        )

        call_args = pipe.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[0]["content"], "You expand prompts for image generation."
        )
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "a cat")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_no_system_prompt(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text("a cat", device="cpu")

        call_args = pipe.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_max_new_tokens(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text("test", device="cpu", max_new_tokens=200)

        call_args = pipe.call_args
        self.assertEqual(call_args[1]["max_new_tokens"], 200)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_strips_whitespace(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "  some text with spaces  \n"}]
        mock_pipeline.return_value = pipe

        result = generate_text("test", device="cpu")
        self.assertEqual(result, "some text with spaces")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_chat_template_kwargs_reach_the_text_template(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "a cat",
            device="cpu",
            model_name="m/text-chat-kwargs",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertEqual(
            pipe.call_args[1]["tokenizer_encode_kwargs"], {"enable_thinking": False}
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_chat_template_kwargs_reach_the_vision_template(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "a cat",
            device="cpu",
            model_name="m/vision-chat-kwargs",
            image="https://example.com/a.png",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertIs(pipe.call_args[1]["enable_thinking"], False)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_no_chat_template_kwargs_passes_nothing_extra(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text("a cat", device="cpu", model_name="m/no-chat-kwargs")

        self.assertNotIn("tokenizer_encode_kwargs", pipe.call_args[1])

    def test_chat_template_kwargs_must_be_a_mapping(self):
        with self.assertRaises(ValueError):
            generate_text(
                "a cat", device="cpu", chat_template_kwargs=["enable_thinking"]
            )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_revision_reaches_the_loader(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text("a cat", device="cpu", model_name="m/revision", revision="a" * 40)

        self.assertEqual(mock_pipeline.call_args[1]["revision"], "a" * 40)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_a_list_of_images_becomes_one_part_each_in_order(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "edit",
            device="cpu",
            model_name="m/image-list",
            image=["https://example.com/1.png", "https://example.com/2.png"],
        )

        user = pipe.call_args[1]["text"][-1]["content"]
        self.assertEqual(
            [part.get("url") for part in user[:2]],
            ["https://example.com/1.png", "https://example.com/2.png"],
        )
        self.assertEqual(user[2], {"type": "text", "text": "edit"})
        self.assertEqual(mock_pipeline.call_args[0][0], "image-text-to-text")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_an_empty_or_all_null_image_list_is_the_text_path(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        for empty in ([], [None]):
            generate_text(
                "a cat",
                device="cpu",
                model_name=f"m/empty-{len(empty)}",
                image=empty,
            )
            self.assertEqual(mock_pipeline.call_args[0][0], "text-generation")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_a_non_scalar_chat_template_value_is_refused(self, mock_pipeline):
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            generate_text(
                "a cat",
                device="cpu",
                model_name="m/nonscalar",
                chat_template_kwargs={"enable_thinking": {"a": 1}},
            )
        mock_pipeline.assert_not_called()

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_a_reserved_chat_template_key_is_refused_before_any_call(
        self, mock_pipeline
    ):
        pipe = MagicMock()
        mock_pipeline.return_value = pipe
        for key in ("images", "tools", "tokenizer_encode_kwargs"):
            with self.assertRaisesRegex(ValueError, f"'{key}'.*not a chat-template"):
                generate_text(
                    "a cat",
                    device="cpu",
                    model_name="m/reserved-vision",
                    image="https://example.com/a.png",
                    chat_template_kwargs={key: "http://169.254.169.254/x"},
                )
            with self.assertRaisesRegex(ValueError, f"'{key}'.*not a chat-template"):
                generate_text(
                    "a cat",
                    device="cpu",
                    model_name="m/reserved-text",
                    chat_template_kwargs={key: "x"},
                )
        pipe.assert_not_called()

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_a_legitimate_setting_passes_on_both_paths(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe
        settings = {"enable_thinking": False, "effort": "low", "n": 2, "t": 0.5}

        generate_text(
            "a cat", device="cpu", model_name="m/ok-text", chat_template_kwargs=settings
        )
        self.assertEqual(pipe.call_args[1]["tokenizer_encode_kwargs"], settings)
        generate_text(
            "a cat",
            device="cpu",
            model_name="m/ok-vision",
            image="https://example.com/a.png",
            chat_template_kwargs=settings,
        )
        self.assertIs(pipe.call_args[1]["enable_thinking"], False)


class TestTextGenerationWithImage(unittest.TestCase):
    """An image switches generation onto a vision-language pipeline."""

    def _make_image(self):
        from PIL import Image

        return Image.new("RGB", (32, 32), color="blue")

    def _mock_pipe(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "a blue square on a plain field"}]
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_image_selects_the_vision_pipeline(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        self.assertEqual(mock_pipeline.call_args[0][0], "image-text-to-text")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_no_image_stays_on_text_generation(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("a cat", device="cpu")

        self.assertEqual(mock_pipeline.call_args[0][0], "text-generation")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_image_uses_the_vision_default_model(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        self.assertEqual(mock_pipeline.call_args[1]["model"], DEFAULT_VISION_MODEL)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_vision_content_is_typed_parts(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)
        image = self._make_image()

        generate_text(
            "describe it",
            device="cpu",
            image=image,
            system_prompt="You are terse.",
        )

        messages = pipe.call_args[1]["text"]
        self.assertEqual(
            messages[0]["content"], [{"type": "text", "text": "You are terse."}]
        )
        self.assertEqual(
            messages[1]["content"],
            [
                {"type": "image", "image": image},
                {"type": "text", "text": "describe it"},
            ],
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_string_image_is_passed_as_a_url(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image="https://example.com/cat.jpg")

        content = pipe.call_args[1]["text"][-1]["content"]
        self.assertEqual(
            content[0], {"type": "image", "url": "https://example.com/cat.jpg"}
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_sampling_is_disabled_through_generate_kwargs(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        # This pipeline forwards unrecognised kwargs to the processor, which
        # drops them - a bare do_sample=False would silently leave sampling on
        self.assertIs(pipe.call_args[1]["generate_kwargs"]["do_sample"], False)
        self.assertNotIn("do_sample", pipe.call_args[1])

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_vision_defaults_to_a_repetition_penalty(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        # Greedy decoding loops on long format specs; the penalty is what stops
        # that without giving up reproducible output
        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["repetition_penalty"],
            _VISION_REPETITION_PENALTY,
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_repetition_penalty_is_overridable(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "describe it",
            device="cpu",
            image=self._make_image(),
            repetition_penalty=1.2,
        )

        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["repetition_penalty"], 1.2
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_text_path_takes_no_repetition_penalty(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("a cat", device="cpu")

        # The text models do not loop, and changing their decoding would change
        # the output of every workflow already using them
        self.assertNotIn("generate_kwargs", pipe.call_args[1])

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_text_and_vision_variants_cache_separately(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        # Same model name under both pipelines - they are not interchangeable,
        # so the cache must not hand one back for the other
        generate_text("a cat", device="cpu", model_name="some/model")
        generate_text(
            "a cat", device="cpu", model_name="some/model", image=self._make_image()
        )

        self.assertEqual(mock_pipeline.call_count, 2)
        tasks = [call[0][0] for call in mock_pipeline.call_args_list]
        self.assertEqual(tasks, ["text-generation", "image-text-to-text"])


class TestGenerateKwargsPassThrough(unittest.TestCase):
    """Anything generate() understands can be set from a workflow."""

    def _make_image(self):
        from PIL import Image

        return Image.new("RGB", (32, 32), color="blue")

    def _mock_pipe(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_reaches_the_text_pipeline(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "a cat", device="cpu", generate_kwargs={"no_repeat_ngram_size": 25}
        )

        self.assertEqual(pipe.call_args[1]["no_repeat_ngram_size"], 25)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_reaches_the_vision_pipeline(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "describe it",
            device="cpu",
            image=self._make_image(),
            generate_kwargs={"no_repeat_ngram_size": 25},
        )

        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["no_repeat_ngram_size"], 25
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_overrides_what_the_task_decided(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "describe it",
            device="cpu",
            image=self._make_image(),
            generate_kwargs={"repetition_penalty": 1.3},
        )

        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["repetition_penalty"], 1.3
        )


class TestTextGenerationLoadDevice(unittest.TestCase):
    """How the model gets onto the device it runs on.

    transformers materializes weights across a thread pool, and on MPS each
    thread's cast-copy races inside torch's Metal shader cache - a crash
    during loading, not during generation. Handing the pipeline a `device`
    instead of a `device_map` keeps that load on the CPU and leaves the move
    to the calling thread.
    """

    @staticmethod
    def _mock_pipe(mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_mps_loads_on_cpu_then_moves(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("test", device="mps")

        kwargs = mock_pipeline.call_args[1]
        self.assertEqual(kwargs["device"], "mps")
        self.assertNotIn("device_map", kwargs)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_other_devices_still_use_device_map(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("test", device="cuda")

        kwargs = mock_pipeline.call_args[1]
        self.assertEqual(kwargs["device_map"], "cuda")
        self.assertNotIn("device", kwargs)


PIN = "3f3edb52b2c55de0d49feab28598b68f7f8e4830"


def _spec(**over):
    spec = {
        "repo": "black-forest-labs/flux-3-image-klein-pup",
        "file": "system_prompts.json",
        "revision": PIN,
        "key": "t2i",
    }
    spec.update(over)
    return {"from_hub": spec}


class TestSystemPromptFromHub(unittest.TestCase):
    def setUp(self):

        self.dir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.dir, "system_prompts.json")
        with open(self.json_path, "w") as f:
            _json.dump({"t2i": "expand the request", "i2i": "edit it", "n": 3}, f)

    def test_a_string_or_none_passes_through(self):
        self.assertEqual(resolve_system_prompt("plain"), "plain")
        self.assertIsNone(resolve_system_prompt(None))

    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_the_key_s_text_is_returned_from_the_pinned_file(self, download):
        download.return_value = self.json_path
        self.assertEqual(resolve_system_prompt(_spec()), "expand the request")
        download.assert_called_once_with(
            "black-forest-labs/flux-3-image-klein-pup",
            "system_prompts.json",
            revision=PIN,
        )

    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_a_missing_key_names_the_keys_the_file_has(self, download):
        download.return_value = self.json_path
        with self.assertRaises(ValueError) as caught:
            resolve_system_prompt(_spec(key="t2v"))
        self.assertIn("i2i", str(caught.exception))
        self.assertIn("t2i", str(caught.exception))

    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_a_value_that_is_not_text_is_refused(self, download):
        download.return_value = self.json_path
        with self.assertRaises(ValueError):
            resolve_system_prompt(_spec(key="n"))

    def test_a_short_hash_or_branch_is_refused(self):
        for revision in ("3f3edb52", "main", PIN.upper()):
            with self.assertRaises(ValueError) as caught:
                resolve_system_prompt(_spec(revision=revision))
            self.assertIn("40", str(caught.exception))

    def test_a_path_that_escapes_the_repo_is_refused(self):
        for file in (
            "../secrets.json",
            "/etc/passwd",
            "a\\b.json",
            "",
            "x/../../y.json",
        ):
            with self.assertRaises(ValueError):
                resolve_system_prompt(_spec(file=file))

    def test_a_bad_repo_id_is_refused(self):
        from dw.security import InvalidInputError

        with self.assertRaises((ValueError, InvalidInputError)):
            resolve_system_prompt(_spec(repo="https://evil.example/x"))

    def test_an_unknown_field_or_shape_is_refused(self):
        for bad in (
            {"from_hub": {**_spec()["from_hub"], "branch": "main"}},
            {"from_url": "x"},
            ["t2i"],
        ):
            with self.assertRaises(ValueError):
                resolve_system_prompt(bad)

    @patch("dw.tasks.text_generation.hf_pipeline")
    @patch("dw.tasks.text_generation.hf_hub_download")
    def test_generate_text_sends_the_resolved_text(self, download, mock_pipeline):
        download.return_value = self.json_path
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "out"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "a dog", device="cpu", model_name="m/from-hub", system_prompt=_spec()
        )

        messages = pipe.call_args[0][0]
        self.assertEqual(
            messages[0], {"role": "system", "content": "expand the request"}
        )


if __name__ == "__main__":
    unittest.main()
