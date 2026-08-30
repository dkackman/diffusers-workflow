"""Prompt enhancement presets and the inline workflows that run them.

The enhancer is not a new execution path - each preset builds a one-step
inline workflow that the ordinary job queue runs, so progress streaming,
cancellation, history and the worker's model cache all apply unchanged.
The step saves its text result, which is how the enhanced prompt comes
back: as the single file in the job's manifest.
"""

import uuid

# A generic system prompt for expanding an idea into an image-generation
# prompt - the counterpart of the H3 preset's Context-IR spec, which lives
# in the builtin workflow rather than here
_T2I_SYSTEM_PROMPT = (
    "You expand a user's idea into a single detailed text-to-image prompt. "
    "Describe the subject, setting, lighting, mood, composition and style "
    "concretely, in one flowing paragraph of comma-separated phrases. "
    "Reply with the prompt text only - no preamble, no quotes, no headings."
)

# Each preset names what the UI shows (label, placeholder, curated models)
# and what the builder needs (which workflow or task runs the enhancement).
# 'intended_models' are the stored-prompt intended_model values the preset
# is preselected for
PRESETS = {
    "h3": {
        "label": "MiniMax-H3 Context-IR",
        "workflow": "builtin:h3_context_ir.json",
        "default_model": "Qwen/Qwen3-4B-Instruct-2507",
        "models": [
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ],
        "intended_models": ["minimax-h3"],
        "placeholder": (
            "Task: T2VA. Duration: 5.17 seconds. "
            "Idea: a red fox trotting through a snowy pine forest at dawn."
        ),
    },
    "t2i": {
        "label": "Text-to-image augmenter",
        "task": "text_generation",
        "system_prompt": _T2I_SYSTEM_PROMPT,
        "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "models": [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen3-4B-Instruct-2507",
        ],
        "intended_models": ["z-image", "flux"],
        "placeholder": "a cat portrait in a sunlit window",
    },
}


def preset_descriptions():
    """The presets as the UI consumes them - everything but how they run."""
    return [
        {
            "key": key,
            "label": preset["label"],
            "default_model": preset["default_model"],
            "models": preset["models"],
            "intended_models": preset["intended_models"],
            "placeholder": preset["placeholder"],
        }
        for key, preset in PRESETS.items()
    ]


def build_enhance_workflow(preset_key, idea, model_name=None, device=None):
    """An inline workflow that enhances one idea with one preset.

    The workflow id is unique per call: output file names derive from the id,
    so two enhancements never overwrite each other's text in the output
    directory.

    Args:
        preset_key: Key into PRESETS
        idea: The idea text to expand
        model_name: LLM repo id; the preset's default when omitted
        device: Device override for the language model; the workflow or
            task default (cpu) when omitted

    Returns:
        The inline workflow definition as a dict

    Raises:
        ValueError: If the preset is unknown or the idea is empty
    """
    preset = PRESETS.get(preset_key)
    if preset is None:
        raise ValueError(
            f"Unknown enhancer preset '{preset_key}' - one of {sorted(PRESETS)}"
        )
    if not idea or not idea.strip():
        raise ValueError("Provide an idea to enhance")

    model = model_name or preset["default_model"]

    if "workflow" in preset:
        arguments = {"prompt": idea, "model_name": model}
        if device:
            arguments["device"] = device
        action = {"workflow": {"path": preset["workflow"], "arguments": arguments}}
    else:
        arguments = {
            "prompt": idea,
            "model_name": model,
            "system_prompt": preset["system_prompt"],
            "max_new_tokens": 400,
        }
        if device:
            arguments["device"] = device
        action = {"task": {"command": preset["task"], "arguments": arguments}}

    return {
        "id": f"enhance_{uuid.uuid4().hex[:8]}",
        "steps": [
            {
                "name": "enhance",
                **action,
                # The save is the return channel: the builtin's own steps stay
                # save:false, so the manifest holds exactly this text file
                "result": {"content_type": "text/plain", "save": True},
            }
        ],
    }
