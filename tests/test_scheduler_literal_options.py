"""#1491: a scheduler's from_config_args string options are not type references."""

from dw.arguments import realize_args
from dw.type_references import component_type_errors


def test_string_option_under_type_key_stays_a_string():
    args = {
        "scheduler": {
            "configuration": {"scheduler_type": "DPMSolverMultistepScheduler"},
            "from_config_args": {
                "use_karras_sigmas": True,
                "algorithm_type": "dpmsolver++",
                "solver_type": "{midpoint}",
            },
        }
    }
    realize_args(args)
    opts = args["scheduler"]["from_config_args"]
    assert opts["algorithm_type"] == "dpmsolver++"
    assert opts["solver_type"] == "midpoint"
    assert opts["use_karras_sigmas"] is True
    assert isinstance(args["scheduler"]["configuration"]["scheduler_type"], type)


def test_real_scheduler_accepts_the_value():
    from diffusers import DPMSolverMultistepScheduler

    args = {"from_config_args": {"algorithm_type": "dpmsolver++"}}
    realize_args(args)
    sched = DPMSolverMultistepScheduler.from_config({}, **args["from_config_args"])
    assert sched.config.algorithm_type == "dpmsolver++"


def test_validation_does_not_treat_option_as_type():
    wf = {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "configuration": {"pipeline_type": "StableDiffusionPipeline"},
                    "scheduler": {
                        "configuration": {
                            "scheduler_type": "DPMSolverMultistepScheduler"
                        },
                        "from_config_args": {
                            "algorithm_type": "dpmsolver++",
                            "solver_type": "midpoint",
                        },
                    },
                },
            }
        ]
    }
    errors = component_type_errors(wf)
    assert not [
        e for e in errors if "_type" in e["message"] and "algorithm" in e["message"]
    ]
