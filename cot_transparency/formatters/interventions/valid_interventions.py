from typing import Type, Sequence


if True:
    # hack to register all the interventions from consistency.py, and make sure lint does not remove this import
    from cot_transparency.formatters.interventions.consistency import Intervention  # noqa: F811
    from cot_transparency.formatters.interventions.prompt_sensitivity import Intervention  # noqa: F811
    from cot_transparency.formatters.interventions.coup_intervention import CoupInstruction  # noqa: F811

VALID_INTERVENTIONS: dict[str, Type[Intervention]] = Intervention.all_interventions()


def name_to_intervention(name: str) -> Type[Intervention]:
    if name not in VALID_INTERVENTIONS:
        raise ValueError(f"intervention {name} is not valid. Valid intervention are {list(VALID_INTERVENTIONS.keys())}")
    return VALID_INTERVENTIONS[name]


def get_valid_stage1_intervention(intervention: str) -> Type[Intervention]:
    return name_to_intervention(intervention)


def get_valid_stage1_interventions(interventions: Sequence[str | None]) -> Sequence[Type[Intervention] | None]:
    # assert that the formatters are valid
    for intervention in interventions:
        if intervention is None:
            # Allow None to be passed in to indicate no intervention
            continue
        if intervention not in VALID_INTERVENTIONS:
            raise ValueError(
                f"intervention {intervention} is not valid. Valid intervention are {list(VALID_INTERVENTIONS.keys())}"
            )

    validated: list[Type[Intervention] | None] = [
        VALID_INTERVENTIONS[i] if i is not None else None for i in interventions
    ]
    return validated
