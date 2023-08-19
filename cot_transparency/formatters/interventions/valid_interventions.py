from typing import Type

from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.interventions.vanilla import *

VALID_INTERVENTIONS: dict[str, Type[Intervention]] = Intervention.all_interventions()


def get_valid_stage1_interventions(interventions: list[str]) -> list[Type[Intervention]]:
    # assert that the formatters are valid
    for intervention in interventions:
        if intervention not in VALID_INTERVENTIONS:
            raise ValueError(
                f"intervention {intervention} is not valid. Valid intervention are {list(VALID_INTERVENTIONS.keys())}"
            )

    validated: list[Type[Intervention]] = [VALID_INTERVENTIONS[i] for i in interventions]
    return validated
