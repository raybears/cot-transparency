from typing import Type

from cot_transparency.formatters.interventions.consistency import PairedConsistency6, PairedConsistency10, \
    BiasedConsistency10, NaiveFewShot10
from cot_transparency.formatters.interventions.intervention import Intervention

VALID_INTERVENTIONS: dict[str, Type[Intervention]] = {
    PairedConsistency6.name(): PairedConsistency6,
    PairedConsistency10.name(): PairedConsistency10,
    BiasedConsistency10.name(): BiasedConsistency10,
    NaiveFewShot10.name(): NaiveFewShot10,
}


def get_valid_stage1_interventions(interventions: list[str]) -> list[Type[Intervention]]:
    # assert that the formatters are valid
    for intervention in interventions:
        if intervention not in VALID_INTERVENTIONS:
            raise ValueError(
                f"intervention {intervention} is not valid. Valid intervention are {list(VALID_INTERVENTIONS.keys())}"
            )

    validated: list[Type[Intervention]] = [VALID_INTERVENTIONS[i] for i in interventions]
    return validated
