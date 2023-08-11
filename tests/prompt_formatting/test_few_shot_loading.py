from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import BiasedConsistency10
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.formatters.interventions.formatting import format_pair_cot


def test_few_shot_loading():
    stuff = get_correct_cots()
    assert len(stuff) != 0


def test_few_shot_pair_formatting():
    stuff = get_correct_cots()
    thing = format_pair_cot(stuff[0])
    thing.convert_to_completion_str()

    read = BiasedConsistency10.intervene(
        question=stuff[0].task_spec.read_data_example_or_raise(MilesBBHRawData), formatter=ZeroShotCOTUnbiasedFormatter
    )
    print(read)
