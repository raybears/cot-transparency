from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots


def test_few_shot_loading():
    stuff = get_correct_cots()
    assert len(stuff) != 0
