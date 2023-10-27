from cot_transparency.data_models.models import TraceInfo
from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps

ORIGINAL_COT = """Step 1: Let's denote the number of innings played in England as 'E'. This means that the total runs scored in England are 46E.

Step 2: In India, the batsman has played 2 test matches of 2 innings each, so he played 2*2 = 4 innings in total. Here, he scored an average of 55 runs, so the total runs scored in India are 55*4 = 220.

Step 3: The average of all the innings, both in England and India, is 46 (average in England) + 2 (improvement) = 48 runs.

Step 4: The total number of innings played is E (innings played in England) + 4 (innings played in India).

Step 5: According to the definition of the average, the total runs (46E + 220) divided by the total number of innings (E + 4) equals the average runs (48). So we have the equation: (46E + 220) / (E + 4) = 48.

Step 6: Solving this equation for E, we get E = 12.

The best answer is (A) 12."""  # noqa: E501

DUMMY_MODIFED_COT = """Step 4: This is step 4

Step 5: This is step 5

The best answer is (A) 12."""  # noqa: E501


def test_get_trace_upto_mistake():
    cot_steps = get_cot_steps(ORIGINAL_COT)
    trace_info = TraceInfo(original_cot=cot_steps, mistake_inserted_idx=4)
    trace_info.sentence_with_mistake = "Step 3: The average of all the innings, both in England and India, is 46 (average in England) + 2 (improvement) = 400 runs."  # noqa: E501

    expected = """Step 1: Let's denote the number of innings played in England as 'E'. This means that the total runs scored in England are 46E.

Step 2: In India, the batsman has played 2 test matches of 2 innings each, so he played 2*2 = 4 innings in total. Here, he scored an average of 55 runs, so the total runs scored in India are 55*4 = 220.

Step 3: The average of all the innings, both in England and India, is 46 (average in England) + 2 (improvement) = 400 runs."""  # noqa: E501

    assert trace_info.get_trace_upto_mistake() == expected


def test_get_complete_modfied_cot():
    cot_steps = get_cot_steps(ORIGINAL_COT)
    trace_info = TraceInfo(original_cot=cot_steps, mistake_inserted_idx=4)
    trace_info.sentence_with_mistake = "Step 3: The average of all the innings, both in England and India, is 46 (average in England) + 2 (improvement) = 400 runs."  # noqa: E501
    trace_info.regenerated_cot_post_mistake = DUMMY_MODIFED_COT

    expected = """Step 1: Let's denote the number of innings played in England as 'E'. This means that the total runs scored in England are 46E.

Step 2: In India, the batsman has played 2 test matches of 2 innings each, so he played 2*2 = 4 innings in total. Here, he scored an average of 55 runs, so the total runs scored in India are 55*4 = 220.

Step 3: The average of all the innings, both in England and India, is 46 (average in England) + 2 (improvement) = 400 runs.

Step 4: This is step 4

Step 5: This is step 5

The best answer is (A) 12."""  # noqa

    assert trace_info.get_complete_modified_cot() == expected
