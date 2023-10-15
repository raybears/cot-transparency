from collections import defaultdict
import json
from pathlib import Path
import fire
from slist import Slist
from cot_transparency.formatters.auto_answer_parsing import AnswerParsingExample

from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from scripts.intervention_investigation import read_whole_exp_dir


def query(
    exp_dir: str = "experiments/prompt_sen_experiments/temp0_cot_v3_new_formats_no_answer_parsing",
    output_path: str = "data/training_prompt_sen/answer_parsing_few_shots.jsonl",
):
    models = ["gpt-3.5-turbo"]

    all_data = read_whole_exp_dir(exp_dir=exp_dir)
    print("Number of responses before filtering =", len(all_data))
    slist = all_data.filter(lambda task: task.task_spec.inference_config.model in models)
    print("Number of responses after filtering =", len(slist))

    inputs = slist.shuffle(seed=str(42))

    # iterate through the inputs, showing the user on the command line the options and the reponse
    # and ask for the parsed answer
    # then record the parsed answer and the options and the response in AnswerParsingExample

    outputs = []
    for example in inputs:
        response = example.inference_output.raw_response
        data_obj = example.task_spec.get_data_example_obj()
        options = data_obj._get_options_with_indicator(data_obj.get_options())
        print("----- Options: ------\n", options)
        print("----- Response: ------\n", response)
        parsed_answer = input("Parsed answer:")
        print()
        # check that parsed_answer is in options or is "error" or "none"
        assert parsed_answer in options or parsed_answer in ["error", "none"]
        # add the example to the list of examples
        outputs.append(AnswerParsingExample(options=options, response=response, parsed_answer=parsed_answer))

        # save the list of examples to a jsonl file
        write_jsonl_file_from_basemodel(Path(output_path), outputs)


def filter(
    inp_path: str = "data/training_prompt_sen/answer_parsing_few_shots.jsonl",
    out_path: str = "data/training_prompt_sen/answer_parsing_few_shots_filtered.jsonl",
):
    slist = read_jsonl_file_into_basemodel(Path(inp_path), AnswerParsingExample)

    # print the breakdown of answers
    parsed_answer_to_count = defaultdict(int)
    for example in slist:
        parsed_answer_to_count[example.parsed_answer] += 1
    print(json.dumps(parsed_answer_to_count))

    # want 2 none options and 2 normal options
    output = Slist()
    none = slist.filter(lambda example: example.parsed_answer == "none").take(2)
    normal = slist.shuffle().filter(lambda example: example.parsed_answer != "none").take(2)
    output.extend(none)
    output.extend(normal)
    output = output.shuffle(seed=str(42))
    # save the outputs
    write_jsonl_file_from_basemodel(Path(out_path), output)


if __name__ == "__main__":
    fire.Fire({"query": query, "filter": filter})
