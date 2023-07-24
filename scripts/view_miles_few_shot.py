import json

if __name__ == '__main__':
    few_shot_path = "data/bbh/snarks/few_shot_prompts.json"
    with open(few_shot_path, "r") as f:
        few_shot_prompts = json.load(f)
        all_a_few_shot_prompt = few_shot_prompts["all_a_few_shot_prompt"]
        print(few_shot_prompts)