from scripts.training_formatters import TRAINING_COT_FORMATTERS_FEW_SHOT
from stage_one import main

if __name__ == "__main__":
    # test on TRAINING_COT_FORMATTERS_FEW_SHOT
    test_formatters = [f.name() for f in TRAINING_COT_FORMATTERS_FEW_SHOT]
    main(
        exp_dir="experiments/finetune_3",
        models=[
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89GzBGx0",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89G8xNY5",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89FBrz5b",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89GKVlFT",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89FCpgle",
        ],
        formatters=test_formatters,
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
        batch=20,
    )
