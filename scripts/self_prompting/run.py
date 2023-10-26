from stage_one import main


def baseline():
    EXP_DIR = "experiments/self_prompting/baseline2"
    main(
        tasks=["truthful_qa"], models=["gpt-3.5-turbo"], example_cap=10, exp_dir=EXP_DIR
    )


def run():
    EXP_DIR = "experiments/self_prompting/question_debiasing"
    main(
        dataset="bbh",
        models=["gpt-4"],
        example_cap=2,
        exp_dir=EXP_DIR,
        interventions=["SelfPromptingFormatter"],
    )


if __name__ == "__main__":
    baseline()
    # run()
