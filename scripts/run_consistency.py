from cot_transparency.formatters.core.prompt_sensitivity_map import no_cot_sensitivity_formatters
from scripts.prompt_sensitivity_improved import prompt_metrics_2
from stage_one import main

if __name__ == "__main__":
    # Run the experiment for prompt sensitivity
    models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
    ]
    model_name_override = {
        "gpt-3.5-turbo": "gpt-3.5-turbo ",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7ryTmccr": "Finetuned 6000 COTs with unbiased questions",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7semB2r8": "Finetuned 6000 COTs with 3 different types of biased questions,<br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7uWGH06b": "Finetuned 6000 COTs with 5 different types of biased questions,<br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7uXhCnI7": "Finetuned 6000 COTs with 5 different types of biased questions,<br> leaving out bias of Wrong Fewshot, make sure these actually biased the model",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7t5OEDT9": "Finetuned 18000 COTs with biased questions,<br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7vVCogry": "Finetuned 72000 COTs with 5 different types of biased questions,<br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7tmQDS49": "Finetuned 72000 COTS with biased questions",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7t8IvMic": "Finetuned 18000 COTs with unbiased questions",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7rg7aRbV": "Finetuned 6000 COTs with biased questions,<br> including ALL biases",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7skb05DZ": "Finetuned 6000 COTs with biased questions,<br> leaving out bias of I think the answer is (X)",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7smTRQCv": "Finetuned 6000 COTs with biased questions,<br> leaving out bias of Stanford Professor opinion",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7soRFrpt": "Finetuned 6000 COTs with biased questions,<br> leaving out bias of More Reward for (X)",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::7wWkPEKY": "Finetuned 72000 COTs",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::80R5ewb3": "Finetuned 95%  COTs, biased questions,<br> 5%  non cots, unbiased questions <br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::80nD19wy": "Finetuned 64800 non COTs, biased questions,<br> 7200 cots, unbiased questions <br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF": "Finetuned 98% (70560) non COTs, biased questions,<br> 2% (1440) cots, unbiased questions <br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0": "All unbiased 98% COT, 2% non COT",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81Eu4Gp5": "Finetuned 98% (70560) COTs, biased questions,<br> 7200 2% (1440) non cots, unbiased questions <br> leaving out bias of Wrong Fewshot",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV": "50% COT, 50% no COT",
    }
    main(
        dataset="cot_testing",
        formatters=[f.name() for f in no_cot_sensitivity_formatters],
        example_cap=100,
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
        ],
        exp_dir="experiments/sensitivity",
    )
    prompt_metrics_2(
        exp_dir="experiments/sensitivity",
        models=models,
        name_override=model_name_override,
    )
