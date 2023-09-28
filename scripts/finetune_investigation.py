from typing import Sequence, Type, Optional, Mapping

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.prompt_sensitivity_map import unbiased_sensitivity_prompt
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
    ZeroShotTellTruthCOTFormatter,
)
from cot_transparency.formatters.more_biases.anchor_initial_wrong import ZeroShotInitialWrongFormatter
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantTargetedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.formatters.verbalize.formatters import StanfordBiasedFormatter
from scripts.intervention_investigation import (
    read_whole_exp_dir,
    plot_dots_for_intervention,
    DottedLine,
    bar_plot,
    ConsistentOnly,
    TaskOutputFilter,
    InconsistentOnly,
)
from scripts.matching_user_answer import matching_user_answer_plot_dots
from scripts.multi_accuracy import PlotDots
from scripts.simple_model_names import MODEL_SIMPLE_NAMES


def make_finetune_graph(
    biased_formatters: Sequence[Type[StageOneFormatter]],
    finetuned_models: Sequence[str],
    unbiased_model: str,
    unbiased_formatter: Type[StageOneFormatter],
    all_read: Slist[TaskOutput],
    accuracy_plot_name: Optional[str] = None,
    percent_matching_plot_name: Optional[str] = None,
    filterer: Type[TaskOutputFilter] = ConsistentOnly,
    tasks: Sequence[str] = [],
    model_name_override: Mapping[str | None, str] = {},
    must_include_task_hashes: set[str] = set(),
):
    filtered_read = filterer.filter(all_read).filter(lambda task: task.task_spec.task_name in tasks if tasks else True)
    filtered_read_hashed = filtered_read.filter(
        lambda task: task.task_spec.task_hash in must_include_task_hashes if must_include_task_hashes else True
    )

    # unbiased acc
    unbiased_plot: PlotDots = plot_dots_for_intervention(
        None,
        filtered_read_hashed,
        for_formatters=[unbiased_formatter],
        name_override=f"Unbiased context, {unbiased_model}",
        model=unbiased_model,
    )

    plot_dots: Slist[PlotDots] = Slist(finetuned_models).map(
        lambda m: plot_dots_for_intervention(
            None,
            filtered_read_hashed,
            for_formatters=biased_formatters,
            model=m,
            name_override=model_name_override.get(m, m),
        )
    )
    dotted = DottedLine(name="Zero shot unbiased context performance", value=unbiased_plot.acc.accuracy, color="red")

    bar_plot(
        plot_dots=plot_dots,
        title=accuracy_plot_name or "Accuracy of different models in biased contexts",
        dotted_line=dotted,
        y_axis_title="Accuracy",
        add_n_to_name=True,
    )
    # accuracy_diff_intervention(interventions=interventions, unbiased_formatter=unbiased_formatter)
    matching_user_answer: list[PlotDots] = [
        matching_user_answer_plot_dots(
            intervention=None,
            all_tasks=filtered_read_hashed,
            for_formatters=biased_formatters,
            model=model,
            name_override=model_name_override.get(model, model),
        )
        for model in finetuned_models
    ]
    unbiased_matching_baseline = matching_user_answer_plot_dots(
        intervention=None, all_tasks=filtered_read_hashed, for_formatters=[unbiased_formatter], model=unbiased_model
    )
    dotted_line = DottedLine(
        name="gpt-3.5-turbo in Unbiased context,zeroshot", value=unbiased_matching_baseline.acc.accuracy, color="red"
    )
    dataset_str = Slist(tasks).mk_string(", ")
    bar_plot(
        plot_dots=matching_user_answer,
        title=percent_matching_plot_name or f"How often does each model choose the user's view Dataset: {dataset_str}",
        y_axis_title="Answers matching biased answer (%)",
        dotted_line=dotted_line,
        add_n_to_name=True,
    )


if __name__ == "__main__":
    filterer = InconsistentOnly
    tasks = ["truthful_qa", "logiqa", "hellaswag", "mmlu"]
    dataset_str = Slist(tasks).mk_string(", ")
    selected_bias = ZeroShotInitialWrongFormatter

    bias_name_map = {
        WrongFewShotIgnoreMistakesBiasedFormatter: "biased by Wrong Fewshot, COT response",
        WrongFewShotIgnoreMistakesBiasedNoCOTFormatter: "biased by Wrong Fewshot, no COT",
        StanfordBiasedFormatter: "biased by Stanford Professor Opinion",
        MoreRewardBiasedFormatter: "biased by More Reward for (X)",
        ZeroShotCOTSycophancyFormatter: "biased by I think the answer is (X)",
        ZeroShotCOTUnbiasedFormatter: "on unbiased questions",
        ZeroShotUnbiasedFormatter: "on unbiased questions, no COT",
        DeceptiveAssistantTargetedFormatter: "biased by Deceptive Assistant",
        ZeroShotInitialWrongFormatter: "biased by the Assistant's initial wrong answer",
        ZeroShotTellTruthCOTFormatter: "Tell the truth in a scientific manner",
        unbiased_sensitivity_prompt: f"on unbiased questions, no COT, prompt sensitivity format {unbiased_sensitivity_prompt.name()}",
    }
    bias_name = bias_name_map[selected_bias]
    bias_to_leave_out_model_map = {
        WrongFewShotIgnoreMistakesBiasedFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7semB2r8",
        StanfordBiasedFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7smTRQCv",
        MoreRewardBiasedFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7soRFrpt",
        ZeroShotCOTSycophancyFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7skb05DZ",
        # this is the model across all biases
        ZeroShotCOTUnbiasedFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7rg7aRbV",
        DeceptiveAssistantTargetedFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7tWKhqqg",
    }
    biased_model_name = "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV"
    all_read = read_whole_exp_dir(exp_dir="experiments/finetune_2")
    enforce_all_same = False
    biased_task_hashes_1 = (
        (
            all_read.filter(lambda task: task.task_spec.formatter_name == selected_bias.name())
            .filter(
                # intervention is None
                lambda task: task.task_spec.intervention_name
                is None
            )
            .filter(
                # model is the biased model
                lambda task: task.task_spec.inference_config.model
                == "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV"
            )
            .map(lambda task: task.task_spec.task_hash)
        ).to_set()
        if enforce_all_same
        else set()
    )

    biased_task_hashes_2 = (
        (
            all_read.filter(lambda task: task.task_spec.formatter_name == selected_bias.name())
            .filter(
                # intervention is None
                lambda task: task.task_spec.intervention_name
                is None
            )
            .filter(
                # model is the biased model
                lambda task: task.task_spec.inference_config.model
                == "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF"
            )
            .map(lambda task: task.task_spec.task_hash)
        ).to_set()
        if enforce_all_same
        else set()
    )
    biased_task_hashes_3 = (
        (
            all_read.filter(lambda task: task.task_spec.formatter_name == selected_bias.name())
            .filter(
                # intervention is None
                lambda task: task.task_spec.intervention_name
                is None
            )
            .filter(
                # model is the biased model
                lambda task: task.task_spec.inference_config.model
                == "gpt-3.5-turbo"
            )
            .map(lambda task: task.task_spec.task_hash)
        ).to_set()
        if enforce_all_same
        else set()
    )

    biased_task_hashes = biased_task_hashes_1.intersection(biased_task_hashes_2).intersection(biased_task_hashes_3)

    print(f"Number of biased task hashes: {len(biased_task_hashes)}")
    bias_name = bias_name_map[selected_bias]
    make_finetune_graph(
        must_include_task_hashes=biased_task_hashes,
        biased_formatters=[selected_bias],
        finetuned_models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::81Eu4Gp5",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8373aRST",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::83O1S0zn",
        ],
        unbiased_model="gpt-3.5-turbo",
        unbiased_formatter=ZeroShotCOTUnbiasedFormatter,
        all_read=all_read,
        accuracy_plot_name=f"Accuracy on questions {bias_name}<br>Train Dataset: BBH, aqua, arc, Test Dataset: {dataset_str}<br>{filterer.name()}",
        percent_matching_plot_name=f"Percentage of times the model chooses the answer {bias_name}<br>Train Dataset: BBH, aqua, arc, Test Dataset: {dataset_str}<br>{filterer.name()}",
        filterer=filterer,
        tasks=tasks,
        model_name_override=MODEL_SIMPLE_NAMES,
    )
