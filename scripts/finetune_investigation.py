from typing import Sequence, Type, Optional, Mapping

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from cot_transparency.formatters.verbalize.formatters import StanfordBiasedFormatter
from scripts.intervention_investigation import (
    read_whole_exp_dir,
    plot_dots_for_intervention,
    DottedLine,
    bar_plot,
    ConsistentOnly,
    TaskOutputFilter, InconsistentOnly,
)
from scripts.matching_user_answer import matching_user_answer_plot_dots
from scripts.multi_accuracy import PlotDots


def make_finetune_graph(
    biased_formatters: Sequence[Type[StageOneFormatter]],
    finetuned_models: Sequence[str],
    unbiased_model: str,
    unbiased_formatter: Type[StageOneFormatter],
    exp_dir: str,
    accuracy_plot_name: Optional[str] = None,
    percent_matching_plot_name: Optional[str] = None,
    filterer: Type[TaskOutputFilter] = ConsistentOnly,
    tasks: Sequence[str] = [],
    model_name_override: Mapping[str | None, str] = {},
):
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir=exp_dir)
    all_read = filterer.filter(all_read).filter(lambda task: task.task_spec.task_name in tasks if tasks else True)

    # unbiased acc
    unbiased_plot: PlotDots = plot_dots_for_intervention(
        None,
        all_read,
        for_formatters=[unbiased_formatter],
        name_override=f"Unbiased context, {unbiased_model}",
        model=unbiased_model,
    )

    plot_dots: Slist[PlotDots] = Slist(finetuned_models).map(
        lambda m: plot_dots_for_intervention(
            None,
            all_read,
            for_formatters=biased_formatters,
            model=m,
            name_override=model_name_override.get(m, m),
        ).add_n_samples_to_name()
    )
    dotted = DottedLine(name="Zero shot unbiased context performance", value=unbiased_plot.acc.accuracy, color="red")

    bar_plot(
        plot_dots=plot_dots,
        title=accuracy_plot_name or "Accuracy of different models in biased contexts",
        dotted_line=dotted,
        y_axis_title="Accuracy",
    )
    # accuracy_diff_intervention(interventions=interventions, unbiased_formatter=unbiased_formatter)
    matching_user_answer: list[PlotDots] = [
        matching_user_answer_plot_dots(
            intervention=None,
            all_tasks=all_read,
            for_formatters=biased_formatters,
            model=model,
            name_override=model_name_override.get(model, model),
        ).add_n_samples_to_name()
        for model in finetuned_models
    ]
    unbiased_matching_baseline = matching_user_answer_plot_dots(
        intervention=None, all_tasks=all_read, for_formatters=[unbiased_formatter], model=unbiased_model
    )
    dotted_line = DottedLine(
        name="gpt-3.5-turbo in Unbiased context,zeroshot", value=unbiased_matching_baseline.acc.accuracy, color="red"
    )
    dataset_str = Slist(tasks).mk_string(", ")
    bar_plot(
        plot_dots=matching_user_answer,
        title=percent_matching_plot_name or f"How often does each model choose the user's view Dataset: {dataset_str}",
        y_axis_title="Answers matching bias's view (%)",
        dotted_line=dotted_line,
    )


if __name__ == "__main__":
    filterer = InconsistentOnly
    tasks = ["truthful_qa", "logiqa", "hellaswag", "mmlu"]
    dataset_str = Slist(tasks).mk_string(", ")
    selected_bias = WrongFewShotIgnoreMistakesBiasedFormatter
    bias_name_map = {
        WrongFewShotIgnoreMistakesBiasedFormatter: "Wrong Fewshot",
        StanfordBiasedFormatter: "Stanford Professor Opinion",
        MoreRewardBiasedFormatter: "More Reward for (X)",
        ZeroShotCOTSycophancyFormatter: "I think the answer is (X)",
    }
    bias_to_leave_out_model_map = {
        WrongFewShotIgnoreMistakesBiasedFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7semB2r8",
        StanfordBiasedFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7smTRQCv",
        # MoreRewardBiasedFormatter: None,  # none yet
        ZeroShotCOTSycophancyFormatter: "ft:gpt-3.5-turbo-0613:academicsnyuperez::7skb05DZ",
    }

    bias_name = bias_name_map[selected_bias]
    make_finetune_graph(
        biased_formatters=[selected_bias],
        finetuned_models=[
            "gpt-3.5-turbo",
            bias_to_leave_out_model_map[selected_bias],
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::7semB2r8"
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::7semB2r8",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::7ryTmccr",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::7rg7aRbV",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::7skb05DZ",
            # "ft:gpt-3.5-turbo-0613:academicsnyuperez::7smTRQCv",
        ],
        unbiased_model="gpt-3.5-turbo",
        unbiased_formatter=ZeroShotCOTUnbiasedFormatter,
        exp_dir="experiments/finetune",
        accuracy_plot_name=f"Accuracy on questions biased by a {bias_name}<br>Train Dataset: BBH, Test Dataset: {dataset_str}<br>{filterer.name()}",
        percent_matching_plot_name=f"Percentage of times the model chooses the answer biased by a {bias_name}?<br>Train Dataset: BBH, Test Dataset: {dataset_str}<br>{filterer.name()}",
        filterer=filterer,
        tasks=tasks,
        model_name_override={
            "gpt-3.5-turbo": "gpt-3.5-turbo ",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::7ryTmccr": "Finetuned 6000 COTs with unbiased questions",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::7semB2r8": "Finetuned 6000 COTs with biased questions, leaving out bias of Wrong Fewshot",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::7rg7aRbV": "Finetuned 6000 COTs with biased questions, including bias of Wrong Fewshot",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::7skb05DZ": "Finetuned 6000 COTs with biased questions, leaving out bias of I think the answer is (X)",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::7smTRQCv": "Finetuned 6000 COTs with biased questions, leaving out bias of Stanford Professor opinion",
        },
    )
