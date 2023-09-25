from typing import Mapping, Sequence, TypeVar, Tuple

from plotly import graph_objects as go
from pydantic import BaseModel
from slist import Slist, identity

from cot_transparency.data_models.models import TaskOutput
from scripts.intervention_investigation import read_whole_exp_dir, bar_plot, plot_dots_for_intervention
from scripts.multi_accuracy import PlotDots, AccuracyOutput


def modal_agreement(tasks: Slist[TaskOutput]) -> float:
    answer = tasks.map(lambda task: task.first_parsed_response).mode_or_raise()
    return tasks.filter(lambda task: task.first_parsed_response == answer).length / len(tasks)


def modal_agreement_for_task_hash(unique_over_model: Slist[TaskOutput]) -> float:
    # should already be unique over a model
    assert unique_over_model.length > 0, f"grouped_by_task_hash should not be empty"
    # now we have a list of tasks for each task_hash
    # we want to calculate the modal agreement for each task_hash
    modal_agreements: Slist[float] = unique_over_model.group_by(lambda task: task.task_spec.task_hash).map_2(
        lambda task_hash, tasks: modal_agreement(tasks)
    )
    # take the mean of the modal agreements
    average = modal_agreements.average()
    if average is None:
        print("breakpooint")
    assert average is not None
    return average


class BootstrappedOutput(BaseModel):
    median: float
    lower_95: float
    upper_95: float
    samples: float


A = TypeVar("A")


def resample_with_replacement(items: Slist[A], resample_count: int = 100) -> Slist[Slist[A]]:
    return Slist(items.sample(items.length, seed=str(i)) for i in range(resample_count))


def resample_formatter(items: Slist[TaskOutput]) -> Slist[Slist[TaskOutput]]:
    # Resamples for formatters and tasks
    grouped_by_formatters: Slist[tuple[str, Slist[TaskOutput]]] = items.group_by(
        lambda task: task.task_spec.formatter_name
    )
    # resample formatters first
    print("Resampling for formatters")
    resampled: Slist[Slist[tuple[str, Slist[TaskOutput]]]] = resample_with_replacement(grouped_by_formatters)
    # now we need to flatten the list
    output: Slist[Slist[TaskOutput]] = Slist()
    for resample in resampled:
        for formatter, tasks in resample:
            output.append(tasks)
    return output


def bootstrap_over_formatters(items: Slist[TaskOutput]) -> BootstrappedOutput:
    resampled_formatters: Slist[Slist[TaskOutput]] = resample_formatter(items)
    print("Resampling for tasks")
    resampled_tasks: Slist[Slist[Slist[TaskOutput]]] = Slist()
    for formatter, tasks in resampled_formatters:
        resample = resample_with_replacement(tasks)
        resampled_tasks.append(resample)
    print("Resampling done! Calculating modal agreement")
    # apply modal_agreement_for_task_hash
    modal_agreements: Slist[float] = Slist()
    for resample_tasks in resampled_tasks:
        for resample_formatters in resample_tasks:
            modal_agreements.append(modal_agreement_for_task_hash(resample_formatters))
    median = modal_agreements.median_by(identity)
    lower_95 = modal_agreements.percentile_by(key=identity, percentile=0.05)
    upper_95 = modal_agreements.percentile_by(key=identity, percentile=0.95)
    return BootstrappedOutput(median=median, lower_95=lower_95, upper_95=upper_95, samples=items.length)


def prompt_metrics_2(
    exp_dir: str,
    name_override: Mapping[str, str] = {},
    models: Sequence[str] = [],
    formatters: Sequence[str] = [],
):
    read: Slist[TaskOutput] = (
        read_whole_exp_dir(exp_dir=exp_dir)
        .filter(lambda task: task.task_spec.inference_config.model in models if models else True)
        .filter(lambda task: task.task_spec.formatter_name in formatters if formatters else True)
    )

    # group the tasks by model name
    grouped: Slist[tuple[str, Slist[TaskOutput]]] = read.group_by(lambda task: task.task_spec.inference_config.model)
    # calculate modal agreement
    modal_agreement_scores: Slist[tuple[str, BootstrappedOutput]] = grouped.map_2(
        lambda model, tasks: (model, bootstrap_over_formatters(tasks))
    )
    names = modal_agreement_scores.map(lambda x: name_override.get(x[0], x[0]))
    scores = modal_agreement_scores.map(lambda x: x[1].median)
    bootstrapped_metrics: Slist[BootstrappedOutput] = modal_agreement_scores.map(lambda x: x[1])
    fig = go.Figure(
        data=go.Bar(
            name="Modal Agreement Score",
            x=names,
            y=scores,
            error_y=dict(type="data", array=[x.upper_95 - x.median for x in bootstrapped_metrics]),
        )
    )
    # Customize layout
    fig.update_layout(
        title_text="Modal Agreement Score by Model",
        xaxis_title="Model",
        yaxis_title="Modal Agreement Score",
        template="plotly_white",
    )

    # Display plot
    fig.show()

    plot_dots: list[PlotDots] = [
        plot_dots_for_intervention(
            intervention=None,
            all_tasks=read,
            model=model,
            name_override=name_override.get(model, model),
        )
        for model in models
    ]

    bar_plot(
        plot_dots=plot_dots,
        title="Accuracy on test set",
        y_axis_title="Accuracy",
    )


if __name__ == "__main__":
    prompt_metrics_2(
        exp_dir="experiments/sensitivity",
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
        ],
        name_override={
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
        },
    )
