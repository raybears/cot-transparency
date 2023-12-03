from typing import Mapping, Sequence

from slist import Slist, Group

from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.models import BaseTaskOutput, TaskOutput
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import AccuracyOutput, PlotInfo


def modal_agreement(tasks: Slist[BaseTaskOutput]) -> float:
    answer = tasks.map(lambda task: task.inference_output.parsed_response).mode_or_raise()
    return tasks.filter(lambda task: task.inference_output.parsed_response == answer).length / len(tasks)





def modal_is_wrong(tasks: Slist[TaskOutput]) -> bool:
    answer = tasks.map(lambda task: task.first_parsed_response).mode_or_raise()
    distinct_task_hashes = tasks.map(lambda task: task.task_spec.task_hash).distinct_unsafe()
    assert len(distinct_task_hashes) == 1, f"should only be one task hash, got {distinct_task_hashes.length}"
    first_task = tasks.first_or_raise()
    correct_answer = first_task.task_spec.ground_truth

    return answer != correct_answer


def modal_agreement_for_task_hash(_unique_over_model: Sequence[BaseTaskOutput]) -> float:
    # should already be unique over a model
    unique_over_model = Slist(_unique_over_model)
    assert unique_over_model.length > 0, "grouped_by_task_hash should not be empty"
    grouped_by_qn = unique_over_model.group_by(lambda task: task.get_task_spec().get_task_hash())
    # now we have a list of tasks for each task_hash
    # we want to calculate the modal agreement for each task_hash
    modal_agreements: Slist[float] = grouped_by_qn.map_2(lambda task_hash, tasks: modal_agreement(tasks))
    # take the mean of the modal agreements
    average = modal_agreements.average()
    assert average is not None
    return average


def calculate_modal_agreement(name: str, items: Slist[TaskOutput]) -> PlotInfo:
    # non bootstrap version
    # run modal_agreement_for_task_hash
    modal_agreement = modal_agreement_for_task_hash(items)
    return PlotInfo(
        name=name,
        acc=AccuracyOutput(accuracy=modal_agreement, error_bars=0, samples=items.length),
    )


def filter_modal_wrong(items: Slist[TaskOutput]) -> Slist[TaskOutput]:
    print("Filtering modal wrong")
    # group them by the model, task hash
    grouped_by_model: Slist[Group[str, Slist[TaskOutput]]] = items.group_by(
        lambda task: task.task_spec.inference_config.model + task.task_spec.task_hash
    )
    # for each model, filter out the ones that are modal wrong
    filtered: Slist[Slist[TaskOutput]] = Slist()
    for model, tasks in grouped_by_model:
        if modal_is_wrong(tasks):
            filtered.append(tasks)
    return filtered.flatten_list()


def prompt_metrics_plotly(
    exp_dir: str,
    only_modally_wrong: bool = True,
    name_override: Mapping[str, str] = {},
    models: Sequence[str] = [],
    formatters: Sequence[str] = [],
    tasks: Sequence[str] = [],
):
    print(f"filtering on {len(formatters)} formatters")
    read: Slist[TaskOutput] = (
        read_whole_exp_dir(exp_dir=exp_dir)
        .filter(lambda task: task.task_spec.inference_config.model in models if models else True)
        .filter(lambda task: task.task_spec.formatter_name in formatters if formatters else True)
        .filter(lambda task: task.task_spec.task_name in tasks if tasks else True)
    )
    if only_modally_wrong:
        read = filter_modal_wrong(read)

    # group the tasks by model name
    grouped: Slist[Group[str, Slist[TaskOutput]]] = read.group_by(lambda task: task.task_spec.inference_config.model)
    # calculate modal agreement
    modal_agreement_scores: Slist[PlotInfo] = grouped.map_2(
        lambda model, items: calculate_modal_agreement(name=model, items=items)
    )
    # order by original order
    modal_agreement_scores = modal_agreement_scores.sort_by(lambda plot_dots: models.index(plot_dots.name))

    dataset_str = Slist(tasks).mk_string(", ")
    bar_plot(
        plot_infos=modal_agreement_scores,
        name_override=name_override,
        title=f"Modal Agreement Score by Model<br>{dataset_str}<br>{len(formatters)} formatters",
        y_axis_title="Modal Agreement Score",
        add_n_to_name=True,
    )

    plot_dots: list[PlotInfo] = [
        plot_for_intervention(
            intervention=None,
            all_tasks=read,
            model=model,
            name_override=name_override.get(model, model),
            distinct_qns=False,
        )
        for model in models
    ]

    bar_plot(
        plot_infos=plot_dots,
        title=f"Accuracy on {dataset_str}<br> {len(formatters)} formatters",
        y_axis_title="Accuracy",
        add_n_to_name=True,
    )
