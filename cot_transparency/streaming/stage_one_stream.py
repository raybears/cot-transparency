from typing import Sequence, Optional, Literal

from grugstream import Observable
from tqdm import tqdm

from cot_transparency.apis import ModelCaller, UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.tasks import task_function
from stage_one import create_stage_one_task_specs


def stage_one_stream(
    tasks: Sequence[str] = [],
    dataset: Optional[str] = None,
    models: Sequence[str] = ["gpt-3.5-turbo", "gpt-4"],
    formatters: Sequence[str] = [ZeroShotCOTSycophancyFormatter.name(), ZeroShotCOTUnbiasedFormatter.name()],
    # Pass in a list of interventions to run, indicate None to run no intervention as well
    interventions: Sequence[str | None] = [],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int] = 1000000,
    batch: int = 20,
    repeats_per_question: int = 1,
    temperature: Optional[float] = None,
    raise_after_retries: bool = True,
    raise_on: Literal["all", "any"] = "all",
    num_tries: int = 10,
    max_tokens: Optional[int] = None,
    n_responses_per_request: Optional[int] = None,
    caller: ModelCaller = UniversalCaller(),
    add_tqdm: bool = True,
) -> Observable[TaskOutput]:
    """A version of stage_one.py, but streaming
    Note that this doesn't manage any cache for you,
    so maybe you want to attach a cache to the ModellCaller with with_file_cache
    """
    tasks_to_run = create_stage_one_task_specs(
        tasks=tasks,
        dataset=dataset,
        models=models,
        formatters=formatters,
        interventions=interventions,
        exp_dir=exp_dir,
        experiment_suffix=experiment_suffix,
        example_cap=example_cap,
        batch=batch,
        repeats_per_question=repeats_per_question,
        temperature=temperature,
        raise_after_retries=raise_after_retries,
        max_tokens=max_tokens,
        n_responses_per_request=n_responses_per_request,
    )

    obs = (
        Observable.from_iterable(tasks_to_run)
        .map_blocking_par(
            lambda task_spec: task_function(
                task=task_spec,
                raise_after_retries=raise_after_retries,
                raise_on=raise_on,
                caller=caller,
                num_tries=num_tries,
            ),
            max_par=batch,
        )
        .flatten_list()
    )
    if add_tqdm:
        obs = obs.tqdm(tqdm_bar=tqdm(total=len(tasks_to_run)))
    return obs
