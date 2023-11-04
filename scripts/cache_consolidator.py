# convert task outputs to cache
import fire

from tqdm import tqdm

from cot_transparency.apis.base import APIRequestCache, CachedValue, InferenceResponse, file_cache_key
from cot_transparency.data_models.io import read_whole_exp_dir


def main(input_cache_dir: str, output_cache_dir: str):
    caches: dict[str, APIRequestCache] = {}
    outputs = read_whole_exp_dir(exp_dir=input_cache_dir)

    for output in tqdm(outputs):
        model = output.get_task_spec().inference_config.model
        out_cache_path = f"{output_cache_dir}/{model}.jsonl"
        if out_cache_path not in caches:
            cache = APIRequestCache(out_cache_path)
            caches[out_cache_path] = cache.load()
        else:
            cache = caches[out_cache_path]
        config = output.get_task_spec().inference_config
        assert config.n == 1
        key = file_cache_key(messages=output.get_task_spec().messages, config=config)
        cache[key] = CachedValue(
            response=InferenceResponse(raw_responses=[output.inference_output.raw_response]),
            messages=output.get_task_spec().messages,
            config=output.get_task_spec().inference_config,
        )

    for cache in tqdm(caches.values(), desc="Saving caches"):
        cache.save()


if __name__ == "__main__":
    fire.Fire(main)
