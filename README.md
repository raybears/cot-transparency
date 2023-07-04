# SERI MATS: Chain of Though Transparency

This is a private fork of [cot-unfaithfulness](https://github.com/milesaturpin/cot-unfaithfulness)

## Installation
Install python environment

Pyenv:
```bash
pyenv install 3.9
pyenv virtualenv 3.9 cot
```

Install requirements
```
make env
```

Install pre-commmit hooks
```bash
make hooks
```

## Usage
Set your OpenAI API key in `OPENAI_API_KEY` environment variable.
```bash
export OPENAI_API_KEY=xxx
```

To generate examples e.g. 
```python
python run_eval.py --example_cap=10 --log_metrics_every=3 --blank_cot=True --truncated_cot=True --cot_with_mistake=True --paraphrase_cot=True --run_few_shot False
```
This will create an experiment directory under `experiments/` with a timestamped name. For all options see `run_eval.py::main()` or run `python run_eval.py --help`

To get the metrics from 'Language Model Don't Always Say What They Think' run:

```python
python bbh_analysis.py --exp_dir experiments/<exp_timestamp>
```

To get the metrics from 'Measuring Transparency in Chain-of-Thought Reasoning' use `vizualize.ipynb`.