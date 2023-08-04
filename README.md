# SERI MATS: Chain of Though Transparency

This is a private fork of [cot-unfaithfulness](https://github.com/milesaturpin/cot-unfaithfulness)

[![Build Status](https://github.com/raybears/cot-transparency/actions/workflows/main.yml/badge.svg)](https://github.com/raybears/cot-transparency/actions/workflows/main.yml)

## Installation

Install python environment, requires python >= 3.11

Pyenv (we need tkinter hence the extra steps)

```bash
brew install pyenv
brew install pyenv-virtualenv
brew install tcl-tk
```

```bash
PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I/usr/local/opt/tcl-tk/include' --with-tcltk-libs='-L/usr/local/opt/tcl-tk/lib -ltcl8.6 -ltk8.6'" pyenv install 3.11.1
pyenv virtualenv 3.11 cot
pyenv local cot
```

Install requirements

```
make env
```

Install pre-commmit hooks

```bash
make hooks
```

## Checks

To run linting / type checks

```bash
make check
```

To run tests

```bash
pytest tests
```

## Usage

Set your OpenAI API key as `OPENAI_API_KEY` in a `.env` file.

To generate examples e.g. this will compare 20 samples for each task in bbh for sycophancy

```python
python stage_one.py --exp_dir experiments/stage_one/dummy_run --models "['text-davinci-003']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'ZeroShotCOTSycophancyFormatter']" --repeats_per_question 1 --batch=10 --example_cap 20
```

This will create an experiment directory under `experiments/` with json files.

To run analysis

```python
python analysis.py accuracy --exp_dir experiments/stage_one/dummy_run
```

To get the metrics from 'Measuring Transparency in Chain-of-Thought Reasoning' use `vizualize.ipynb`.