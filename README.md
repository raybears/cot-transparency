# SERI MATS: Chain of Thought Transparency

This is a private fork of [cot-unfaithfulness](https://github.com/milesaturpin/cot-unfaithfulness)

[![Build Status](https://github.com/raybears/cot-transparency/actions/workflows/main.yml/badge.svg)](https://github.com/raybears/cot-transparency/actions/workflows/main.yml)

## Installation

Install python environment, requires python >= 3.11.4

Pyenv (we need tkinter hence the extra steps)

```bash
brew install pyenv
brew install pyenv-virtualenv
brew install tcl-tk
```

```bash
pyenv install 3.11.4
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

Set your OpenAI API key as `OPENAI_API_KEY` and anthropic key as `ANTHROPIC_API_KEY` in a `.env` file.

To generate examples e.g. this will compare 20 samples for each task in bbh for sycophancy

```python
python stage_one.py --exp_dir experiments/stage_one/dummy_run --models "['text-davinci-003']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'ZeroShotCOTSycophancyFormatter']" --repeats_per_question 1 --batch=10 --example_cap 20
```

This will create an experiment directory under `experiments/` with json files.

To run analysis

```python
python analysis.py accuracy --exp_dir experiments/stage_one/dummy_run
```
