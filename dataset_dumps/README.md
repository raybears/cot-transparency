This folders contains training samples for BCT training with the original GPT-3.5-turbo completions.

The `train_seed_42` folder contains training samples for the BCT training run with seed 42.
The samples are split into three jsonl files:
- `instruct_samples.jsonl`: Samples to retain the original instruction-following behavior of the model.
- `bct_cot.jsonl`: Samples to train the model on BCT in a CoT setting.
- `bct_non_cot.jsonl`: Samples to train the model on BCT in a non-CoT setting.

We separated the samples into these three files for you incase you don't want all of them.
But to reproduce the training set, concatenate the files to get the full training set and shuffle the samples for training.


The `control_seed_42` folder contains control samples for the BCT training run with seed 42.
These prompts don't have BCT applied to them.

Note: If you want to reproduce BCT with a new model, you should take the `control_seed_42` folder and sample from the model to get your unbiased CoTs.
Then, you apply back the biased prompts from the `train_seed_42` folder.



The `train_original_release` folder contains training samples for the original release.
But we forgot to include the paired control samples in the original release, so you should ignore this folder unless you want to lookback at the original release.