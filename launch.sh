#!/bin/bash
set -euxo pipefail

example_cap=300
dataset='transparency'
exp_dir=experiments/stage_one/prompt_sensitivity/cot_try4_with_none_option

# Declare the initial name as a variable
initial_name="NoCotPromptSenFormatter"

# Define the list of suffixes
suffixes=("ROMAN" "NUMBERS" "FOO" "LETTERS")

# Start the formatters list
formatters="["

# Loop through the suffixes to construct each formatter
for suffix in "${suffixes[@]}"; do
    formatters+="\"${initial_name}_${suffix}_FULL_ANS_CHOICES\", "
done

# Remove the trailing comma and space, then add the closing bracket
formatters="${formatters%,*}]"

LOG_RATE_LIMITS=true python stage_one.py --models "['gpt-3.5-turbo-16k', 'gpt-4']" --formatters "['CotPromptSenFormatter*FULL_ANS_CHOICES*']" --repeats_per_question 1 --batch=40 --exp_dir $exp_dir --example_cap $example_cap --tasks "['aqua']" --interventions "['VanillaFewShot10']"

python stage_one.py --models "['gpt-3.5-turbo-16k', 'gpt-4']" --formatters "['NoCotPromptSenFormatter*FULL_ANS_CHOICES*']" --repeats_per_question 1 --batch=40 --exp_dir $exp_dir --example_cap $example_cap --tasks "['aqua']" --interventions "['VanillaFewShotLabelOnly10', 'VanillaFewShotLabelOnly20']"
