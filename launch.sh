#!/bin/bash
set -euxo pipefail

example_cap=50
dataset='transparency'
exp_dir=experiments/stage_one/prompt_sensitivity/try_5_many_formatters_intervention

# Declare the initial name as a variable
initial_name="ZeroShotPromptSenFormatter"

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

python stage_one.py --models "['gpt-3.5-turbo', 'gpt-4']" --formatters "['ZeroShotPromptSenFormatter*']" --repeats_per_question 1 --batch=40 --exp_dir $exp_dir --example_cap $example_cap --tasks "['aqua']" --interventions "['VanillaFewShotLabelOnly10']"
