import pandas as pd
import torch
import os
import json
import argparse
# import concurrent.futures

from dotenv import load_dotenv
from constants import BBH_TASK_LIST, ANS_MAPPING, FILENAMES
from inspect_layers import Llama7BHelper

from utils import check_directory, save_csv, load_df

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

def print_intermediate_results(df, filename):
    df['correct_ans'] = df['correct_ans'].astype(str)
    df.loc[:, f'layer_{START_LAYER}':f'layer_{END_LAYER}'] = df.loc[:, f'layer_{START_LAYER}':f'layer_{END_LAYER}'].astype(str)
    accuracy_scores = {}
    for layer in df.columns:
        if layer.startswith('layer_'):
            accuracy_scores[layer] = (df['correct_ans'] == df[layer]).mean()
    df_accuracy = pd.DataFrame(accuracy_scores, index=[0])
    save_csv(df_accuracy, export_path + filename)
    print(f'Accuracy Scores so far for {filename}: {accuracy_scores}')

def load_all_df(export_path):
    dataframes = {}
    for filename in FILENAMES:
        df = load_df(res_schema, export_path, f'{filename}.csv')
        dataframes[filename] = df
    return tuple(dataframes[f] for f in FILENAMES)

def layers_accuracy_runner(prompt, answer, export_path):
    df_attention_acc, df_residual_acc, df_mlp_acc, df_block_acc = load_all_df(export_path)
    df_attention_prob, df_residual_prob, df_mlp_prob, df_block_prob = load_all_df(export_path)

    with torch.no_grad():
        res = llama7b_helper.decode_all_layers(prompt, answer, topk=1, start_layer=START_LAYER, end_layer=END_LAYER)
        llama7b_helper.reset_all() # reset all layers

    layer_wise_attention, layer_wise_residual, layer_wise_mlp, layer_wise_block = res

    dfs_acc = [df_attention_acc, df_residual_acc, df_mlp_acc, df_block_acc]
    dfs_prob = [df_attention_prob, df_residual_prob, df_mlp_prob, df_block_prob]
    layer_wise_data = [layer_wise_attention, layer_wise_residual, layer_wise_mlp, layer_wise_block]

    for df_acc, df_prob, layer_acc, layer_prob, filename in zip(dfs_acc, dfs_prob, layer_wise_data[0], layer_wise_data[1], FILENAMES):
        df_acc = pd.concat([df_acc, pd.DataFrame(layer_acc)])
        df_prob = pd.concat([df_prob, pd.DataFrame(layer_prob)])
        save_csv(df_acc, export_path + f'{filename}_acc.csv')
        save_csv(df_prob, export_path + f'{filename}_prob.csv')
        print_intermediate_results(df_acc, f'{filename}_acc_mean.csv')
        print_intermediate_results(df_prob, f'{filename}_prob_mean.csv')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='ruin_names')
    # parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads to use.')
    parser.add_argument('--biased_context', type=bool, default=False)
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--LLextension', type=bool, default=False, help='Use the LL extension')
    parser.add_argument('--start_layer', type=int, default=15, help='Start layer')
    parser.add_argument('--end_layer', type=int, default=31, help='End layer')
    
    args = parser.parse_args()

    all_file = args.file
    task_name = 'all_data' if all_file else args.task_name
    num_workers = args.num_workers
    biased_context = args.biased_context
    extension = args.LLextension
    START_LAYER = args.start_layer
    END_LAYER = args.end_layer

    res_schema = pd.DataFrame({"question": [], **{f"layer_{i}": [] for i in range(START_LAYER, END_LAYER)}}) # result schema dataframe
    
    llama7b_helper = Llama7BHelper(hf_token, extension)

    if task_name not in BBH_TASK_LIST:
        raise ValueError(f"Invalid task_name. It must be one of: {', '.join(BBH_TASK_LIST)}")

    data_path = 'val_data.json' if not all_file else all_file
    # data_path = f'../data/bbh/{task_name}/val_data.json' if not all_file else f'../data/bbh/{all_file}'
    export_path = f'experiments/{task_name}/' if not biased_context else f'experiments/{task_name}_biased_context/'

    check_directory('experiments')
        
    if not biased_context:
        check_directory(f'experiments/{task_name}')
    else:
        check_directory(f'experiments/{task_name}_biased_context')

    with open(data_path, 'r') as json_file:
        data_list = json.load(json_file)['data'] if not all_file else [json.loads(line) for line in json_file]

        for i, obj in enumerate(data_list):
            # if i < 1263:
            #     continue
            print(f'\n\nQuestion {i+1} of {len(data_list)}\n')
            question = obj['parsed_inputs']
            correct_ans_index = obj['multiple_choice_scores'].index(1)
            correct_ans = ANS_MAPPING[correct_ans_index]
            direct_answer_trigger = "\nThe best answer is: ("
            random_index_label = ANS_MAPPING[obj['random_ans_idx']]
            biased_instr = f"\nI think the answer is ({random_index_label}) but I'm curious to hear what you think."
            prompt = question + direct_answer_trigger if not biased_context else question + biased_instr + direct_answer_trigger
            print(prompt)
            layers_accuracy_runner(prompt, correct_ans, export_path)
