import argparse
import json
from datasets import load_dataset
import os
import numpy as np
import random


def get_correct_prompt_answer(prompts, selected_answers, orig_selected_answers=None, dreamer=False):
    """
    Get the correct answer for each prompt from the selected answers
    """
    assert len(prompts) == len(selected_answers)
    if orig_selected_answers is not None:
        assert len(prompts) == len(orig_selected_answers)
    selected_prompts_answers = {'prompts':[], 'answers':[], 'orig_answers':[]}
    # for prompt, selected_answer, orig_selected_answer in zip(prompts, selected_answers, orig_selected_answers):
    for i, (prompt, selected_answer) in enumerate(zip(prompts, selected_answers)):
        if orig_selected_answers:
            orig_selected_answer = orig_selected_answers[i]
        if selected_answer:
            if not dreamer:
                selected_prompts_answers['prompts'].append(prompt)
                # currently we only randomly select one answer
                random.seed(0)
                selected_idx = random.randint(0, len(selected_answer) - 1)
                selected_prompts_answers['answers'].append(selected_answer[selected_idx])
            else:
                dreamer_prompts = [ans['question'] for ans in selected_answer]
                # select the answer that has minimum lexical overlap with the prompt
                selected_idx = np.argmin([len(set(dreamer_prompt.split()) & set(prompt.split()))/len(set(prompt.split())) for dreamer_prompt in dreamer_prompts])
                # currently we randomly select one answer
                random.seed(0)
                ans_idx = random.randint(0, len(selected_answer[selected_idx]['answer']) - 1)
                selected_prompts_answers['answers'].append(selected_answer[selected_idx]['answer'][ans_idx])
                selected_prompts_answers['prompts'].append(selected_answer[selected_idx]['question'])
                if orig_selected_answers:
                    random.seed(0)
                    orig_selected_idx = random.randint(0, len(orig_selected_answer) - 1)
                    selected_prompts_answers['answers'].append(orig_selected_answer[orig_selected_idx])
                    selected_prompts_answers['prompts'].append(prompt)
    if selected_answers and orig_selected_answers:
        print(f"Total prompts: {len(prompts)*2}, selected prompts: {len(selected_prompts_answers['prompts'])}")
    else:
        print(f"Total prompts: {len(prompts)}, selected prompts: {len(selected_prompts_answers['prompts'])}")
    return selected_prompts_answers


def sft_formatter(prompts, answers, dreamer=False):
    """
    Format the prompts and answers for SFT
    """
    if not dreamer:
        #print one example
        print('Example instruction: ')
        print(prompts[0])
        print('Example output: ')
        print(answers[0])
        return [{"instruction": prompt, "input": "", "output": answer} for prompt, answer in zip(prompts, answers)]
    else:
        return [{"instruction": prompt, "input": "", "output": answer} for prompt, answer in zip(prompts, answers)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=False, default="../data/asynchow/asynchow_natural")
    parser.add_argument('--selected_answer_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=False, default='../data')
    parser.add_argument('--task', type=str, required=False, default='asynchow')
    parser.add_argument('--out_name', type=str, required=False, default='')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dreamer', action='store_true', required=False, default=False)
    args = parser.parse_args()
    if args.input_path.endswith('.json'):
        with open(args.input_path, 'r') as f:
            input_data = json.load(f)
        prompts = [sample['instruction'] for sample in input_data]
    else:
        input_data = load_dataset('json', data_files=args.input_path+'_train.jsonl')
        prompts = [sample['question'] for sample in input_data]
    with open(args.selected_answer_path, 'r') as f:
        answer_data = json.load(f)
    selected_answers = answer_data['selected_samples']
    orig_selected_answers = None

    selected_prompts_answers = get_correct_prompt_answer(prompts, selected_answers, orig_selected_answers, args.dreamer)
    if args.out_name == '':
        out_path = os.path.join(args.output_path, f'{args.model_name.split("/")[-1]}_{args.task}.json')
    else:
        out_path = os.path.join(args.output_path, f'{args.model_name.split("/")[-1]}_{args.task}_{args.out_name}.json')
    with open(out_path, 'w') as f:
        json.dump(sft_formatter(selected_prompts_answers['prompts'], selected_prompts_answers['answers'], dreamer=False), f) # we hardcode dreamer=False here for ablation
    
    # add data information to dataset_info.json
    dataset_info_path = os.path.join(args.output_path, 'dataset_info.json')
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    dataset_info[f'{args.model_name.split("/")[-1]}_{args.task}'] = {'file_name': out_path.split('/')[-1]}
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)

if __name__ == '__main__':
    main()
