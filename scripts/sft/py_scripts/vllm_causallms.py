import argparse
import copy
import re
import os
import json
import logging
import random
from importlib.metadata import version
from pathlib import Path
from importlib.util import find_spec
from asynchow_utils import *
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union
from physic_parser import *
from math_parser import *
from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm
from datasets import load_dataset

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
# from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    handle_stop_sequences,
    undistribute,
)
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)

try:
    import ray
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

import torch

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

eval_logger = logging.getLogger(__name__)


def extract_answer(output: str, task: str) -> Optional[int]:
    """Extract an integer answer from the generated output."""
    if task == 'prototypical':
        match = re.search(r'answer is (\d+)', output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # match yes or no
        if "answer is yes" in output:
            return "yes"
        else:
            assert "answer is no" in output, ("Exceptional case")
            return "no"
    else:
        return output


def eval_instance(output: str, ground_truth, task):
    """Extract an integer answer from the generated output."""
    if task == 'prototypical':
        ground_truth = extract_answer(ground_truth)
        binary = type(ground_truth)==str
        if not binary:
            match = re.search(r'is (\d+)', output, re.IGNORECASE)
            if match:
                return int(match.group(1))==ground_truth
        else:
            lower_output = set(output.lower().split())
            # match yes or no
            if 'no.' not in lower_output and 'no,' not in lower_output and 'no' not in lower_output:
                return 'yes' == ground_truth
            else:
                return 'no' == ground_truth
        return False
    elif task == 'explagraph':
        if '<answer>support</answer>' in output and '<answer>counter</answer>' not in output:
            if '<answer>support</answer>' == ground_truth:
                return True
        elif '<answer>counter</answer>' in output and '<answer>support</answer>' not in output:
            if '<answer>counter</answer>' == ground_truth:
                return True
        return False
    elif task in ['asynchow', 'asynchow_graph', 'asynchow/asynchow_natural', 'asynchow/asynchow_graph', 'asynchow/asynchow_python', 'proto_graph']:
        return check_correctness_comprehensive(output, ground_truth, task)
    elif task in ["advanced_math", 'advanced_physics', 'chemistry', 'games']:
        return check_correctness_other(output, ground_truth, task)
    elif task in ['igsm', 'prontoqa', 'proofwriter']:
        if isinstance(ground_truth, str):
            if '<answer>' in ground_truth:
                return ground_truth in output
        return '<answer>'+str(ground_truth)+'</answer>' in output
    else:
        raise Exception(f"Task {task} not implemented")


def check_correctness_other(output, ground_truth, task):
    ans = find_answer_comprehensive(output)
    try:
        if task in ['advanced_physics']:
            comparitor = PhysicsSolutionComparitor(ans, ground_truth,  1)
            res = comparitor.compare_solution_to_reference()
        elif task in ['games']:
            return int(ans.strip().lower()==ground_truth.strip().lower())
        else:
            res = process_results(ans, ground_truth)
        assert int(res) in [0, 1]
        return int(res)
    except Exception as e:
        return 0


def eval_generation(outputs, labels, task, return_samples=False, round_acc=True, dreamer=False):
    labels = [extract_answer(label, task) for label in labels]
    if not dreamer:
        correct = [[eval_instance(outputs[i][j], labels[i], task) for j in range(len(outputs[i]))] for i in range(len(outputs))]
        acc = np.mean([True in c for c in correct])
    else:
        correct = [[[eval_instance(outputs[i][j]['answer'][k], labels[i], task) for k in range(len(outputs[i][j]['answer']))] for j in range(len(outputs[i]))] for i in range(len(outputs))]
        acc = np.mean([True in c for cs in correct for c in cs])
    if round_acc:
        acc = round(acc, 3)
    print(f'Current model accuracy is :{acc}')
    if return_samples:
        return correct, acc
    return acc


def rejection_sampler(outputs,
                      eval_res,
                      n_per_prompt=4,
                      fill_with_false=False,
                      sample_cot=True):
    """
    Selects up to n_per_prompt generations per prompt based on boolean evaluation results.
    
    For each prompt:
      - First, select all generations with a True evaluation.
      - If the number of True outputs is less than n_per_prompt, fill up the selection
        with the remaining (False) outputs, preserving their original order.
      - If there are more True outputs than n_per_prompt, randomly sample n_per_prompt of them.
    
    Args:
        outputs (dict): Dictionary containing 'generated_outputs', a list of lists (one per prompt).
        eval_res (list of lists): Boolean evaluation matrix corresponding to each generation.
        n_per_prompt (int): Maximum number of selected generations per prompt.
    
    Returns:
        List of lists: A list (per prompt) containing the selected generations.
    """
    selected_samples = []
    filled_idxes = []
    
    # Loop over each prompt's generated outputs and its evaluation results.
    for prompt_idx, gen_outputs in tqdm(enumerate(outputs), total=len(outputs), desc="Processing prompts"):
        # Extract the boolean evaluation results for this prompt.
        eval_row = eval_res[prompt_idx]
        if not eval_row:
            selected_samples.append([])
            continue
        if type(eval_row[0]) is bool:
            # First, select the outputs evaluated as True.
            accepted = [gen for gen, flag in zip(gen_outputs, eval_row) if flag]
            if sample_cot:
                accepted = [gen for gen in accepted if not gen.startswith("<answer>")]

            if len(accepted) < n_per_prompt:
                if fill_with_false:
                    # If there are fewer than n_per_prompt True outputs, fill up with the False ones.
                    false_outputs = [gen for gen, flag in zip(gen_outputs, eval_row) if not flag]
                    num_needed = n_per_prompt - len(accepted)
                    accepted.extend(false_outputs[:num_needed])
            else:
                filled_idxes.append(prompt_idx)

            # # If there are more than n_per_prompt accepted outputs, randomly sample n_per_prompt.
            # elif len(accepted) > n_per_prompt:
            #     random.seed(1234)
            #     accepted = random.sample(accepted, n_per_prompt)
            
            selected_samples.append(accepted)
        else:
            accepted = []
            print(f'eval_row: {eval_row}')
            print(f'gen_outputs: {gen_outputs}')
            for i, gen in enumerate(gen_outputs):
                curr_eval_row = eval_row[i]
                if not curr_eval_row:
                    continue
                assert len(gen['answer']) == len(curr_eval_row)
                curr_accepted = [ans for ans, flag in zip(gen['answer'], curr_eval_row) if flag]
                if sample_cot:
                    curr_accepted = [ans for ans in curr_accepted if not ans.startswith("<answer>")]

                if curr_accepted:
                    accepted.append({'orig_output': gen['orig_output'], 'question': gen['question'], 'answer': curr_accepted})
            if len(accepted) < n_per_prompt:
                if fill_with_false:
                    # If there are fewer than n_per_prompt True outputs, fill up with the False ones.
                    false_outputs = [gen for gen, flag in zip(gen_outputs, eval_row) if flag not in eval_row[0]]
                    num_needed = n_per_prompt - len(accepted)
                    accepted.extend(false_outputs[:num_needed])
            else:
                filled_idxes.append(prompt_idx)
            selected_samples.append(accepted)

    return selected_samples, filled_idxes


def get_unfilled_indices(total, filled_idxes):
    """Return indices that are not in filled_idxes."""
    return [i for i in range(total) if i not in filled_idxes]


def gen_and_reject(model,
                   prompts,
                   sampling_params,
                   out_dir,
                   labels,
                   task,
                   all_prompt_count,
                   chat_template=True,
                   record_likelihoods=False,
                   return_samples=True,
                   n_per_prompt=1,
                   prev_generated_outputs=None,
                   filled_idxes=None,
                   overwrite=False):
    """
    Generate model outputs (or load previous ones), evaluate them, perform rejection sampling,
    and update filled indices based on accepted samples.
    """
    if filled_idxes is None:
        filled_idxes = []
        
    results_path = os.path.join(out_dir, 'output_likelihood.json')
    # # load previous results if file exists
    # if os.path.exists(results_path):
    #     with open(results_path, 'r') as f:
    #         data = json.load(f)
    #         prev_generated_outputs = data.get('generated_outputs')
    #         filled_idxes = data.get('filled_idxes', filled_idxes)
    # else:
    # If no previous outputs, generate new ones
    generated_outputs = model.generate_with_loglikelihood(prompts,
                                                            sampling_params=sampling_params,
                                                            chat_template=chat_template, 
                                                            record_likelihood=record_likelihoods)
    # print one example
    print('Example prompt: ')
    print(prompts[0])
    print('Example output: ')
    print(generated_outputs[0])
    
    # If previous outputs exist, merge new generations into the unfilled positions.
    if prev_generated_outputs is not None:
        # Determine the original indices corresponding to these new prompts
        current_indices = get_unfilled_indices(all_prompt_count, filled_idxes)
        for i, gen in enumerate(generated_outputs):
            # Extend the previously generated outputs at the corresponding original index
            prev_generated_outputs[current_indices[i]].extend(gen)
        generated_outputs = prev_generated_outputs

    # Save outputs with prompts
    with open(results_path, 'w') as f:
        json.dump({'prompts': prompts, 'generated_outputs': generated_outputs}, f)

    # Evaluate the generated outputs.
    eval_res, acc = eval_generation(generated_outputs, labels, task=task, return_samples=return_samples)
    
    # Perform rejection sampling.
    # Assume rejection_sampler returns:
    #   selected_samples: accepted outputs per prompt,
    #   new_unfilled: indices (relative to current prompt list) that did not pass rejection.
    selected_samples, new_filled = rejection_sampler(generated_outputs, eval_res, n_per_prompt=n_per_prompt)
    
    # Save rejection sampling results
    rejection_path = os.path.join(out_dir, 'rejection_sampling.json')
    with open(rejection_path, "w") as f:
        json.dump({"selected_samples": selected_samples, "filled_idxes": new_filled}, f)
    
    return generated_outputs, new_filled

def regenerate_with_few_shot(model,
                             prompts,
                             sampling_params,
                             out_dir,
                             labels,
                             task,
                             generated_outputs=None,
                             filled_idxes=None,
                             nshot=5,
                             chat_template=True,
                             record_likelihoods=False,
                             max_iter=5,
                             overwrite=True,
                             iter_increase_n_samples=2,
                             return_samples=True,
                             n_per_prompt=1,
                             rationalization=False,
                             initialization=True,
                             dreamer=False):

    """
    Regenerate outputs for prompts that have not received sufficiently good generations.
    Uses few-shot examples from already accepted outputs.
    """
    dreamer_template = '''I will give you a task, then you will generate 1 analogue of the task and solve them. When finishing generating and solving the analogue problem, you will solve the original task.

For each analogue:
1. *Domain shift*: pick a domain that is as different as possible from the original.
2. *Problem*: describe a task in the new domain that is solvable by the **same procedure**.
3. *Solution*: step-by-step reasoning then solution of the problem.
4. *Answer*: restate the final numerical answer for your analogue.

- begin of example -
### Question
Below is a python function that adds two integers and return the result.
```python
def add(a, b):
    return a + b
```
Suppose your inputs are as follows:
```python
a = 1
b = 2
```
Think step by step. Then, encode the output of the function in <answer></answer> (e.g. <answer>1</answer>).

### Analogue (generate *1* analogue of the task and solve it)

### Answer to the original question
Answer the original question.

### Analogue
Generate **1** analogue task in a different domain, and then solve it.
**Analogue**
Domain: Shopping
Problem: Alice buys 1 apple and 2 bananas. How many fruits does she have?
Solution:
1. Count the number of apples: 1
2. Count the number of bananas: 2
3. Add the counts: 1 + 2 = 3
4. Answer: 3 fruits
Answer: <answer>3</answer>

### Answer to the original question
1. Replace a with 1
2. Replace b with 2
3. Add the numbers: 1 + 2 = 3
4. Return the result: 3
Answer: <answer>3</answer>
- end of example -

Now please do the same:
### Question
[QUESTION]

### Analogue (generate *1* analogue of the task and solve it)

### Answer to the original question
Answer the original question.'''
    # dreamer_template = '''When you are awake, I give you a question. Then, you fall asleep and start to have wild dreams. You imagine that you have an alternative task that asks for the same procedure and produces the same answer, then solve the question you dream for yourself. In this new naturalistic task, there are NO keywords (e.g. DAG) in your rephrasing or any algorithmic hints. The rephrased task is extremely different in format from the original question, but you know the answer is the same as for the question that I give you. Now you are awake, I give you the question.

    # [QUESTION]
    
    # Now, you fall asleep, and start to dream about then solve the question. Keep the required answer format when you answer. Remember, you are not awake until you finish dreaming about the alternative task.'''

    if filled_idxes is None:
        filled_idxes = []
    results_path = os.path.join(out_dir, 'output_likelihood.json')
    rejection_path = os.path.join(out_dir, 'rejection_sampling.json')
    unfilled_idxes = get_unfilled_indices(len(prompts), filled_idxes)
    if not initialization:
        assert max_iter == 1, "max_iter should be 1 when initialization is False"
    for i in range(max_iter):
        if not generated_outputs and os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                generated_outputs = data.get('generated_outputs')
            eval_res, acc = eval_generation(generated_outputs, labels, task=task, return_samples=return_samples)
            if not os.path.exists(rejection_path):
                selected_samples, filled_idxes = rejection_sampler(generated_outputs, eval_res, n_per_prompt=n_per_prompt)
                with open(rejection_path, "w") as f:
                    json.dump({"selected_samples": selected_samples, "filled_idxes": filled_idxes}, f)
            else:
                with open(rejection_path, 'r') as f:
                    rejection = json.load(f)
                    selected_samples = rejection.get('selected_samples', filled_idxes)
                    filled_idxes = rejection.get('filled_idxes', filled_idxes)
            if not filled_idxes:
                filled_idxes = [_ for _ in range(len(prompts)) if len(selected_samples[_])>=n_per_prompt]
            unfilled_idxes = get_unfilled_indices(len(prompts), filled_idxes)
        elif not initialization:
            par_dir = '/'.join(str(out_dir).split('/')[:-1])
            model_name = str(out_dir).split('/')[-1]
            if '_0' in model_name:
                if '1.5' in model_name:
                    last_path = 'Qwen2.5-1.5B-Instruct'
                elif '3' in model_name:
                    last_path = 'Qwen2.5-3B-Instruct'
                elif '7' in model_name:
                    last_path = 'Qwen2.5-7B-Instruct'
                else:
                    raise ValueError(f"Unknown model name: {model_name}")
            else:
                iter_num = int(model_name.split('_')[-1])
                last_path = '_'.join(model_name.split('_')[:-1])+f"_{iter_num-1}"
            last_dir = os.path.join(par_dir, last_path)
            prev_res_path = os.path.join(last_dir, 'output_likelihood.json')
            with open(prev_res_path, 'r') as f:
                data = json.load(f)
                generated_outputs = data.get('generated_outputs')
            prev_rejection_path = os.path.join(last_dir, 'rejection_sampling.json')
            with open(prev_rejection_path, 'r') as f:
                rejection = json.load(f)
                selected_samples = rejection.get('selected_samples', filled_idxes)
                filled_idxes = rejection.get('filled_idxes', filled_idxes)
            print(f"Length of filled_idxes: {len(filled_idxes)}")
            unfilled_idxes = get_unfilled_indices(len(prompts), filled_idxes)
        if not unfilled_idxes:
            print("All prompts have acceptable outputs.")
            return
        print(f"Iteration {i+1}/{max_iter} started.")
        print(f"Current number of prompts for generation is {len(unfilled_idxes)} / {len(prompts)}")
        # Optionally increase the number of samples after a given iteration.
        if i + 1 >= iter_increase_n_samples:
            sampling_params['n'] = min(sampling_params['n'] * 2, 64)  # Cap 'n' at 64
        # Randomly sample few-shot examples from already filled prompts.
        random.seed(42)
        if filled_idxes:
            few_shot_idxes = random.sample(filled_idxes, min(nshot, len(filled_idxes)))
        else:
            few_shot_idxes = []
        if few_shot_idxes:
            if not dreamer:
                few_shot_prompt = '\n\n'.join([f"{prompts[idx]}\n{generated_outputs[idx][0]}" for idx in few_shot_idxes]) + '\n\n'
                # few_shot_prompt = '\n\n'.join([f"{prompts[idx]}\n{selected_samples[idx][0]}" for idx in few_shot_idxes]) + '\n\n'
                # Build new prompts for unfilled indices.
                new_prompts = [few_shot_prompt + prompts[idx] for idx in unfilled_idxes]
            else:
                few_shot_prompt = '\n\n'.join([dreamer_template.replace('[QUESTION]', prompts[idx])+f"\n{generated_outputs[idx][0]}" for ii, idx in enumerate(few_shot_idxes)]) + '\n\n'
                # few_shot_prompt = '\n\n'.join([dreamer_template.replace('[QUESTION]', prompts[idx])+f"\n{selected_samples[idx][0]}" for ii, idx in enumerate(few_shot_idxes)]) + '\n\n'
                # Build new prompts for unfilled indices.
                new_prompts = [few_shot_prompt + dreamer_template.replace('[QUESTION]', prompts[idx]) for idx in unfilled_idxes]
            
            # Generate new outputs and update filled_idxes.
            generated_outputs, filled_idxes = gen_and_reject(model,
                                                            new_prompts,
                                                            sampling_params,
                                                            out_dir,
                                                            labels,
                                                            task,
                                                            all_prompt_count=len(prompts),
                                                            chat_template=chat_template,
                                                            record_likelihoods=record_likelihoods,
                                                            return_samples=return_samples,
                                                            n_per_prompt=n_per_prompt,
                                                            prev_generated_outputs=generated_outputs,
                                                            filled_idxes=filled_idxes,
                                                            overwrite=overwrite)
            unfilled_idxes = get_unfilled_indices(len(prompts), filled_idxes)
            if rationalization:
                print('Start rationalization for unfilled prompts')
                print(f"{len(unfilled_idxes)} / {len(prompts)} prompts unfilled.")
                if 'timedelta' in str(labels[0]):
                    # read timedelta from ground truth
                    ground_truths = [str_to_timedelta_list(labels[idx])[0] for idx in few_shot_idxes]
                    # convert timedelta into string time description
                    time_strs = [timedelta_to_description(td) for td in ground_truths]
                else:
                    time_strs = [labels[idx] for idx in few_shot_idxes]
                
                if not dreamer:
                    few_shot_prompt = '\n\n'.join([f"{prompts[idx]} (hint: the correct final answer is <answer>{time_strs[ii]}</answer>)\n{generated_outputs[idx][0]}" for ii, idx in enumerate(few_shot_idxes)]) + '\n\n'
                    # Build new prompts for unfilled indices.
                    new_prompts = [few_shot_prompt + prompts[idx] for idx in unfilled_idxes]
                else:
                    few_shot_prompt = '\n\n'.join([dreamer_template.replace('[QUESTION]', f"{prompts[idx]} (hint: the correct final answer is <answer>{time_strs[ii]}</answer>)")+f"\n{generated_outputs[idx][0]}" for ii, idx in enumerate(few_shot_idxes)]) + '\n\n'
                    # Build new prompts for unfilled indices.
                    new_prompts = [few_shot_prompt + dreamer_template.replace('[QUESTION]', prompts[idx]) for idx in unfilled_idxes]
                # generate new outputs and update filled_idxes
                generated_outputs, filled_idxes = gen_and_reject(model,
                                                                new_prompts,
                                                                sampling_params,
                                                                out_dir,
                                                                labels,
                                                                task,
                                                                all_prompt_count=len(prompts),
                                                                chat_template=chat_template,
                                                                record_likelihoods=record_likelihoods,
                                                                return_samples=return_samples,
                                                                n_per_prompt=n_per_prompt,
                                                                prev_generated_outputs=generated_outputs,
                                                                filled_idxes=filled_idxes,
                                                                overwrite=overwrite)
                unfilled_idxes = get_unfilled_indices(len(prompts), filled_idxes)
        else:
            few_shot_prompt = ''
            if not dreamer:
                new_prompts = prompts
            else:
                new_prompts = [dreamer_template.replace('[QUESTION]', prompts[idx]) for idx in unfilled_idxes]
        
            # Generate new outputs and update filled_idxes.
            generated_outputs, filled_idxes = gen_and_reject(model,
                                                            new_prompts,
                                                            sampling_params,
                                                            out_dir,
                                                            labels,
                                                            task,
                                                            all_prompt_count=len(prompts),
                                                            chat_template=chat_template,
                                                            record_likelihoods=record_likelihoods,
                                                            return_samples=return_samples,
                                                            n_per_prompt=n_per_prompt,
                                                            prev_generated_outputs=generated_outputs,
                                                            filled_idxes=filled_idxes,
                                                            overwrite=overwrite)
            unfilled_idxes = get_unfilled_indices(len(prompts), filled_idxes)
        
        print(f"Iteration {i+1}/{max_iter} completed.")
        print(f"{len(filled_idxes)} / {len(prompts)} prompts filled.")

    return generated_outputs, filled_idxes


# @register_model("vllm")
class VLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        max_num_seqs: int = 256,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert max_length is None or max_model_len is None, (
            "Either max_length or max_model_len may be provided, but not both"
        )

        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "max_num_seqs": int(max_num_seqs),
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
        }
        self.model_args.update(kwargs)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["distributed_executor_backend"] = "ray"
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )
        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            revision=tokenizer_revision,
        )
        self.tokenizer = configure_pad_token(self.tokenizer)
        self.add_bos_token = add_bos_token
        if "gemma" in pretrained.lower():
            self.add_bos_token = True
            eval_logger.info(
                "Found 'gemma' in model name, a BOS token will be used as Gemma series models underperform without it."
            )

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version("0.3.0"), (
                "lora adapters only compatible with vllm > v0.3.0."
            )
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding: Union[List[List[int]], List[int]] = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]
        return encoding

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = ["<|im_end|>", "\r", "\r\n", "</answer>", "\u4e09\u5927\u53d1\u5feb"],
        **kwargs,
    ):

        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if self.data_parallel_size > 1:

            @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
            def run_inference_one_model(
                model_args: dict,
                sampling_params,
                requests: List[List[int]],
                lora_request: LoRARequest,
            ):
                llm = LLM(**model_args)
                print('Start generating')
                return llm.generate(
                    prompt_token_ids=requests,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                    use_tqdm=True if self.batch_size == "auto" else False,
                )

            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = (
                (self.model_args, sampling_params, req, self.lora_request)
                for req in requests
            )
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            ray.shutdown()
            return undistribute(results)

        outputs = self.model.generate(
            prompts=requests,
            # prompt_token_ids=requests,
            sampling_params=sampling_params,
            use_tqdm=True if self.batch_size == "auto" else False,
            lora_request=self.lora_request,
        )
        return outputs

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        adaptive_batch_size = None
        if self.batch_size == "auto":
            adaptive_batch_size = len(requests)

        all_windows = []  
        request_window_counts = []  

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )
            windows = [(None,) + x for x in rolling_token_windows]
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls = []
        batch_size = adaptive_batch_size or int(self.batch_size)
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            batch_indices, batch_windows = zip(*batch)
            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
            )
            all_nlls.extend(zip(batch_indices, batch_nlls))

        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            return -len(_requests[0][1]), _requests[0][0]

        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        eos = self.tokenizer.decode(self.eot_token_id)
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            gen_kwargs = all_gen_kwargs[0]
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            cont = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )

            for output, context in zip(cont, context):
                generated_text = output.outputs[0].text
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )
                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )
                res.append(answer)

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return outputs, re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            if logprob_dict:
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            eval_logger.debug(
                "Got `do_sample=False` and no temperature value, setting VLLM temperature to 0.0 ..."
            )
            kwargs["temperature"] = 0.0
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        return kwargs

    def generate_with_loglikelihood(
        self, prompts: List[str], gen_kwargs: Optional[dict] = None, disable_tqdm: bool = False,
        sampling_params: Optional[dict] = None, chat_template=True, record_likelihood: bool=False
    ) -> List[Tuple[str, float]]:
        if gen_kwargs is None:
            gen_kwargs = {}
        
        if chat_template:
            inputs = [self.apply_chat_template([{'role':'user','content':prompt}]) for prompt in prompts]
        else:
            inputs = prompts

        context_encodings = [self.tok_encode(prompt, add_special_tokens=self.add_bos_token) for prompt in inputs]

        temperature, n = sampling_params['temperature'], sampling_params['n']

        outputs = self._model_generate(
            requests=inputs,
            generate=True,
            max_tokens=self.max_gen_toks,
            temperature=temperature,
            n=n,
            **gen_kwargs,
        )
        generated_texts = [[output.outputs[i].text for i in range(len(output.outputs))] for output in outputs]
        for i, texts in enumerate(generated_texts):
            for ii, text in enumerate(texts):
                if "<answer>" in text and "</answer>" not in text:
                    generated_texts[i][ii] = generated_texts[i][ii]+"</answer>"

        if record_likelihood:

            ll_requests = []
            for prompt, gen_texts in zip(prompts, generated_texts):
                # ll_requests.append([])
                # loop over all generated texts
                for gen_text in gen_texts:
                    prompt_tokens = self.tok_encode(prompt, add_special_tokens=self.add_bos_token)
                    full_text = prompt + gen_text
                    full_tokens = self.tok_encode(full_text, add_special_tokens=self.add_bos_token)
                    continuation_tokens = full_tokens[len(prompt_tokens):]
                    ll_requests.append(((prompt, gen_text), prompt_tokens, continuation_tokens))

            ll_results = self._loglikelihood_tokens(ll_requests, disable_tqdm=disable_tqdm)
            # reorganize by instance to match the shape of generated texts

            return generated_texts, ll_results
        else:
            return generated_texts


def timedelta_to_description(td):
    """Convert timedelta to human-readable time description."""
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
    
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="VLLM Model Inference")
    parser.add_argument("--pretrained", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Pretrained model identifier.")
    parser.add_argument("--input_path", type=str, default="../data/expla_graph_train_sft.json",
    help="Input file path")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type to use (e.g., float16, bfloat16, float32, auto).")
    parser.add_argument("--revision", type=str, default=None, help="Model revision if needed.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--n", type=int, default=32, help="Number of best completions to return.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for custom models.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer identifier (defaults to pretrained model's tokenizer).")
    parser.add_argument("--tokenizer_mode", type=str, default="auto", help="Tokenizer mode.")
    parser.add_argument("--tokenizer_revision", type=str, default=None, help="Tokenizer revision if needed.")
    parser.add_argument("--add_bos_token", action="store_true", help="Whether to add beginning-of-sequence token.")
    parser.add_argument("--prefix_token_id", type=int, default=None, help="Custom prefix token id.")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Tensor parallel size.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization scheme, if any.")
    parser.add_argument("--max_gen_toks", type=int, default=6000, help="Maximum tokens to generate.")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space value for the model.")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size (or 'auto').")
    parser.add_argument("--max_model_len", type=int, default=16384, help="Maximum model sequence length.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generation.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory to utilize.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (e.g., cuda).")
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Data parallelism size.")
    parser.add_argument("--lora_local_path", type=str, default=None, help="Path to LoRA adapters if using them.")
    parser.add_argument("--save_path", type=str, default="../saves/rejection_sampling/sample_32", help="Path to store log probs for generation")
    parser.add_argument("--chat_template", type=bool, default=True, help="Apply chat template to the input")
    parser.add_argument("--enforce_eager", type=bool, default=True, help="Enforce eager execution")
    parser.add_argument("--max_num_seqs", type=int, default=128, help="Maximum number of sequences to generate")
    parser.add_argument("--record_likelihoods", type=bool, default=False, help="Record likelihoods for each generation")
    parser.add_argument("--task", type=str, default='asynchow', help="Task to perform")
    parser.add_argument("--rejection_sampling", type=bool, default=True, help="Whether to perform rejection sampling")
    parser.add_argument("--overwrite", type=bool, default=True, help="Whether to overwrite existing results")
    parser.add_argument("--max_iter", type=int, default=5, help="Maximum number of iterations for rejection sampling")
    parser.add_argument("--iter_increase_n_samples", type=int, default=5, help="Iteration number to increase after each iteration")
    parser.add_argument("--rationalization", action='store_true', help="Whether to use rationalization")
    parser.add_argument("--initialization", action='store_true', help="Whether to use initialization")
    parser.add_argument("--dreamer", action='store_true', help="Whether to use dreamer")
    args = parser.parse_args()
    print(args)
    if args.input_path.endswith('.json'):
        with open(args.input_path, 'r') as f:
            input_file = json.load(f)
        prompts = [sample['instruction'] for sample in input_file]
        labels = [sample['output'] for sample in input_file]
    else:
        assert 'asynchow' in args.task
        input_file = load_dataset('json', data_files=args.input_path+'_train.jsonl')
        prompts = [sample['question'] for sample in input_file]
        labels = [sample['answer'] for sample in input_file]

    out_dir = Path(os.path.join(args.save_path, args.pretrained.split("/")[-1]))
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print('Created a directory', out_dir)

    model = VLLM(
        pretrained=args.pretrained,
        dtype=args.dtype,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        tokenizer_revision=args.tokenizer_revision,
        add_bos_token=args.add_bos_token,
        prefix_token_id=args.prefix_token_id,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        max_gen_toks=args.max_gen_toks,
        swap_space=args.swap_space,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        device=args.device,
        data_parallel_size=args.data_parallel_size,
        lora_local_path=args.lora_local_path,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs
    )
    sampling_params = {'n': args.n,
    'temperature': args.temperature}
    if args.record_likelihoods:
        generated_outputs, likelihoods = model.generate_with_loglikelihood(prompts, sampling_params=sampling_params, chat_template=args.chat_template, record_likelihood=args.record_likelihoods)
        outputs, re_ord = likelihoods
        # save the results
        print(len(re_ord))
        log_probs = []
        for i in range(len(generated_outputs)): 
            log_probs.append([o[0] for o in re_ord[i*len(generated_outputs[0]):(i+1)*len(generated_outputs[0])]])
        
        with open(os.path.join(out_dir, 'all_logs.json'), 'w') as f:
            json.dump({'re_ord':re_ord, 'prompts':prompts, 'generated_outputs':generated_outputs, 'log_probs':log_probs, "logs":str(outputs)}, f)
        with open(os.path.join(out_dir, 'output_likelihood.json'), 'w') as f:
            json.dump({'prompts':prompts, 'generated_outputs':generated_outputs, 'log_probs':log_probs}, f)
    else:
        assert args.rejection_sampling
        # rejection sampling with iteration
        regenerate_with_few_shot(model,
                                prompts,
                                sampling_params,
                                out_dir,
                                labels,
                                task=args.task,
                                chat_template=args.chat_template,
                                record_likelihoods=args.record_likelihoods,
                                max_iter=args.max_iter,
                                overwrite=args.overwrite,
                                iter_increase_n_samples=args.iter_increase_n_samples,
                                rationalization=args.rationalization,
                                initialization=args.initialization,
                                dreamer=args.dreamer)
        # # load results if there exist path
        # results_path = os.path.join(out_dir, 'output_likelihood.json')
        # # skip if file exists
        # if os.path.exists(results_path):
        #     with open(results_path, 'r') as f:
        #         data = json.load(f)
        #         generated_outputs = data['generated_outputs']
        # else:
        #     generated_outputs = model.generate_with_loglikelihood(prompts,
        #                                                           sampling_params=sampling_params,
        #                                                           chat_template=args.chat_template, 
        #                                                           record_likelihood=args.record_likelihoods)
        #     # save generated outputs
        #     with open(results_path, 'w') as f:
        #         json.dump({'prompts':prompts, 'generated_outputs':generated_outputs}, f)

        # eval_res, acc = eval_generation(generated_outputs, labels, task=args.task, return_samples=True)
        # # perform rejection sampling
        # selected_samples, unfilled_idxes = rejection_sampler(generated_outputs, eval_res,
        # prompts=prompts)
        # # save results
        # with open(os.path.join(out_dir, 'rejection_sampling.json'), "w") as f:
        #     json.dump({"selected_samples": selected_samples, "unfilled_idxes": unfilled_idxes}, f)


if __name__ == "__main__":
    main()