# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from collections import Counter
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]
    
def apply_ttrl_gt(batch, gen_batch_output, n_votes_per_prompt, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n_votes_per_prompt == 0, "gen_batch_output length must be divisible by n_votes_per_prompt"
    num_prompts = len(gen_batch_output) // n_votes_per_prompt
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        for j in range(n_votes_per_prompt):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    ttrl_gt, majority_vote_ratio = _ttrl_maj_vote_batch(model_outputs, n_votes_per_prompt)
    
    assert len(batch) == len(ttrl_gt), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = ttrl_gt[i]
        data_item.non_tensor_batch["reward_model"]["ttrl_gt"] = ttrl_gt[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_vote_ratio"] = majority_vote_ratio
    return batch

def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch

def compute_ttrl_metrics(batch, n_samples_per_prompt):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n_samples_per_prompt == 0, "batch length must be divisible by n_samples_per_prompt"
    num_prompts = len(batch) // n_samples_per_prompt

    # Sort the batch by the ID
    sorted_batch = sorted(batch, key=lambda x: x.non_tensor_batch["extra_info"]["index"])

    ttrl_reward = []
    gt_reward = []
    ttrl_label = []
    gt_label = []

    for i in range(len(sorted_batch)):
        data_item = sorted_batch[i]
        ttrl_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        ttrl_label.append(data_item.non_tensor_batch["reward_model"]["ttrl_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _compute_ttrl_metrics_batch(ttrl_reward, gt_reward, ttrl_label, gt_label, n_samples_per_prompt=n_samples_per_prompt)
    majority_vote_ratio = batch.non_tensor_batch["majority_vote_ratio"]
    majority_vote_ratio = sum(majority_vote_ratio) / len(majority_vote_ratio)
    ttrl_metrics["majority_ratio"] = majority_vote_ratio

    return ttrl_metrics


def _ttrl_maj_vote_batch(model_outputs: List[str], n_votes_per_prompt: int) -> List[str]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n_votes_per_prompt: int
    Output: 
        maj_vote_gt: list of str
    """
    maj_vote_gt = []
    maj_vote_ratio = []
    assert len(model_outputs) % n_votes_per_prompt == 0
    n_prompts = len(model_outputs) // n_votes_per_prompt
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n_votes_per_prompt:(i + 1) * n_votes_per_prompt]
        prompt_gt, prompt_maj_vote_ratio = _ttrl_maj_vote_prompt(prompt_outputs)
        maj_vote_gt.append(prompt_gt)
        maj_vote_ratio.append(prompt_maj_vote_ratio)
        
    return maj_vote_gt, maj_vote_ratio


def _compute_ttrl_metrics_batch(
    ttrl_reward: List[float],
    gt_reward: List[float],
    ttrl_label: List[str],
    gt_label: List[str],
    n_samples_per_prompt: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(ttrl_reward) == len(gt_reward) == len(ttrl_label) == len(gt_label)
    assert len(ttrl_reward) % n_samples_per_prompt == 0
    n_prompts = len(ttrl_reward) // n_samples_per_prompt
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_ttrl_reward = ttrl_reward[i * n_samples_per_prompt:(i + 1) * n_samples_per_prompt]
        prompt_gt_reward = gt_reward[i * n_samples_per_prompt:(i + 1) * n_samples_per_prompt]
        prompt_ttrl_label = ttrl_label[i * n_samples_per_prompt:(i + 1) * n_samples_per_prompt]
        prompt_gt_label = gt_label[i * n_samples_per_prompt:(i + 1) * n_samples_per_prompt]

        assert Counter(prompt_ttrl_label).most_common(1)[0][1] == n_samples_per_prompt
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n_samples_per_prompt

        prompt_ttrl_label = prompt_ttrl_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _compute_ttrl_metrics_prompt(prompt_ttrl_reward, prompt_gt_reward, prompt_ttrl_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _ttrl_maj_vote_prompt(model_outputs: List[str]) -> str:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return "None"
    
    counter = Counter(model_answers)
    
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)
    
    return majority_answer, majority_ratio

def _compute_ttrl_metrics_prompt(
    ttrl_reward: List[float],
    gt_reward: List[float],
    ttrl_label: str,
    gt_label: str,
    ):    
    assert len(ttrl_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(ttrl_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(ttrl_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(ttrl_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(ttrl_reward) / len(ttrl_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(ttrl_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric