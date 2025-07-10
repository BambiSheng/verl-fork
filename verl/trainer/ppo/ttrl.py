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

from verl.utils.reward_score.ttrl_math import ttrl_maj_vote_fn, ttrl_metrics_fn

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

    ttrl_gt, majority_vote_ratio = ttrl_maj_vote_fn(model_outputs, n_votes_per_prompt)
    
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

    ttrl_metrics = ttrl_metrics_fn(ttrl_reward, gt_reward, ttrl_label, gt_label, n_samples_per_prompt=n_samples_per_prompt)
    majority_vote_ratio = batch.non_tensor_batch["majority_vote_ratio"]
    majority_vote_ratio = sum(majority_vote_ratio) / len(majority_vote_ratio)
    ttrl_metrics["majority_ratio"] = majority_vote_ratio

    return ttrl_metrics
