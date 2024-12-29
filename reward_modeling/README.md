# Generalizable Reward Model for LLMs
Code for NeurIPS 2024 paper ["Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs"](https://arxiv.org/abs/2406.10216). Our open-sourced reward models are available at [🤗 huggingface](https://huggingface.co/collections/Ray2333/grm-66882bdf7152951779506c7b).


## Models

Check out our GRM series below, which are evlauated on [reward-bench](https://huggingface.co/spaces/allenai/reward-bench).



|       Model               | Average       |  Chat     |     Chat Hard      |     Safety      |     Reasoning     |   
|:-------------------------:|:-------------:|:---------:|:---------:|:--------:|:-----------:|
|[GRM_Llama3.1_8B_rewardmodel-ft](https://huggingface.co/Ray2333/GRM_Llama3.1_8B_rewardmodel-ft)**(8B)**| 92.6|95.0 |87.7|91.4|96.4| 
|[GRM-Llama3-8B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Llama3-8B-rewardmodel-ft)**(8B)**|91.5|95.5|86.2|90.8|93.6|
|[GRM-Llama3.2-3B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft)**(3B)**|90.9|91.6|84.9|92.7|94.6|
| [GRM-gemma2-2B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-gemma2-2B-rewardmodel-ft) **(2B)**| 88.4 | 93.0 | 77.2 | 92.2 | 91.2 |
| google/gemini-1.5-pro-0514 | 88.2 | 92.3 | 80.6 | 87.9 |92.0 |
|RLHFlow/pair-preference-model-LLaMA3-8B |87.1 | 98.3 | 65.8|89.7|94.7|
|[GRM-llama3-8B-sftreg](https://huggingface.co/Ray2333/GRM-llama3-8B-sftreg)**(8B)**|87.0|98.6|67.8|89.2|92.3|
|google/gemini-1.5-pro-0924 | 86.8 | 94.1|77.0|85.8 |90.2|
|[GRM-llama3.2-3B-sftreg](https://huggingface.co/Ray2333/GRM-llama3.2-3B-sftreg)**(3B)**|85.8|96.4|67.1|88.2|91.6|
|[GRM-Gemma-2B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Gemma-2B-rewardmodel-ft) **(2B)**|  84.7 | 89.4 | 75.2 | 85.5 | 88.8 |
|  [GRM-Gemma2-2B-sftreg](https://huggingface.co/Ray2333/GRM-Gemma2-2B-sftreg)**(2B)** | 81.0 |  97.2    |  59.6 | 86.9 |   80.3 |
| openai/gpt-4o-2024-05-13 | 84.6|	96.6	| 70.4	| 86.5	| 84.9 |
|  [GRM-Gemma-2B-sftreg](https://huggingface.co/Ray2333/GRM-Gemma-2B-sftreg)**(2B)** | 75.3    |   95.5  |  48.7 |   80.0 | 76.8     |  
|  [Gemma-2B-rewardmodel-baseline](https://huggingface.co/Ray2333/Gemma-2B-rewardmodel-baseline)**(2B)** | 73.7    |   94.1  |  46.1 |  79.6 |  75.0   |  



We also evaluated the GRM series using [PPE](https://github.com/lmarena/PPE/tree/main), which demonstrates better correlation with post-RLHF performance. In this benchmark, we found 'sftreg' reward models performs better than 'ft' models. So we suggest using 'sftreg' reward models for RLHF.

|       Model               | Average       |  MMLU-Pro    |   IFEval |    GPQA    |  MATH       |       MBPP-Plus   |   Human Preference |
|:-------------------------:|:-------------:|:---------:|:---------:|:--------:|:-----------:|   :-----------:  |:-----------:  |
| InternLM2-20B-Reward | 62.7 | 67.5  | 62.5 | 57.5 | 70.3 | 57.6 | 61.0 |
|[GRM-llama3-8B-sftreg](https://huggingface.co/Ray2333/GRM-llama3-8B-sftreg)**(8B)**| 62.7 | 66.6 | 60.4| 55.6| 70.9|   59.5    | 63.4|
|[GRM-Llama3-8B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Llama3-8B-rewardmodel-ft)**(8B)**| 61.4 | 64.2 | 59.6 | 56.2 | 72.3 | 53.3 | 62.5 |
|[GRM-llama3.2-3B-sftreg](https://huggingface.co/Ray2333/GRM-llama3.2-3B-sftreg)**(3B)**|  61.3  |63.9 |58.7   | 55.6| 74.7| 53.1 |  62.0     |
| ArmoRM-Llama3-8B-v0.1 | 61.2  | 66.5 | 58.4 | 57.0 | 70.7 | 54.2 | 60.6| 
|Skywork-Reward-Llama-3.1-8B | 61.0 |  64.3 | 61.5 | 56.5 | 69.7 | 51.6   | 62.4|
|Nemotron-4-340B-Reward | 60.4| 69.7 | 62.7 | 56.6 | 65.1 | 49.2 | 59.3 |
|[GRM-Llama3.2-3B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-Llama3.2-3B-rewardmodel-ft)**(3B)**| 59.2 | 62.2 | 57.4 | 56.1 | 72.4 | 46.2 | 60.8 |
| [GRM-gemma2-2B-rewardmodel-ft](https://huggingface.co/Ray2333/GRM-gemma2-2B-rewardmodel-ft) **(2B)**| 58.8 | 58.1 | 55.9 | 54.2 | 62.9 | 62.1 | 59.6 |
|  [GRM-Gemma2-2B-sftreg](https://huggingface.co/Ray2333/GRM-Gemma2-2B-sftreg)**(2B)** | 57.6  | 60.0 | 57.5 | 53.6 | 62.9 | 49.7 | 61.9 |



## Usage 
First set the environment variable.
```
export HF_HOME='your HF token'
```
Then install the environment. Note that we found error in transformers==4.46.1, please use early versions.
```
pip install -r requirements.txt
```

### Training and evaluation of reward models
Then, go to the `scripts' folder and train the reward model with the default hyperparameters
```
cd scripts
sh train_bt_rm_full.sh
sh train_bt_rm_lora.sh
sh train_grm_full.sh
sh train_grm_lora.sh
```

Evaluating trained models on 'llm-blender/Unified-Feedback', 'HuggingFaceH4/hhh_alignment', 'lmsys/mt_bench_human_judgments':
```
sh eval_bt_rm.sh
sh eval_grm_rm.sh
```

### RLHF

#### Data Generation

```
cd scripts/rlhf
sh data_generation4rlhf.sh
```

#### BoN
Go to the `scripts/rlhf/bon` folder and run the scripts. For more information about each step, please refer to `rlhf/bon/README.md` .

**Note: please set the path to your dataset and reward model in the corresponding shells.**
```
cd scripts/rlhf/bon
sh step1_train_proxy_reward_model_baseline.sh
sh step1_train_proxy_reward_model_grm.sh
sh step2_generate_samples.sh
sh step3_obtain_proxy_score.sh
sh step4_choose_best_of_n.sh
sh step5_obtain_bon_gold_score.sh
sh step6_collect.sh
```

#### PPO
Go to the `scripts/rlhf/ppo' folder and train the gemma-2b-it model with the default parameters.

**Note: please set the path to your reward model in the corresponding shells.**
```
cd scripts/rlhf/ppo
sh train_ppo.sh
sh train_ppo.grm.sh
sh train_ppo_ensemble.sh
```


## Citation

```
@inproceedings{yang2024regularizing,
  title={Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs},
  author={Yang, Rui and Ding, Ruomeng and Lin, Yong and Zhang, Huan and Zhang, Tong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgment
This repo is built upon [transformers](https://github.com/huggingface/transformers) and [trl](https://github.com/huggingface/trl), with also inspiration from [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling). 


