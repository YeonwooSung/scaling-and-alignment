gpu=4,5,6,7
base_model_name="google/gemma-2b-it"
reward_base_model="google/gemma-2b-it"
dataset_path="rlhf/data/unified_20k" # set the train dataset path, refer to the BoN experiments
eval_dataset_path="rlhf/data/unified_1k" # set the eval dataset
init_kl_coef=0.0
log_dir='rlhf/log_ppo_noise'
eval_every=4

cd ../../../

# you need set the path
reward_peft_path1='save_reward_models/gemma-2b-it_BT_RM_seed0_len1024_lora32_1e-05_dataUnified-Feedback/logs/checkpoint-663'
reward_peft_path2='save_reward_models/gemma-2b-it_BT_RM_seed1_len1024_lora32_1e-05_dataUnified-Feedback/logs/checkpoint-1326'
reward_peft_path3='save_reward_models/gemma-2b-it_BT_RM_seed2_len1024_lora32_1e-05_dataUnified-Feedback/logs/checkpoint-3536'

ensemble_method='avg'
wandb_name="ppo_ensemble_avg_lr1e-5_kl0.0_tau0.7_len512_normrewards"
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9991 rlhf/ppo/ppo_rm_ensemble.py \
    --base_model_name ${base_model_name} \
    --reward_base_model ${reward_base_model} \
    --reward_peft_path ${reward_peft_path1},${reward_peft_path2},${reward_peft_path3} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --eval_every ${eval_every} \
    --ensemble_method ${ensemble_method} \
    --normalize_rewards True \
    --learning_rate 1e-5 



ensemble_method='min'
wandb_name="gemma-2b-it_ppo_ensemble_min_lr1e-5_kl0.0_tau0.7_len512_normrewards_noadapkl"
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9998 rlhf/ppo/ppo_rdm_ensemble.py \
    --base_model_name ${base_model_name} \
    --reward_peft_path ${reward_peft_path1},${reward_peft_path2},${reward_peft_path3} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --eval_every ${eval_every} \
    --ensemble_method ${ensemble_method} \
    --learning_rate 1e-5 \




   



