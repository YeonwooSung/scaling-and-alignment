
log_dir='rlhf/logs_ppo'
init_kl_coef=0.00
base_model_name="google/gemma-2b-it" # policy base model
dataset_path="rlhf/data/unified_20k" # set the train dataset path, refer to the BoN experiments
eval_dataset_path="rlhf/data/unified_1k" # set the eval dataset


cd ../../../

# 4 gpus for 2b rm
gpu=0,1,2,3 
num_processes=4
reward_base_model="google/gemma-2b-it"
### you need set this path
reward_peft_path='rlhf/save_reward_models/gemma-2b-it_BT_RM_seed2_len1024_lora32_1e-05_dataUnified-Feedback/logs/checkpoint-3536' 
wandb_name="ppo_rm2B_lr1e-5_klreg0.0_normrewards"
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9989 --num_processes ${num_processes} rlhf/ppo/ppo.py \
    --base_model_name ${base_model_name} \
    --reward_base_model ${reward_base_model} \
    --reward_peft_path ${reward_peft_path} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --normalize_rewards True \
    --learning_rate 1e-5 \


# training 7B reward model requires 6 gpus and 4 process (other 2 gpus for reward inference)
gpu='1,2,3,4,5,6'
num_processes=4
reward_base_model="mistralai/Mistral-7B-Instruct-v0.2"
### just an example, you need set this path
reward_peft_path='rlhf/save_reward_models/Mistral-7B-Instruct-v0.2_20kunifed_label_smooth/logs/checkpoint-1666'
wandb_name="ppo_labelsmooth7B_lr1e-5_klreg0.0_normrewards"
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 9989 --num_processes ${num_processes} ppo.py \
    --base_model_name ${base_model_name} \
    --reward_base_model ${reward_base_model} \
    --reward_peft_path ${reward_peft_path} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --normalize_rewards True \
    --learning_rate 1e-5 \

   