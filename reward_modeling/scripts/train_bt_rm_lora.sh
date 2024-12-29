devices=0,1,2,3,4,5
n_gpu=6
dataset_name='llm-blender/Unified-Feedback'
dataset_mode='40K' # 400K
base_model='google/gemma-2b-it'
wandb_name="BT_RM_seed2"
log_dir='../save_reward_models'
main_process_port=9994
loss_type='bt'

learning_rate=1e-5
lora_r=32
lora_alpha=64
max_length=1024
num_train_epochs=2
gradient_accumulation_steps=4


cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora True \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} --loss_type ${loss_type} \
    --dataset ${dataset_name} --dataset_mode ${dataset_mode} 