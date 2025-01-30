devices=0,1,2,3
n_gpu=4
dataset_name='hendrydong/preference_700K'
base_model='google/gemma-2b-it'
wandb_name="GRM"
log_dir='../save_reward_models'
main_process_port=9994

learning_rate=5e-6
max_length=3000 
num_train_epochs=1
gradient_accumulation_steps=64

# GRM parameters
weight_ratio=0.01
layer_type='mlp'
sft_only=True
reference_free=True

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_grm_reward_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --use_lora False \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --weight_ratio ${weight_ratio}  --layer_type ${layer_type} \
    --reference_free ${reference_free} --sft_only ${sft_only}