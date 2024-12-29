devices=0,1,2,3
n_gpu=4
main_process_port=9994


cd ../../../
# Fill the peft path

# For GRM
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "rlhf/bon/step2_generate_samples/generated_samples_unified" \
    --model_type "grm" \
    --base_model "google/gemma-2b-it" \
    --peft_name "rlhf/bon/save_reward_models/.." \
    --save_path "rlhf/bon/step3_obtain_proxy_score/gemma-2b-it" \
    --layer_type "linear" \
    --num_layers 1 \

# For baselines
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "rlhf/bon/step2_generate_samples/generated_samples_unified" \
    --model_type "bt" \
    --base_model "google/gemma-2b-it" \
    --peft_name "rlhf/bon/save_reward_models/.." \
    --save_path "rlhf/bon/step3_obtain_proxy_score/gemma-2b-it" \



    

