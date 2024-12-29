devices=0,1,2,3
n_gpu=4
main_process_port=9994

cd ../../../

# Replace the model_type and data_path
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step5_obtain_bon_gold_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "rlhf/bon/step4_choose_best_of_n/gemma-2b-it/grm/bon_selected_proxy_grm_drop_duplicates" \
    --method "grm" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/bon/step5_obtain_bon_gold_score/gemma-2b-it" \


