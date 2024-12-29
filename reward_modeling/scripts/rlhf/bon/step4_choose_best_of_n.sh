cd ../../../

# Replace the model_type and data_path
python rlhf/bon/step4_choose_best_of_n.py \
    --model_type grm \
    --data_path "rlhf/bon/step3_obtain_proxy_score/gemma-2b-it/grm" \
    --n_values_start 1 \
    --n_values_end 406 \
    --save_path "rlhf/bon/step4_choose_best_of_n/gemma-2b-it"
