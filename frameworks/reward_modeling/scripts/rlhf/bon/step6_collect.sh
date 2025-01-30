cd ../../../

# Replace the score_path
python rlhf/bon/step6_collect.py \
    --proxy_score_path 'rlhf/bon/step4_choose_best_of_n/gemma-2b-it/grm/bon_selected_proxy_grm.csv' \
    --gold_score_path 'rlhf/bon/step5_obtain_bon_gold_score/gemma-2b-it/grm/gold_score.csv' \
    --output_path 'rlhf/bon/step6_collect/gemma-2b-it/grm' \
    --n_values_start 1 \
    --n_values_end 406 
   
