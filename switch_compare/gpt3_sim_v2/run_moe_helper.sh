dir_path=$1
log_path="${dir_path}/logs"
mkdir -p $log_path
card_arr=(1024 4096 10752 86016)
for(( i=0;i<${#card_arr[@]};i++)) do
    cards=${card_arr[i]};
    # python compute_llm_train_v3.py -i ${dir_path}/inter_gpt3_train_$cards.xlsx -o ${log_path}/res_$cards.xlsx --is_vis --tp_no_overlap 0.2
    python compute_llm_train_moe.py -i ${dir_path}/inter_gpt3_train_$cards.xlsx -o ${log_path}/res_$cards.xlsx --is_vis --tp_no_overlap 1.0 
done
