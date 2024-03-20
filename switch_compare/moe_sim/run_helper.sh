dir_path=$1
card_arr=(1024 4096 10752 86016)
for(( i=0;i<${#card_arr[@]};i++)) do
    cards=${card_arr[i]};
    # python compute_moe_sim_tp_opt.py -i ${dir_path}/gpt-8x-60B_$cards.xlsx -o ${dir_path}/res_$cards.xlsx --is_vis
    python compute_moe_sim_tp_opt.py -i ${dir_path}/gpt-8x-60B_$cards.xlsx -o ${dir_path}/res_$cards.xlsx --is_vis
done
