#!/bin/bash
OUTFOLDER=$1
TYPE_RUN=$2
MAX_ITER=$3
echo $1
if [[ ${TYPE_RUN} == "auc" ]]; then
    QUANTITIES=("r_cost" "cost" "auc" "r_auc" "f_auc" "r_f_auc" "l2" "c")
else
    QUANTITIES=("r_cost" "cost" "auc" "r_auc" "l2" "mean_c" "var_c" \
	"dFPR_m" "dFPR_v" "dFNR_m" "dFNR_v")
fi 

# 1. Generates the outfolders

if [[ ${MAX_ITER} =~ "[0-9]+" ]]; then # Max iter
    DYN_ANA="dyn_ana_"${MAX_ITER}
else
    DYN_ANA="dyn_analysis"
fi

mkdir ${OUTFOLDER}"/"${DYN_ANA}
mkdir ${OUTFOLDER}"/"${DYN_ANA}"/files" 
mkdir ${OUTFOLDER}"/"${DYN_ANA}"/plots" 
mkdir ${OUTFOLDER}"/"${DYN_ANA}"/plots_fancy" 

# 2. Generates the data - a) regular data

cp ${OUTFOLDER}"/learning_log.log" ${OUTFOLDER}"/tmp_learning_log.log"

{
    n_last_line=$(awk '{print NR" "$0}' ${OUTFOLDER}"/tmp_learning_log.log" |
	grep "^[0-9]* Iter" | tail -n 1 | awk '{print $1}')
    echo ${QUANTITIES[@]} | sed "s/ /,/g"
    head -n $n_last_line ${OUTFOLDER}"/tmp_learning_log.log" | grep "^te |" \
	| awk 'BEGIN{FS=" *[|:] "}{for (i=2; i<NF; i++){printf $i","} printf $NF"\n"}';
} > ${OUTFOLDER}"/"${DYN_ANA}"/files/data.csv" &
wait

# 2. Generates the data - b) iteration #

if [[ ${MAX_ITER} =~ "[0-9]+" ]]; then # Max iter
    DYN_ANA="dyn_ana_"${MAX_ITER}
    grep "^Iter"  ${OUTFOLDER}"/tmp_learning_log.log" | \
        awk '{printf $2"\n"}' | head -n $((${MAX_ITER}/100)) > ${OUTFOLDER}"/"${DYN_ANA}"/files/iter.txt"
else
    DYN_ANA="dyn_analysis"
    grep "^Iter"  ${OUTFOLDER}"/tmp_learning_log.log" | \
        awk '{printf $2"\n"}' > ${OUTFOLDER}"/"${DYN_ANA}"/files/iter.txt"
fi
wait
sleep 0.5

# 3. Plots the data

for val in ${QUANTITIES[@]}
do
    # echo python monitoring_plot_raw.py ${OUTFOLDER} ${val} ${DYN_ANA}
    python monitoring_plot_raw.py ${OUTFOLDER} ${val} ${DYN_ANA} &
done
wait

# 4. Generates special plots for the pointwise experiments

if [[ $TYPE_RUN = "ptw" ]]; then 
    grep "^c: [+-][\.0-9]*" ${OUTFOLDER}"/tmp_learning_log.log" \
	| awk 'BEGIN{FS=" *[|:] "}{for(i=2; i<NF; i++){printf $i","} printf $NF"\n"}' \
	> ${OUTFOLDER}"/"${DYN_ANA}"/files/c.csv"

    grep "^FPRs" ${OUTFOLDER}"/tmp_learning_log.log" \
	| awk 'BEGIN{FS=" *[|:] "}{for(i=2; i<NF; i++){printf $i","} printf $NF"\n"}' \
	> ${OUTFOLDER}"/"${DYN_ANA}"/files/fpr.csv"

    grep "^TPRs" ${OUTFOLDER}"/tmp_learning_log.log" \
	| awk 'BEGIN{FS=" *[|:] "}{for(i=2; i<NF; i++){printf $i","} printf $NF"\n"}' \
	> ${OUTFOLDER}"/"${DYN_ANA}"/files/tpr.csv"

    grep "^biases" ${OUTFOLDER}"/tmp_learning_log.log" \
	| awk 'BEGIN{FS=" *[|:] "}{for(i=2; i<NF; i++){printf $i","} printf $NF"\n"}' \
	> ${OUTFOLDER}"/"${DYN_ANA}"/files/biases.csv"

    for val in "c" "fpr" "tpr" "biases"
    do 
        # echo python monitoring_plot_raw.py ${OUTFOLDER} ${val} ${DYN_ANA} ${val}
        python monitoring_plot_raw.py ${OUTFOLDER} ${val} ${DYN_ANA} ${val} # &
    done
    wait
fi

# Plot the fancy version of the plots:

if ! [[ ${MAX_ITER} =~ "[0-9]+" ]]; then # Max iter
    if [[ $TYPE_RUN = "ptw" ]]; then
        python monitoring_plot_ptw.py ${OUTFOLDER}
    else
        python monitoring_plot_auc.py ${OUTFOLDER}
    fi
fi
