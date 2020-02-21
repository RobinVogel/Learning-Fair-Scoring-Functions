#!/bin/bash
crossval () {
    db_name=$1
    model=$2
    type_eval=$3
    for reg_no in {0..6}
    do
	for lambda_no in {0..5} 
	do
	    echo "***** ON "${db_name}" "${model}" reg_"${reg_no}" lambda_"${lambda_no}" *****"
	    w_f=results/${db_name}/${model}/reg_${reg_no}/lambda_${lambda_no}/
	    mkdir -p ${w_f}
	    if [[ ${type_eval} = "auc" ]]; then
		param_model="params/real_data/default.json"
	    else
		param_model="params/real_data/default_ptw.json"
	    fi
	    param_lambda="params/lambda/lambda_"${lambda_no}".json"
	    param_reg="params/reg/reg_"${reg_no}".json"
	    params="params/fixed.json "${param_model}" "${param_lambda}" "${param_reg}
	    if [[ ${db_name} = "german" ]]; then
		params=${params}" params/real_data/vsize_german.json"
	    fi
	    echo "########## PERFORMING EXPERIMENTS ##########"
 	    echo "python fit.py --weights_folder ${w_f} --model ${model} --db_name ${db_name} "
	    echo "--param_files ${params} > ${w_f}/learning_log.log "
	    python fit.py --weights_folder ${w_f} --model ${model} --db_name ${db_name} \
	        --param_files ${params} > ${w_f}/learning_log.log

	    echo "########## PROCESSING EVAL FILES ##########"
	    ./monitoring_process.sh ${w_f} ${type_eval}
	    echo "########## GENERATING RESULT SUMMARY ##########"
	    python ./result_summary.py --model ${model} \
		--weights_folder ${w_f} --db  ${db_name}\
		--param_files ${params} "params/real_data/mon_pts.json"
	    echo "########## DONE ##########"
	done
    done
}

model="auc_cons"
for db in "adult" "bank"
do
    crossval ${db} ${model} "auc" &
done

db="german"
model="auc_cons_bnsp"
crossval ${db} ${model} "auc" &

db="compas"
model="auc_cons_bpsn"
crossval ${db} ${model} "auc" &
wait

model="ptw_cons"
for db in "bank" "german" "adult" "compas"
do
    crossval ${db} ${model} "ptw" &
done
wait
