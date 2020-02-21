#!/bin/bash
do_run () {
    db_name=$1
    model=$2
    model_type=$3
    lamb_no=$4
    i=$5
    echo "***** WORKING ON "${db_name}" "${model}" lambda_"${lamb_no}" run "${i}" *****"
    w_f=results/avg_${db_name}/${model}/lambda_${lamb_no}/run_${i}
    params="params/fixed.json params/toy_"${model_type}".json params/lambda/lambda_"${lamb_no}".json"
    mkdir -p ${w_f}
    echo "########## PERFORMING EXPERIMENTS ##########"
    python fit.py --weights_folder ${w_f} --model ${model} --db_name ${db_name} \
      --param_files ${params} \
      > ${w_f}/learning_log.log

    echo "########## PROCESSING EVAL FILES ##########"
    ./monitoring_process.sh ${w_f} ${model_type}
    echo "########## GENERATING RESULT SUMMARY ##########"
    python ./result_summary.py --model ${model} \
	--weights_folder ${w_f} --db  ${db_name}\
	--param_files ${params}
    echo "########## GENERATING PLOT SQUARE ##########"
    python toy_2d_plot.py --model ${model} \
        --weights_folder ${w_f} --db  ${db_name}\
        --param_files ${params}
    echo "########## DONE ##########"
}

db_name="toy1" 
model="auc_cons"
{ 
    for lamb_no in 0 3
    do
	for i in {0..100}
	do
	    do_run ${db_name} ${model} "auc" ${lamb_no} $i
	done
    done

    for lamb_no in 0 3
    do
	fold=results/avg_${db_name}/${model}/lambda_${lamb_no}
	python toy_result_summary.py $fold
    done
} & 

{
    model="auc_cons"
    db_name="toy2"
    for i in {0..100}
    do
	for lamb_no in 0 3
	do
	    do_run ${db_name} ${model} "auc" ${lamb_no} $i
	done
    done

    for lamb_no in 0 3
    do
	fold=results/avg_${db_name}/${model}/lambda_${lamb_no}
	python toy_result_summary.py $fold
    done
} & 


{
    model="ptw_cons"
    db_name="toy2"
    for i in {0..100}
    do
	for lamb_no in 0 3
	do
	    do_run ${db_name} ${model} "ptw" ${lamb_no} $i
	done
    done

    for lamb_no in 0 3
    do
	fold=results/avg_${db_name}/${model}/lambda_${lamb_no}
	python toy_result_summary.py $fold
    done
} & 

wait
