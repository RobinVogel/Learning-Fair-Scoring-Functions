# Starting all of the experiments 
# Will take a few days on a simple machine, can run quickly
# when parallelized, details in toy_experiments.sh and real_data_experiments.sh.

bash toy_experiments.sh

bash real_data_experiments.sh

# For toy1

mkdir -p figures/sec4/limits-AUC/
python illu-sec4-limits-AUC.py
# figures/sec4/limits-AUC/dist.pdf
# figures/sec4/limits-AUC/roc.pdf

# For toy1

mkdir -p figures/supp/synth_data/toy1/
python illu-supp-sgd-solutions.py toy1
# figures/supp/synth_data/toy1/sgd_solutions.pdf

fold_in="results/avg_toy1/auc_cons"
fold_out="figures/supp/synth_data/toy1/avg_rocs"
mkdir -p ${fold_out}"/lambda_0"
mkdir -p ${fold_out}"/lambda_3"
cp ${fold_in}/lambda_0/legend-sec3.pdf ${fold_out}/legend-sec3.pdf
cp ${fold_in}/lambda_0/roc_curves_quant_sec3.pdf ${fold_out}/lambda_0/roc_quant_sec3.pdf
cp ${fold_in}/lambda_3/roc_curves_quant_sec3.pdf ${fold_out}/lambda_3/roc_quant_sec3.pdf

fold_in="results/avg_toy1/auc_cons"
fold_out="figures/supp/synth_data/toy1/score_2d"
mkdir -p ${fold_out}
cp ${fold_in}/lambda_0/run_0/final_analysis/decision_plot.pdf ${fold_out}/no_lambda.pdf
cp ${fold_in}/lambda_3/run_0/final_analysis/decision_plot.pdf ${fold_out}/lambda.pdf
cp ${fold_in}/lambda_0/run_0/final_analysis/legend_decision_plot.pdf figures/supp/synth_data/score_2d_legend.pdf

# For toy2

python illu-supp-toy2.py toy2
# figures/supp/synth_data/toy2/roc_sec3_s1.pdf
# figures/supp/synth_data/toy2/roc_sec4_s1.pdf
# figures/supp/synth_data/toy2/roc_sec3_s2.pdf
# figures/supp/synth_data/toy2/roc_sec4_s2.pdf

mkdir -p figures/supp/synth_data/toy2/
python illu-supp-sgd-solutions.py toy2
#  figures/supp/synth_data/toy2/sgd_solutions.pdf


fold_in="results/avg_toy2/ptw_cons"
fold_out="figures/supp/synth_data/toy2/avg_rocs"
mkdir -p ${fold_out}"/lambda_0"
mkdir -p ${fold_out}"/lambda_3"
cp ${fold_in}/lambda_0/legend-sec4.pdf ${fold_out}/legend-sec4.pdf
cp ${fold_in}/lambda_0/roc_curves_quant_sec4.pdf ${fold_out}/lambda_0/roc_quant_sec4.pdf
cp ${fold_in}/lambda_3/roc_curves_quant_sec4.pdf ${fold_out}/lambda_3/roc_quant_sec4.pdf

fold_in="results/avg_toy2/ptw_cons"
fold_out="figures/supp/synth_data/toy2/score_2d"
mkdir -p ${fold_out}
cp ${fold_in}/lambda_0/run_0/final_analysis/decision_plot.pdf ${fold_out}/no_lambda.pdf
cp ${fold_in}/lambda_3/run_0/final_analysis/decision_plot.pdf ${fold_out}/lambda.pdf
cp ${fold_in}/lambda_0/run_0/final_analysis/legend_decision_plot.pdf figures/supp/synth_data/score_2d_legend.pdf

# For real data
python tables_generation.py

get_roc_for_paper () {
    outname=$1
    db_name=$2
    model_name=$3
    no_lambda=$4
    no_reg=$5

    infold=results/${db_name}/${model_name}/reg_${no_reg}/lambda_${no_lambda}/final_analysis
    outfold=figures/supp/real_data/${db_name}/${model_name}/
    mkdir -p ${outfold}
    cp ${infold}/roc_sec4.pdf ${outfold}/${outname}.pdf
}

# Selected ROC curves
legend_in=results/german/auc_cons_bnsp/reg_0/lambda_0/final_analysis/leg_sec4.pdf
legend_out=figures/supp/real_data/leg_roc_sec4.pdf
cp ${legend_in} ${legend_out}

# TODO: Here, one needs to modify the number to select the most
# striking results for different lambda, lambda_reg, see the tables.

# Baselines
get_roc_for_paper roc_sec4_bl german auc_cons_bnsp 0 5
get_roc_for_paper roc_sec4_bl adult auc_cons 0 3
get_roc_for_paper roc_sec4_bl compas auc_cons_bpsn 0 3
get_roc_for_paper roc_sec4_bl bank auc_cons 0 3

# AUC-based models
get_roc_for_paper roc_sec4 german auc_cons_bnsp 4 5
get_roc_for_paper roc_sec4 adult auc_cons 1 3
get_roc_for_paper roc_sec4 compas auc_cons_bpsn 2 3
get_roc_for_paper roc_sec4 bank auc_cons 1 5

# ROC-based models
get_roc_for_paper roc_sec4 german ptw_cons 5 5
get_roc_for_paper roc_sec4 adult ptw_cons 1 3
get_roc_for_paper roc_sec4 compas ptw_cons 5 1
get_roc_for_paper roc_sec4 bank ptw_cons 3 4

# Selected dynamics
mkdir -p figures/supp/real_data/sel_auc_dyn/
cp results/bank/auc_cons/reg_6/lambda_4/dyn_analysis/plots_fancy/* \
    figures/supp/real_data/sel_auc_dyn/
mkdir -p figures/supp/real_data/sel_ptw_dyn/
cp results/adult/ptw_cons/reg_4/lambda_4/dyn_analysis/plots_fancy/* \
    figures/supp/real_data/sel_ptw_dyn/
