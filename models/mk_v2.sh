filename=$1
echo "Working on "${filename}
mkdir tmp
cp ${filename} tmp/${filename}
tf_upgrade_v2 --intree tmp/ --outtree tmpv2/ --reportfile reportv2.txt
sed -i 's/model_auc_cons/tfv2_model_auc_cons/g' tmpv2/${filename}
cp tmpv2/${filename} tfv2_${filename}
rm -r tmp/
rm -r tmpv2/
