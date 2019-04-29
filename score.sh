dataset=nih_data
model=${dataset}_model_avg.pt

python score_predictions.py -targets data/${dataset}/tgt-test.txt \
    -invalid_smiles -beam_size 50\
    -predictions experiments/results/predictions_${model}_on_${dataset}_test.txt