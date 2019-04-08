dataset=seq2seq_5000
model=${dataset}_model_avg.pt

python score_predictions.py -targets data/${dataset}/tgt-test.txt \
    -invalid_smiles \
    -predictions experiments/results/predictions_${model}_on_${dataset}_test.txt