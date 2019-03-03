dataset=MIT_mixed
model=${dataset}_model_step_50000.pt

python score_predictions.py -targets data/${dataset}/tgt-test.txt \
    -invalid_smiles \
    -predictions experiments/results/predictions_${model}_on_${dataset}_test.txt