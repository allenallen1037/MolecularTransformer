dataset=nih_data
model=${dataset}_model_avg.pt

python translate.py -model experiments/models/${model} \
                    -gpu 2 \
                    -beam_size 50 \
                    -n_best 50 \
                    -src data/${dataset}/src-test.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_test.txt \
                    -batch_size 2 -replace_unk -max_length 200 -fast