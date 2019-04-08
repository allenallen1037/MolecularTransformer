dataset=seq2seq_5000
model=${dataset}_model_avg.pt

python translate.py -model experiments/models/${model} \
                    -gpu 2 \
                    -n_best 5 \
                    -src data/${dataset}/src-test.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_test.txt \
                    -batch_size 128 -replace_unk -max_length 200 -fast