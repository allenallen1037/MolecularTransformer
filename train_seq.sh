dataset=nih_data # MIT_mixed_augm / STEREO_mixed_augm

python  train.py -data data/${dataset}/${dataset} \
      -save_model experiments/checkpoints/${dataset}/${dataset}_model \
      -seed 42 -gpu_ranks 0 1 -world_size 2 -save_checkpoint_steps 5000 -keep_checkpoint 20 \
      -train_steps 500000 -param_init 0  -param_init_glorot -max_generator_batches 16 -valid_steps 5000 -valid_batch_size 16 \
      -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
      -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
      -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
      -rnn_size 512 -word_vec_size 512 -encoder_type brnn -decoder_type rnn \
      -enc_layers 2 -dec_layers 4 -rnn_type LSTM \
      -dropout 0.1 -share_embeddings \
      -global_attention mlp -global_attention_function softmax \

