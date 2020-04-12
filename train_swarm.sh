#!/bin/sh
set -x

targets=
#for i in {50..99}
for i in {0..49}
do
	targets=${targets},grpc://10.48.${i}.2:8470
done
#targets=${targets},grpc://10.48.47.2:8470

#for i in {1..9}
#for i in {0..9}
#do
#	dataset="${dataset},${HOME}/data/2020-01-17-abc-combinedmidi-training-0${i}.txt.npz"
#done
dataset="${HOME}/data/2020-01-17-abc-combinedmidi-training.tok16"

#export TFLEX_DELAY_WRITES=1
export TFLEX_SKIP_WARMUP=1
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 30720 --batch_size 7 --learning_rate 0.000055 --dtype float32 --device 0 --skip_cores 1 --max_cores 7 --colocate_gradients --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 30720 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 1 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"

# this works
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 4096 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 20480 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 12288 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 30720 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 32768 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 28672 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 11264 --batch_size 1 --learning_rate 0.000055 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 11264 --batch_size 1 --learning_rate 0.000027500 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 5632 --batch_size 1 --learning_rate 0.000027500 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 5632 --batch_size 1 --learning_rate 0.0000027500 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 5632 --batch_size 1 --learning_rate 0.0000027500 --dtype float32 --skip_cores 1 --max_cores 4 --memory_saving_gradients --allow_soft_placement --init_tpu "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 5632 --batch_size 1 --learning_rate 0.000027500 --dtype float32 --skip_cores 1 --max_cores 1 --memory_saving_gradients --allow_soft_placement --init_tpu "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 5632 --batch_size 8 --learning_rate 0.000027500 --dtype float32 --skip_cores 0 --max_cores 8 --memory_saving_gradients --allow_soft_placement --init_tpu "$@"
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 5632 --batch_size 8 --learning_rate 0.0000055000 --dtype float32 --skip_cores 0 --max_cores 8 --memory_saving_gradients --allow_soft_placement --init_tpu "$@"

# changed sample_ctx 5632 to 11264 on apr 7 12:18pm PST (Tue Apr  7 19:18:39 UTC 2020)
#exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 11264 --batch_size 8 --learning_rate 0.0000055000 --dtype float32 --skip_cores 0 --max_cores 8 --memory_saving_gradients --allow_soft_placement --init_tpu "$@"
exec python3 -m pdb -c continue train_multi.py --targets "$targets" --dataset "$dataset" --run_name midi117m30k --optimizer adam --model_name 117M --truncate_weights --n_ctx 24576 --sample_ctx 11264 --batch_size 1 --learning_rate 0.0000055000 --dtype float32 --skip_cores 0 --max_cores -1 --memory_saving_gradients --allow_soft_placement --init_tpu --coreless "$@"


