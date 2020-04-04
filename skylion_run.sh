set -ex
export TPU_HOST=10.255.128.3
export TPU_NAME=tpu-euw4a-9
exec python3 wrapper2.py train.py --dataset train.py --save_every 999999 --save_time 999999 --sample_every 99999999
