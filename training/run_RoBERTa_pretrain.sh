export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=10"
new_version=$(ls -d lightning_logs/version_* | sed 's/lightning_logs\/version_//' | sort -n | tail -1 | awk '{print $1+1}')
echo $new_version
export CUDA_LAUNCH_BLOCKING=1
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export LSB_MCPU_HOSTS="10.27.233.139 40 10.27.233.134 40 10.27.233.133 40 10.27.233.130 40"
num_nodes=4

export HOSTNAME="10.27.233.134"

gpus=$(nvidia-smi -L | wc -l)
n_head=12 # 12
n_layer=12 # 12
n_emb=$(expr $n_head \* 64) 
n_batch=800
grad_acc=1
max_len=221
model_arch='RoBERTa' # SMILESonly
attention_type='-' # rotary
pretrain_style='subgraph_and_atom' # atom | subgraph_and_atom | subgraph
scale='pubchem-100m'

python train_pubchem_light.py \
        --device cuda \
        --n_batch $n_batch  \
        --grad_acc $grad_acc \
        --n_head $n_head \
        --n_layer $n_layer \
        --n_embd $n_emb \
        --max_len $max_len \
        --gpus $gpus \
        --attention_type $attention_type \
        --model_arch $model_arch \
        --pretrain_style $pretrain_style \
        --graph_padding False \
        --pos_type degree \
        --d_dropout 0.2 \
        --lr_start 3e-5 \
        --lr_multiplier 4 \
        --n_workers 8 \
        --max_epochs 50 \
        --num_nodes $num_nodes \
        --accelerator ddp \
        --num_feats 32 \
        --root_dir . \
        --checkpoint_every 10000 \
        --train_load $scale \
        --eval_every 5 \
        --rotate \
        --debug \
        | tee "lightning_logs/logs/logs_version_${new_version}_$(date +%F_%R | sed 's/:/_/g').txt"
        