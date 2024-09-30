export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/work/miniconda3/envs/gnn/lib/
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb=3000"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=${10}
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
new_version=$(ls -d lightning_logs/version_* | sed 's/lightning_logs\/version_//' | sort -n | tail -1 | awk '{print $1+1}')
echo $new_version

n_head=12 #12 #8
n_layer=12 #12 #10
gpus=1
n_emb=$(expr $n_head \* 64) # 128 | 768
grad_acc=1
max_epochs=50
balance_training_dataset='False'

dataset_name=$1 # BBBP | Tox21
seed=$2
n_batch=$3
model_arch=$4 # CrossAttentionEncoderBuilder | CrossAttentionTail | SMILESonly | Graphonly | MolFormer
attention_type=$5 #full_attention_cross | linear_attention_cross | linear_attention_smiles | full_attention_smiles | full_attention_spatial | linear_attention_spatial | rotary_attention_spatial
graph_padding=$6
pos_type=$7
resume_from_checkpoint=$8
train_load=$9

lr_start=3e-5 # 3e-5 | 1e-5
q_dropout=0.1 # backbone 0.1
d_dropout=0.1 # token encoding 0.1
dropout=0.1 # output layer 0.1
device_id=0
splitting='scaffold_balance' #'scaffold' # random | scaffold | scaffold_balance
aug_training_dataset='False'
freeze='False'
unfreeze=-1

python finetune_pubchem_molformer_multitask.py \
        --device cuda \
        --n_batch $n_batch  \
        --grad_acc $grad_acc\
        --n_head $n_head \
        --n_layer $n_layer \
        --n_embd $n_emb \
        --seed $seed \
        --dataset_name $dataset_name \
        --splitting $splitting \
        --balance_training_dataset $balance_training_dataset \
        --model_arch $model_arch \
        --attention_type $attention_type \
        --lr_start $lr_start \
        --q_dropout $q_dropout \
        --d_dropout $d_dropout \
        --dropout $dropout \
        --device_id $device_id \
        --freeze $freeze \
        --unfreeze $unfreeze \
        --graph_padding $graph_padding \
        --max_len 1000 \
        --lr_multiplier 1 \
        --n_workers 20 \
        --max_epochs $max_epochs \
        --aug_training_dataset $aug_training_dataset \
        --pos_type $pos_type \
        --gpus $gpus \
        --num_nodes 1 \
        --accelerator ddp \
        --num_feats 32 \
        --root_dir . \
        --checkpoint_every 1000 \
        --train_load $train_load \
        --eval_every 1000 \
        --rotate \
        --debug \
        --valid_size 0.1 \
        --test_size 0.1 \
        --dims 768 768 768 1 \
        --resume_from_checkpoint $resume_from_checkpoint \
        | tee "lightning_logs/logs/$(date +%F_%R | sed 's/:/_/g')_logs_version_${new_version}.txt"
        