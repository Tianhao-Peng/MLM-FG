ckt_l=('checkpoints/trans_10m.ckpt' 'checkpoints/trans_20m.ckpt' 'checkpoints/trans_100m.ckpt')
scale_l=('pubchem-10m' 'pubchem-20m' 'pubchem-100m')

for ckt in "${ckt_l[@]}"; do
    for scale in "${scale_l[@]}"; do
        sh run_one_finetune.sh BBBP 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune.sh BACE 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh ClinTox 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh Tox21 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh SIDER 0 16 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune.sh HIV 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh MUV 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune.sh ESOL 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune.sh FreeSolv 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune.sh Lipo 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune.sh qm7 0 32 SMILESonly rotary False degree "$ckt" "$scale"
        sh run_one_finetune.sh qm8 0 32 SMILESonly rotary False degree "$ckt" "$scale"
    done
done

ckt_l = ('checkpoints/RoBERTa_10m.ckpt' 'checkpoints/RoBERTa_20m.ckpt' 'checkpoints/RoBERTa_100m.ckpt')
for ckt in "${ckt_l[@]}"; do
    for scale in "${scale_l[@]}"; do
        sh run_one_finetune.sh BBBP 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune.sh BACE 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh ClinTox 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh Tox21 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh SIDER 0 16 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune.sh HIV 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune_multitask.sh MUV 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune.sh ESOL 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune.sh FreeSolv 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune.sh Lipo 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune.sh qm7 0 32 RoBERTa - False degree "$ckt" "$scale"
        sh run_one_finetune.sh qm8 0 32 RoBERTa - False degree "$ckt" "$scale"+

    done
done