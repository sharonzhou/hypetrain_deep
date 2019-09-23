declare -a suffixes=("_3on1_auprc_unpretrained" "_3on1_unpretrained" "_3on1_8_unpretrained")

# Test on each individually
B="began"
P="progan"
W="wgan_gp"
S="stylegan"

BP='began,progan'
WS='wgan_gp,stylegan'
BW='began,wgan_gp'
PS='progan,stylegan'
BS='began,stylegan'
PW='progan,wgan_gp'

PWS='progan,wgan_gp,stylegan'
BWS='began,wgan_gp,stylegan'
BPS='began,progan,stylegan'
BPW='began,progan,wgan_gp'

BPWS='began,progan,wgan_gp,stylegan'


for i in "${suffixes[@]}"
do
    SUFFIX="$i"

    # Trained on 1
    B_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/b$SUFFIX/best.pth.tar"
    P_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/p$SUFFIX/best.pth.tar"
    W_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/w$SUFFIX/best.pth.tar"
    S_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/s$SUFFIX/best.pth.tar"

    # Trained on 2
    BP_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/bp$SUFFIX/best.pth.tar"
    BW_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/bw$SUFFIX/best.pth.tar"
    BS_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/bs$SUFFIX/best.pth.tar"
    PW_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/pw$SUFFIX/best.pth.tar"
    PS_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/ps$SUFFIX/best.pth.tar"
    WS_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/ws$SUFFIX/best.pth.tar"

    # Trained on 3
    BPW_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/bpw$SUFFIX/best.pth.tar"
    BPS_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/bps$SUFFIX/best.pth.tar"
    PWS_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/pws$SUFFIX/best.pth.tar"
    BWS_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/bws$SUFFIX/best.pth.tar"

    # Trained on 4
    BPWS_PATH="/deep/group/sharonz/hypetrain_deep/saved_models/bpws$SUFFIX/best.pth.tar"
    
    declare -A paths=(["$S"]="$BPW_PATH" ["$W"]="$BPS_PATH" ["$B"]="$PWS_PATH" ["$P"]="$BWS_PATH")
    
    for j in "${!paths[@]}"
    do
        python test.py --ckpt_path "${paths[$j]}" --models_test "$j" --final_csv
    done
done
