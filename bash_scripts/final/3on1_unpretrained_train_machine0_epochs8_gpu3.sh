SUFFIX="_3on1_8_unpretrained"
METRIC_NAME="--metric_name accuracy"
NUM_EPOCHS="--num_epochs 8"

B="began"
P="progan"
W="wgan_gp"
S="stylegan"

BP="began,progan"
WS="wgan_gp,stylegan"
BW="began,wgan_gp"
PS="progan,stylegan"
BS="began,stylegan"
PW="progan,wgan_gp"

PWS="progan,wgan_gp,stylegan"
BWS="began,wgan_gp,stylegan"
BPS="began,progan,stylegan"
BPW="began,progan,wgan_gp"

BPWS="began,progan,wgan_gp,stylegan"


python train.py --experiment_name bws$SUFFIX --models $BWS --models_valid $BWS --gpu_ids 3 $NUM_EPOCHS $METRIC_NAME --no-pretrained

