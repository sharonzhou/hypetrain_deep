SUFFIX="_unpretrained"
METRIC_NAME="--metric_name auprc_dense"
GPU_IDS="--gpu_ids 1"
NUM_EPOCHS="--num_epochs 150"

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


python train.py --experiment_name w$SUFFIX --models $W --models_valid $W $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained
python train.py --experiment_name s$SUFFIX --models $S --models_valid $S $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained
python train.py --experiment_name ws$SUFFIX --models $WS --models_valid $WS $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained
