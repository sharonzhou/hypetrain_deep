SUFFIX="_dense"
METRIC_NAME="--metric_name pearsonr"
GPU_IDS="--gpu_ids 0"
NUM_EPOCHS="--num_epochs 200"

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


python train.py --experiment_name bp$SUFFIX --models $BP --models_valid $BP $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name b$SUFFIX --models $B --models_valid $B $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name ps$SUFFIX --models $PS --models_valid $PS $GPU_IDS $NUM_EPOCHS $METRIC_NAME

