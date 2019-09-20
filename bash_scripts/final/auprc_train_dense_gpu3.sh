SUFFIX="_auprc_dense"
METRIC_NAME="--metric_name auprc_dense"
GPU_IDS="--gpu_ids 3"
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


python train.py --experiment_name pw$SUFFIX --models $PW --models_valid $PW $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name bps$SUFFIX --models $BPS --models_valid $BPS $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name bpws$SUFFIX --models $BPWS --models_valid $BPWS $GPU_IDS $NUM_EPOCHS $METRIC_NAME

