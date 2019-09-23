SUFFIX="_unpretrained"
METRIC_NAME="--metric_name auprc_dense"
GPU_IDS="--gpu_ids 0"
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


python train.py --experiment_name ps$SUFFIX --models $PS --models_valid $PS $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained
python train.py --experiment_name s$SUFFIX --models $S --models_valid $S $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained
python train.py --experiment_name pw$SUFFIX --models $PW --models_valid $PW $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained
#python train.py --experiment_name b$SUFFIX --models $B --models_valid $B $GPU_IDS $NUM_EPOCHS --metric_name accuracy --no-pretrained
#python train.py --experiment_name bp$SUFFIX --models $BP --models_valid $BP $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained
#python train.py --experiment_name ws$SUFFIX --models $WS --models_valid $WS $GPU_IDS $NUM_EPOCHS $METRIC_NAME --no-pretrained

