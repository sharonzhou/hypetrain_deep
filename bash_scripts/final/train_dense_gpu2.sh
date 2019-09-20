SUFFIX="_auroc_dense"
METRIC_NAME="--metric_name auroc_dense"
GPU_IDS="--gpu_ids 2"
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


python train.py --experiment_name pws$SUFFIX --models $PWS --models_valid $PWS $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name bpw$SUFFIX --models $BPW --models_valid $BPW $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name bs$SUFFIX --models $BS --models_valid $BS $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name w$SUFFIX --models $W --models_valid $W $GPU_IDS $NUM_EPOCHS $METRIC_NAME

