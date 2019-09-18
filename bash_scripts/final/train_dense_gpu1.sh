SUFFIX="_dense"
METRIC_NAME="--metric_name pearsonr"
GPU_IDS="--gpu_ids 1"
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


python train.py --experiment_name p$SUFFIX --models $P --models_valid $P $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name bws$SUFFIX --models $BWS --models_valid $BWS $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name bw$SUFFIX --models $BW --models_valid $BW $GPU_IDS $NUM_EPOCHS $METRIC_NAME
python train.py --experiment_name ws --models $WS --models_valid $WS $GPU_IDS $METRIC_NAME

