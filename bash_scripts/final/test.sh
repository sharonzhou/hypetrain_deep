# Trained on 1, tested on 1 (self + others)
B_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/b/best.pth.tar'
P_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/p/best.pth.tar'
W_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/w/best.pth.tar'
S_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/s/best.pth.tar'

B='began'
P='progan'
W='wgan_gp'
S='stylegan'

# b
python test.py --ckpt_path $B_PATH --models $B --models_valid $B --models_test $B --final_csv
python test.py --ckpt_path $B_PATH --models $B --models_valid $B --models_test $P --final_csv
python test.py --ckpt_path $B_PATH --models $B --models_valid $B --models_test $W --final_csv
python test.py --ckpt_path $B_PATH --models $B --models_valid $B --models_test $S --final_csv

# p
python test.py --ckpt_path $P_PATH --models $P --models_valid $P --models_test $P --final_csv
python test.py --ckpt_path $P_PATH --models $P --models_valid $P --models_test $B --final_csv
python test.py --ckpt_path $P_PATH --models $P --models_valid $P --models_test $W --final_csv
python test.py --ckpt_path $P_PATH --models $P --models_valid $P --models_test $S --final_csv

# w
python test.py --ckpt_path $W_PATH --models $W --models_valid $W --models_test $W --final_csv
python test.py --ckpt_path $W_PATH --models $W --models_valid $W --models_test $B --final_csv
python test.py --ckpt_path $W_PATH --models $W --models_valid $W --models_test $P --final_csv
python test.py --ckpt_path $W_PATH --models $W --models_valid $W --models_test $S --final_csv

# s
python test.py --ckpt_path $S_PATH --models $S --models_valid $S --models_test $S --final_csv
python test.py --ckpt_path $S_PATH --models $S --models_valid $S --models_test $B --final_csv
python test.py --ckpt_path $S_PATH --models $S --models_valid $S --models_test $P --final_csv
python test.py --ckpt_path $S_PATH --models $S --models_valid $S --models_test $W --final_csv


# Trained on 1, tested on 3 (others)
PWS='progan,wgan_gp,stylegan'
BWS='began,wgan_gp,stylegan'
BPS='began,progan,stylegan'
BPW='began,progan,wgan_gp'

python test.py --ckpt_path $B_PATH --models $B --models_valid $B --models_test $PWS --final_csv
python test.py --ckpt_path $P_PATH --models $P --models_valid $P --models_test $BWS --final_csv
python test.py --ckpt_path $W_PATH --models $W --models_valid $W --models_test $BPS --final_csv
python test.py --ckpt_path $S_PATH --models $S --models_valid $S --models_test $BPW --final_csv


# Trained on 2, tested on 2 (self + others)
BP_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/bp/best.pth.tar'
BW_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/bw/best.pth.tar'
BS_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/bs/best.pth.tar'
PW_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/pw/best.pth.tar'
PS_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/ps/best.pth.tar'
WS_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/ws/best.pth.tar'

BP='began,progan'
WS='wgan_gp,stylegan'
BW='began,wgan_gp'
PS='progan,stylegan'
BS='began,stylegan'
PW='progan,wgan_gp'

python test.py --ckpt_path $BP_PATH --models $BP --models_valid $BP --models_test $BP --final_csv
python test.py --ckpt_path $BP_PATH --models $BP --models_valid $BP --models_test $WS --final_csv

python test.py --ckpt_path $BW_PATH --models $BW --models_valid $BW --models_test $BW --final_csv
python test.py --ckpt_path $BW_PATH --models $BW --models_valid $BW --models_test $PS --final_csv

python test.py --ckpt_path $BS_PATH --models $BS --models_valid $BS --models_test $BS --final_csv
python test.py --ckpt_path $BS_PATH --models $BS --models_valid $BS --models_test $PW --final_csv

python test.py --ckpt_path $PW_PATH --models $PW --models_valid $PW --models_test $PW --final_csv
python test.py --ckpt_path $PW_PATH --models $PW --models_valid $PW --models_test $BS --final_csv

python test.py --ckpt_path $PS_PATH --models $PS --models_valid $PS --models_test $PS --final_csv
python test.py --ckpt_path $PS_PATH --models $PS --models_valid $PS --models_test $BW --final_csv

python test.py --ckpt_path $WS_PATH --models $WS --models_valid $WS --models_test $WS --final_csv
python test.py --ckpt_path $WS_PATH --models $WS --models_valid $WS --models_test $BP --final_csv


# Trained on 3, tested on 1 (others)
BPW_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/bpw/best.pth.tar'
BPS_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/bps/best.pth.tar'
PWS_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/pws/best.pth.tar'
BWS_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/bws/best.pth.tar'

python test.py --ckpt_path $BPW_PATH --models $BPW --models_valid $BPW --models_test $S --final_csv
python test.py --ckpt_path $BPS_PATH --models $BPS --models_valid $BPS --models_test $W --final_csv
python test.py --ckpt_path $PWS_PATH --models $PWS --models_valid $PWS --models_test $B --final_csv
python test.py --ckpt_path $BWS_PATH --models $BWS --models_valid $BWS --models_test $P --final_csv


# Trained on 4, tested on 4 (self)
BPWS_PATH='/deep/group/sharonz/hypetrain_deep/saved_models/bpws/best.pth.tar'
BPWS='began,progan,wgan_gp,stylegan'

python test.py --ckpt_path $BPWS_PATH --models $BPWS --models_valid $BPWS --models_test $BPWS --final_csv

