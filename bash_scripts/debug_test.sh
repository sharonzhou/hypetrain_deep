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
python test.py --ckpt_path $B_PATH --models $B --models_valid $B --models_test $P --final_csv
