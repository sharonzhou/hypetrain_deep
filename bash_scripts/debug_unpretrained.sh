#python train.py --experiment_name debug_unpretrained --no-pretrained --num_epochs 5
python test.py --ckpt_path '/deep/group/sharonz/hypetrain_deep/saved_models/debug_unpretrained/best.pth.tar' --models_test began,progan,stylegan,wgan_gp
