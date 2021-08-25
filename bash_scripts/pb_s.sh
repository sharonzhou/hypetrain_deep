python train.py --experiment_name progan-began --models progan,began --models_valid progan,began
python test.py --ckpt_path /deep/group/sharonz/hypetrain_deep/saved_models/progan/best.pth.tar --models_valid progan,began --models_test stylegan
