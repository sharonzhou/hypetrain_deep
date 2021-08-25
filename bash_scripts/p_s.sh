python train.py --experiment_name progan_eval_progan_too --models progan --models_valid progan
python test.py --ckpt_path /deep/group/sharonz/hypetrain_deep/saved_models/progan/best.pth.tar --models_valid progan --models_test stylegan
