python test.py --ckpt_path /deep/group/sharonz/hypetrain_deep/saved_models/bpws_auprc_dense/best.pth.tar --phase train --models began,progan,wgan_gp,stylegan --models_valid began,progan,wgan_gp,stylegan --models_test began,wgan_gp,progan,stylegan --metric_name auprc_dense

python test.py --ckpt_path /deep/group/sharonz/hypetrain_deep/saved_models/bpws_auprc_dense/best.pth.tar --phase train --models began,progan,wgan_gp,stylegan --models_valid began,progan,wgan_gp,stylegan --models_test began --metric_name auprc_dense

python test.py --ckpt_path /deep/group/sharonz/hypetrain_deep/saved_models/bpws_auprc_dense/best.pth.tar --phase train --models began,progan,wgan_gp,stylegan --models_valid began,progan,wgan_gp,stylegan --models_test progan --metric_name auprc_dense

python test.py --ckpt_path /deep/group/sharonz/hypetrain_deep/saved_models/bpws_auprc_dense/best.pth.tar --phase train --models began,progan,wgan_gp,stylegan --models_valid began,progan,wgan_gp,stylegan --models_test wgan_gp --metric_name auprc_dense

python test.py --ckpt_path /deep/group/sharonz/hypetrain_deep/saved_models/bpws_auprc_dense/best.pth.tar --phase train --models began,progan,wgan_gp,stylegan --models_valid began,progan,wgan_gp,stylegan --models_test stylegan --metric_name auprc_dense
