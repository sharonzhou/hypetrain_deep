# Approximating Human Judgment of Generated Image Quality
This is the official repository following work on the 2019 NeurIPS workshop paper Approximating Human Judgment of Generated Image Quality, to approximate perceptual realism scores generated by HYPE from raw pixel data (CelebA and generated faces from StyleGAN, ProGAN, BEGAN, and WGAN-GP) using neural networks -- here, a DenseNet-121 classification model. The goal is to provide image-level labels of human perceptual realism, instead of distribution level, which most scores like FID, Inception Score, precision, HYPE, etc provide. These scores are then compared with NVIDIA's image-level realism score.
