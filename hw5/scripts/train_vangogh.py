import os
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

os.system(
    "python cyclegan/cyclegan/train.py "
    "--dataroot data/vangogh "
    "--name dog_vangogh_fast "
    "--model cycle_gan "
    "--input_nc 3 "
    "--output_nc 3 "
    "--direction AtoB "
    "--batch_size 1 "
    "--load_size 256 "
    "--crop_size 256 "
    "--n_epochs 10 "
    "--n_epochs_decay 10 "
    "--ngf 32 "
    "--ndf 32 "
    "--lambda_identity 0 "
    "--display_freq 10 "
    "--print_freq 10 "
    "--no_html "
)
