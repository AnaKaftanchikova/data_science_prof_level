import os

os.system(
    "python cyclegan/train.py "
    "--dataroot data/monet "
    "--name dog_monet "
    "--model cycle_gan "
    "--batch_size 1 "
    "--gpu_ids -1 "
    "--display_id 0 "
)
