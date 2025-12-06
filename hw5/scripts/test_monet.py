import os

os.system(
    "python cyclegan/test.py "
    "--dataroot data/monet "
    "--name dog_monet "
    "--model cycle_gan "
    "--phase test "
    "--no_dropout "
)
