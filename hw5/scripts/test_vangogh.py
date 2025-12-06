import os

os.system(
    "python cyclegan/cyclegan/test.py "
    "--dataroot data/dogs_preprocessed "
    "--name dog_vangogh_fast "
    "--model cycle_gan "
    "--input_nc 3 "
    "--output_nc 3 "
    "--direction AtoB "
    "--load_size 256 "
    "--crop_size 256 "
    "--num_test 200 "
    "--ngf 32 "
    "--ndf 32 "
)
