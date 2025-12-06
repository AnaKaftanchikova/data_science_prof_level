import os
import random
from PIL import Image, ImageEnhance

RAW_DIR = "data/dogs_raw"
OUT_DIR = "data/dogs_preprocessed"

os.makedirs(OUT_DIR, exist_ok=True)

def augment(img):
    variants = []

    variants.append(img)
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    enhancer = ImageEnhance.Brightness(img)
    variants.append(enhancer.enhance(1.2))

    enhancer = ImageEnhance.Contrast(img)
    variants.append(enhancer.enhance(1.3))

    return variants

images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(("jpg","jpeg","png"))]

all_out = []

for fname in images:
    path = os.path.join(RAW_DIR, fname)
    img = Image.open(path).convert("RGB")
    img = img.resize((256,256))

    aug_imgs = augment(img)

    for i, im in enumerate(aug_imgs):
        outname = fname.split(".")[0] + f"_aug{i}.jpg"
        im.save(os.path.join(OUT_DIR, outname))
        all_out.append(outname)

random.shuffle(all_out)
split = int(len(all_out)*0.8)

trainA = all_out[:split]
testA  = all_out[split:]

trainA_dir = "data/vangogh/trainA"
testA_dir  = "data/vangogh/testA"
trainA_dir2 = "data/monet/trainA"
testA_dir2  = "data/monet/testA"

os.makedirs(trainA_dir, exist_ok=True)
os.makedirs(testA_dir, exist_ok=True)
os.makedirs(trainA_dir2, exist_ok=True)
os.makedirs(testA_dir2, exist_ok=True)

for fname in trainA:
    src = os.path.join(OUT_DIR, fname)
    os.system(f"cp '{src}' '{trainA_dir}/'")
    os.system(f"cp '{src}' '{trainA_dir2}/'")

for fname in testA:
    src = os.path.join(OUT_DIR, fname)
    os.system(f"cp '{src}' '{testA_dir}/'")
    os.system(f"cp '{src}' '{testA_dir2}/'")

print("Готово.")
print("trainA/testA созданы для Van Gogh и Monet.")
