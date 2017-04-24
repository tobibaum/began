# load real data
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

file_dir = '/home/tobi/celeba/img_align_celeba'
files = glob.glob(os.path.join(file_dir, '*.jpg'))
out_img = []
for f in tqdm(files):
    img_pil = Image.open(f)
    img = np.array(img_pil.crop([20,35,168,183]).resize([64,64]))
    img = img / 128. - 1
    out_img.append(img)
imgs = np.array(out_img)
np.save(open('data.npy', 'w'), imgs)
