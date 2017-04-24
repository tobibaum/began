# load real data
import os
import glob
import numpy as np
from PIL import Image

file_dir = '/Users/tobi/Documents/celeba/img_align_celeba'
files = glob.glob(os.path.join(file_dir, '*.jpg'))
out_img = []
for f in files:
    img_pil = Image.open(f)
    img = np.array(img_pil.crop([20,35,168,183]).resize([64,64]))
    out_img.append(img)
imgs = np.array(out_img)
from IPython import embed; embed(); raise
