import numpy as np
import os
from tqdm import tqdm

path = os.path.join('logs','summaries.csv')
print(path)
if not os.path.exists(path):
    print(' file not exist')
    exit()
with open(path, encoding='utf-8') as f:
    # print(path)
    data = np.loadtxt(f, str, delimiter=",")
    if data.size != 0:
        val_psnr = np.array(data[1:-1, -1]).astype(float)

mean_psnr = val_psnr.mean()
std_psnr = np.sqrt(((val_psnr - mean_psnr) ** 2).mean())
print(len(val_psnr), ' images, and the avg psnr is: ', mean_psnr, ' var psnr is: ',std_psnr)
