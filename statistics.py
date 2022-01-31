import numpy as np
import os
from tqdm import tqdm

dirs = ['logs']
image_cat = ['div2k','text']
for dir in dirs:
    fid = open(os.path.join(dir,dir.split('/')[-1] + '_summaries.csv'), 'w')
    tqdm.write("Image, PSNR", fid)

    for imcat in image_cat:
        val_psnr = []
        files = os.listdir(dir)
        files.sort()
        for d in files:
            if imcat in d:
                path = os.path.join(dir,d,'summaries.csv')
                if not os.path.exists(path):
                    print(' file not exist')
                with open(path,encoding = 'utf-8') as f:
                    print(path)
                    data = np.loadtxt(f,str,delimiter=",")
                    if data.size!=0:
                        val_psnr += [float(data[-1][-2])]
                        #val_psnr += [data[1:,-4].astype(np.float32).max()]
                    tqdm.write("%s, %0.6f" % (d, float(data[-1][-2])), fid)
        val_psnr = np.array(val_psnr)
        mean_psnr = val_psnr.mean()
        std_psnr = np.sqrt(((val_psnr - mean_psnr)**2).mean())
        print('Image set ',imcat, ' has ',len(val_psnr), ' images, and the avg psnr is: ', mean_psnr, ' var psnr is: ',std_psnr)
    fid.close()



