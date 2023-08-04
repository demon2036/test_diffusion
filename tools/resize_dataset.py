import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from data.dataset import get_dataloader
from tqdm import tqdm


def save_image(x,count,save_path='/home/john/data/test'):
    try:
        x=np.array(x)
        x=x/2+0.5
        x=x*255
        x=np.clip(x,0,255).astype('uint8')
        # print(x.shape)
        img=Image.fromarray(x)
        img.save(f'{save_path}/{count}.png')
    except Exception as e:
        print(e)


if __name__=='__main__':
    dl = get_dataloader(batch_size=32, file_path='/home/john/data/FFHQ', image_size=256, drop_last=False)
    save_path='/home/john/data/FFHQ256'
    os.makedirs(save_path,exist_ok=True)
    count = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for data in tqdm(dl):
            for x in data:
                pool.submit(save_image,x,count,save_path)
            count += 1


