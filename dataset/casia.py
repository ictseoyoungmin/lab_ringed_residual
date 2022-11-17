from abstractDataset import AbstractDataset
import os
import numpy as np
from PIL import Image
from pathlib import Path

if __name__ == '__main__':
    # CASIA2 has non-jpg files - we convert them here. You can choose original extension or jpeg when you test.
    root = Path(r"C:\Users\zxcas\PythonWork\DATASETS") / "CASIA2.0"

    imlist = []  # format: tamp.ext,mask.png,jpg_converted.jpg (if already .jpg, jpg_converted.jpg is same as tamp.ext)
    # CASIA2
    tamp_root = root / "Tp"
    mask_root = root / "Groundtruth"
    jpg_root = root / "jpg"
    jpg_root.mkdir(exist_ok=True)
    for file in os.listdir(tamp_root):
        if file in ['Tp_D_NRD_S_B_ani20002_nat20042_02437.tif']:
            continue  # stupid file
        if not file.lower().endswith(".jpg"):
            if not file.lower().endswith(".tif"):
                print(file)
                continue
            # convert to jpg
            jpg_im = Image.open(tamp_root / file)
            jpg_im.save(jpg_root/(os.path.splitext(file)[0]+".jpg"), quality=100, subsampling=0)
            imlist.append(','.join([str(Path("Tp") / file),
                                    str(Path("Groundtruth") / (os.path.splitext(file)[0] + "_gt.png")),
                                    str(Path("jpg") / (os.path.splitext(file)[0] + ".jpg"))]))
        else:
            imlist.append(','.join([str(Path("Tp") / file),
                                str(Path("Groundtruth") / (os.path.splitext(file)[0]+"_gt.png")),
                                str(Path("Tp") / file)]))
        assert (mask_root/(os.path.splitext(file)[0]+"_gt.png")).is_file()
    print(len(imlist))  # 6042

    # mask validation
    new_imlist=[]
    for s in imlist:
        im, mask, _ = s.split(',')
        im_im = np.array(Image.open(root / im))
        mask_im = np.array(Image.open(root / mask))
        if im_im.shape[0] != mask_im.shape[0] or im_im.shape[1] != mask_im.shape[1]:
            print("Skip:", im, mask)
            continue
        new_imlist.append(s)

    with open("CASIA_list.txt", "w") as f:
        f.write('\n'.join(new_imlist)+'\n')
    print(len(new_imlist))  # 6025

    # CASIA2 authentic
    tamp_root = root / "Au"
    jpg_root = root / "jpg"
    jpg_root.mkdir(exist_ok=True)
    for file in os.listdir(tamp_root):
        if not file.lower().endswith(".jpg"):
            if not file.lower().endswith(".bmp"):
                print(file)
                continue
            # convert to jpg
            jpg_im = Image.open(tamp_root / file)
            jpg_im.save(jpg_root / (os.path.splitext(file)[0] + ".jpg"), quality=100, subsampling=0)
            imlist.append(','.join([str(Path("Au") / file),
                                    'None',
                                    str(Path("jpg") / (os.path.splitext(file)[0] + ".jpg"))]))
        else:
            imlist.append(','.join([str(Path("Au") / file),
                                    'None',
                                    str(Path("Au") / file)]))
    print(len(imlist))  # 6042

    with open("CASIA_v2_auth_list.txt", "w") as f:
        f.write('\n'.join(imlist)+'\n')


