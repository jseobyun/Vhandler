import os
import cv2
import numpy as np

root = '/home/user/Desktop/yjs/codes/smplify-x/MeshFit/outputs/vis'

ori = root
depth = os.path.join(root, 'chamfer_loss')
save_dir = os.path.join(root, 'cat')

for i in range(43):
    ori_path = os.path.join(ori, str(i)+'.png')
    depth_path = os.path.join(depth, str(i)+'.png')

    ori_img = cv2.imread(ori_path)
    depth_img = cv2.imread(depth_path)

    cat_img = np.concatenate([ori_img, depth_img], axis=1)

    cv2.imwrite(os.path.join(save_dir, str(i)+'.png'), cat_img)
