import os
import cv2
import json
import numpy as np

def get_person_box(person):
    body = np.array(person['pose_keypoints_2d'], dtype=np.float32).reshape([-1,3])
    face = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1,3])
    lhand = np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1,3])
    rhand = np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1,3])

    all = np.concatenate([body, face, lhand, rhand], axis=0)
    #all = body
    all = all[all[:,2] > 0.2]
    xmin = np.min(all[:, 0])
    xmax = np.max(all[:, 0])
    ymin = np.min(all[:, 1])
    ymax = np.max(all[:, 1])
    w = xmax - xmin
    h = ymax - ymin
    cx = (xmax + xmin) / 2.0
    cy = (ymax + ymin) / 2.0
    return (cx, cy, w, h)

if __name__=='__main__':
    root_path = '/home/user/Desktop/yjs/codes/smplify-x/MeshFit/data/realsense'
    depth_dir = os.path.join(root_path, 'depth')
    img_dir = os.path.join(root_path, 'img')
    mask_dir = os.path.join(root_path, 'graphonomy_output')
    save_dir = os.path.join(root_path, 'masks')

    img_names = os.listdir(img_dir)
    depth_names = os.listdir(depth_dir)
    for i in range(115):
        img_path = os.path.join(img_dir, str(i)+'.jpg')
        depth_path = os.path.join(depth_dir, str(i)+'.npy')
        mask_path = os.path.join(mask_dir, str(i)+'_resized.png')
        save_path = os.path.join(save_dir, str(i) + '.png')

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (1280, 720))

        mask_new = mask!=0
        mask_new = np.array(mask_new*255, dtype=np.uint8)
        cv2.imwrite(save_path, mask_new)


        depth = np.load(depth_path)

        valid = (mask!=0)#*(depth != 0)*(depth <3.5)

        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (640, 360))
        cv2.imwrite(os.path.join(root_path, 'img_resized', str(i)+'_resized.jpg'), img_resized)
        img[~valid, :] = 0.0
        cv2.imshow('tmp', img)
        cv2.waitKey(20)

    ## keypoints trim
    # root_path = '/home/user/Desktop/yjs/codes/smplify-x/MeshFit/data/realsense/keypoints_new'
    # save_path = '/home/user/Desktop/yjs/codes/smplify-x/MeshFit/data/realsense/keypoints_new'
    # file_names = os.listdir(root_path)
    # for file_name in file_names:
    #     path = os.path.join(root_path, file_name)
    #     with open(path, 'r') as json_file:
    #         sample = json.load(json_file)
    #         if len(sample['people']) != 1:
    #             print(path)
    #             boxes = []
    #             for i in range(len(sample['people'])):
    #                 person = sample['people'][i]
    #                 boxes.append(get_person_box(person))
    #             boxes = np.array(boxes)
    #             area = boxes[:, 2] * boxes[:, 3]
    #
    #             area_ax = np.argmax(area)
    #             sample['people'] = [sample['people'][area_ax]]
    #     #
    #     # path = os.path.join(save_path, file_name)
    #     # with open(path, 'w') as json_file:
    #     #     json.dump(sample, json_file)




