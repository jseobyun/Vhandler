import os
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
    root_path = '/media/user/SSD_yjs/people_snapshot_public'
    subjects = os.listdir(root_path)
    for subject in subjects:
        print(subject)
        video_dir = os.path.join(root_path, subject)
        image_dir = os.path.join(root_path, subject, 'images')
        kps_dir = os.path.join(root_path, subject, 'keypoints')
        new_kps_dir = os.path.join(root_path, subject, 'new_keypoints')
        if not os.path.exists(new_kps_dir):
            os.mkdir(new_kps_dir)
        kps_name = os.listdir(new_kps_dir)
        for kp in kps_name:
            path = os.path.join(new_kps_dir, kp)
            with open(path, 'r') as json_file:
                sample = json.load(json_file)
                if len(sample['people']) != 1:
                    print(path)
                # if len(sample['people']) != 1:
                #     boxes = []
                #     for i in range(len(sample['people'])):
                #         person = sample['people'][i]
                #         boxes.append(get_person_box(person))
                #     boxes = np.array(boxes)
                #     boxes[:, 0] -= 540
                #     boxes[:, 1] -= 540
                #     dist = np.sqrt(np.sum(boxes[:, :2]**2, axis=1))
                #     dist_min = np.argmin(dist)
                #     area = boxes[:, 2] * boxes[:, 3]
                #     area_ax = np.argmax(area)
                #     assert area_ax == dist_min
                #     sample['people'] = [sample['people'][dist_min]]

            # path = os.path.join(new_kps_dir, kp)
            # with open(path, 'w') as json_file:
            #     json.dump(sample, json_file)




