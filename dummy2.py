import os
import cv2
import numpy as np

if __name__=='__main__':
    root_path = '/home/user/Desktop/yjs/data'
    img_dir = os.path.join(root_path, 'video_source_result')
    file_names = [f'{i}.jpg' for i in range(1500)]


    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(root_path, 'demo_fullbody.mp4'), fourcc, 24, (640, 480))

    for i in range(len(file_names)): # 120 180 216
        file_name = file_names[i]
        img_path = os.path.join(img_dir, file_name)
        if i % 4 ==0:
            img_wifi = cv2.imread(img_path)
        if i % 3 == 0:
            img_5g = cv2.imread(img_path)

        if np.shape(img_wifi)[0] != 480 or np.shape(img_wifi)[1] != 640:
            img_wifi = img_wifi[:480, :640, :]
        if np.shape(img_5g)[0] != 480 or np.shape(img_5g)[1] != 640:
            img_5g = img_5g[:480, :640, :]

        img_wifi = cv2.putText(img_wifi, "1.00x WiFi", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
        img_5g = cv2.putText(img_5g, "LABS Local 5G", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
        # zeros = np.zeros([480, 640, 1], dtype=np.uint8)
        # img_wifi2 = cv2.cvtColor(img_wifi, cv2.COLOR_BGR2GRAY)
        # #img_5g2 = cv2.cvtColor(img_5g, cv2.COLOR_BGR2GRAY)
        #
        # img_wifi2 = np.concatenate([img_wifi2[:,:,None], zeros, zeros], axis=2)
        # #img_5g2 = np.concatenate([zeros, zeros, img_5g2[:, :, None]], axis=2)

        img_concat = np.concatenate([img_wifi, img_5g], axis=1)
        #img_overlay = cv2.addWeighted(img_5g, 0.7, img_wifi2, 0.3, 0)
        out.write(img_5g)


    out.release()





