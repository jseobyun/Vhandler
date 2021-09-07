import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import cv2
import numpy as np
import moviepy.editor as mp
import imageio
from PIL import Image
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.video.fx.all as vfx

class VideoHandler():
    def __init__(self, root):
        self.root = root

    def scale_fps(self, video_name, scale):
        path = os.path.join(self.root, video_name)
        clip = mp.VideoFileClip(path)
        clip_fps_scaled = clip.set_fps(clip.fps*scale)
        clip_fps_scaled = clip_fps_scaled.fx(vfx.speedx, scale)
        save_name = video_name[:video_name.find('.')] + '_fps_scaled' + video_name[video_name.find('.'):]
        save_path = os.path.join(self.root, save_name)
        clip_fps_scaled.write_videofile(save_path)


    def resize_abs(self, video_name, img_size, save=True): # [w, h]
        print(f"resizing vidoe to size {img_size}")
        path = os.path.join(self.root, video_name)
        clip = mp.VideoFileClip(path)
        clip_resized = clip.resize(height=img_size[1], width=img_size[0])
        save_name = video_name[:video_name.find('.')] +'_resized' +video_name[video_name.find('.'):]
        save_path = os.path.join(self.root, save_name)
        clip_resized.write_videofile(save_path)
        return clip_resized

    def resize_rel(self, video_name, scale, save=True):
        print(f"resizing vidoe with scale {scale}")
        path = os.path.join(self.root, video_name)
        clip = mp.VideoFileClip(path)
        video_dim = clip.size
        clip_resized = clip.resize(height=int(video_dim[1] * scale), width=int(video_dim[0] * scale))
        save_name = video_name[:video_name.find('.')] + '_resized' + video_name[video_name.find('.'):]
        save_path = os.path.join(self.root, save_name)
        clip_resized.write_videofile(save_path)
        return clip_resized

    def crop(self, video_name, box):
        x, y, w, h = box[0], box[1], box[2], box[3]
        print(f"cropping vidoe with box {box}")
        path = os.path.join(self.root, video_name)
        clip = mp.VideoFileClip(path)
        clip = clip.crop(x1=x, y1=y, x2=x+w, y2=y+h)
        save_name = video_name[:video_name.find('.')] + '_cropped' + video_name[video_name.find('.'):]
        save_path = os.path.join(self.root, save_name)
        clip.write_videofile(save_path)

    def cut_time(self, video_name, start, end):
        print(f"cutting vidoe : from {start} ~ {end}")
        path = os.path.join(self.root, video_name)
        save_name = video_name[:video_name.find('.')] + '_cut' + video_name[video_name.find('.'):]
        save_path = os.path.join(self.root, save_name)
        ffmpeg_extract_subclip(path, start, end, save_path)

    def video_to_gif(self, video_name, start=None, end=None):
        print(f"video to gif : from {start} ~ {end} sec")
        path = os.path.join(self.root, video_name)
        clip = mp.VideoFileClip(path)
        if start is None:
            start = 0
        if end is None or end > clip.duration:
            end = -1
        clip = clip.subclip(start, end)
        save_name = video_name[:video_name.find('.')] + '.gif'
        save_path = os.path.join(self.root, save_name)
        clip.write_gif(save_path)

    def images_to_gif(self, folder_name, fps=1):
        folder_path = os.path.join(self.root, folder_name)
        file_names = os.listdir(folder_path)
        #file_names = sorted(file_names)
        file_names = [str(i) + '.png' for i in range(43)]
        paths = [os.path.join(self.root, folder_name, file_name) for file_name in file_names]
        imgs = [Image.open(path) for path in paths]

        # for i in range(len(imgs)):
        #     img = imgs[i]
        #     cv2.imshow('tmp',img)
        #     cv2.waitKey(500)
        save_path = os.path.join(self.root, folder_name+'.gif')
        imageio.mimsave(save_path, imgs, fps=fps)

    def gif_to_images(self, gif_name):
        gif_path = os.path.join(self.root, gif_name)
        save_dir = os.path.join(self.root, gif_name[:-4])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img = Image.open(gif_path)
        def iter_frames(img):
            try:
                i = 0
                while 1:
                    img.seek(i)
                    canvas = img.copy()
                    if i == 0:
                        palette = canvas.getpalette()
                    else:
                        canvas.putpalette(palette)
                    yield canvas
                    i += 1
            except EOFError:
                pass

        for i, frame in enumerate(iter_frames(img)):
            frame.save(os.path.join(save_dir, f'{i:05d}.png'), **frame.info)


    def images_to_video(self, folder_name, fps=25):
        print("images to video...")
        path = os.path.join(self.root, folder_name)
        image_list = os.listdir(path)
        ids = [int(image_name.split('_')[0]) for image_name in image_list]
        sort_idx = np.argsort(ids)
        image_list = np.array(image_list)[sort_idx].tolist()
        image_list =[os.path.join(path, image_name) for image_name in image_list]

        # for i in tqdm(range(len(image_list))):
        #     image_list[i] = os.path.join(path, image_list[i])

        save_path = os.path.join(self.root, folder_name+'.mp4')
        clip = mp.ImageSequenceClip(image_list, fps=fps)
        clip.write_videofile(save_path)

    # def images_to_video(self, folder_name, fps=25):
    #     ref_w = 1070
    #     ref_h = 802
    #     path = os.path.join(self.root, folder_name)
    #     save_path = os.path.join(self.root, folder_name + '.mp4')
    #     image_list = os.listdir(path)
    #     image_list = sorted(image_list)
    #     image_list =[os.path.join(path, image_name)for image_name in image_list]
    #
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(save_path, fourcc, fps, (ref_w, ref_h))
    #
    #     for image_path in image_list:
    #         img = cv2.imread(image_path)
    #         img = cv2.resize(img, (ref_w, ref_h))
    #         print(np.shape(img))
    #         out.write(img)
    #
    #     out.release()



    def video_to_images(self, video_name):
        print("video to images...")
        path = os.path.join(self.root, video_name)
        clip = mp.VideoFileClip(path)
        folder_name = video_name[:video_name.find('.')]
        folder_path = os.path.join(self.root, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for i, frame in tqdm(enumerate(clip.iter_frames())):
            frame_name = format(i,'06d') +'.png'
            frame_path = os.path.join(folder_path, frame_name)
            frame = frame[:,:,::-1]
            cv2.imwrite(frame_path, frame)

    def concat_videos(self, video_name_list):
        print("concat videos...")
        paths = [os.path.join(self.root, video_name) for video_name in video_name_list]
        clips = [mp.VideoFileClip(path) for path in paths]
        concat_clip = mp.concatenate_videoclips(clips)
        save_name = video_name_list[0][:video_name_list[0].find('.')] + '_cat' + video_name_list[0][video_name_list[0].find('.'):]
        save_path = os.path.join(self.root, save_name)
        concat_clip.write_videofile(save_path)
        return concat_clip

    def png2jpg(self, folder_name):
        folder_path = os.path.join(self.root, folder_name)
        file_names = os.listdir(folder_path)
        file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]

        for file_path in file_paths:
            img = Image.open(file_path)
            img = img.convert("RGB")
            img.save(file_path[:-4] +'.jpg')

    def jpg2jpg(self, folder_name):
        folder_path = os.path.join(self.root, folder_name)
        file_names = os.listdir(folder_path)
        file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]

        for file_path in file_paths:
            img = Image.open(file_path)
            img = img.convert("RGB")
            img.save(file_path[:-4] + '.jpg')



if __name__ == '__main__':
    vh = VideoHandler(root='/home/user/Desktop/yjs/codes/smplify-x/MeshFit/data/masks' )
    # vh.jpg2jpg(folder_name='cube_jpg')
    #vh.images_to_gif(folder_name='cat', fps=1)
    #vh.images_to_video(folder_name='view1', fps=30)
    vh.gif_to_images(gif_name='view1.gif')
    #vh.concat_videos(['demo_first.mp4', 'demo_second.mp4', 'demo_third.mp4', 'demo_fourth.mp4'])
    # vh.cut_time('smplx.mp4', start=2, end=60)
    # vh.scale_fps('smplx_cut.mp4', scale=1.5)
    # vh.resize_abs('motion_example_cut.mp4', img_size=(1366, 768))
    #vh.resize_rel('vis.mp4', scale=0.5)
    #vh.video_to_gif('hand_demo_resized.mp4', start=10, end=55)
    #vh.images_to_video('vis', fps=10)
    #vh.crop('hand_demo_resized.mp4', box=[480, 0, 480, 270])
    #vh.video_to_images('me.mp4')




