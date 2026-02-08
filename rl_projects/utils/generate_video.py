import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

def main(args):
    root = args.img_path
    video_path = args.save_name
    imgs = sorted(glob(root + '/*.jpg'))
    img = cv2.imread(imgs[0])
    size = (img.shape[1], img.shape[0])

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), int(args.fps), size)

    for img in tqdm(imgs):
        img = cv2.imread(img)
        out.write(img)

    out.release()

    new_filename = os.path.join(os.path.dirname(video_path), os.path.basename(video_path).split('.')[0] + '_convert.mp4')
    os.system(f'ffmpeg -i {video_path} -vcodec libx264 -acodec copy {new_filename}')
    os.system(f'rm -f {video_path}')
    print('video saved to {}'.format(new_filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save Video")
    parser.add_argument("--img_path")
    parser.add_argument("--save_name")
    parser.add_argument("--fps", default=10)
    args = parser.parse_args()
    main(args)
