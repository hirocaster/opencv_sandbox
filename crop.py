import face
import argparse
from imutils import paths
import cv2
import numpy as np
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
                help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

def write_image(file_path, image, sub_dir="/face", suffix=""):
    dir_file = os.path.split(file_path)
    dir = dir_file[0]
    file_name = dir_file[1]
    report_dir = dir + sub_dir

    root, ext = os.path.splitext(report_dir + "/" + file_name)
    export_file_path = root + suffix + ext

    os.makedirs(report_dir, exist_ok=True)
    cv2.imwrite(export_file_path, image)

for image_path in paths.list_images(args["images"]):
    print(image_path)
    img = cv2.imread(image_path)
    faces, hoge = face.detect(img)

    for index, f in enumerate(faces):
        frame = face.gray_in_frame(img)
        rotated = face.rotate(frame, img, f['deg'])

        y = int(f['y'])
        h = int(f['r_h'])
        x = int(f['x'])
        w = int(f['w'])
        y_offset = int(h * 0.1)

        croped = rotated[y + y_offset: y + h, x: x + w]
        write_image(image_path, croped, "/faces", "_" + str(index))
