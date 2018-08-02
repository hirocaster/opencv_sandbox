from imutils import paths
import argparse
import cv2
import os
import math
import numpy

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
                help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)


def report_image(image, laplacian, faces, face_laplacians=None):
    text = "Not Blurry"
    if laplacian.var() < args["threshold"]:
        text = "Blurry"

    if len(faces):
        for index, (x, y, w, h) in enumerate(faces):
             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
             cv2.putText(image, "{}: {:.2f}".format("Face", face_laplacians[index].var()), (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

    cv2.putText(image, "{}: {:.2f}".format(text, laplacian.var()), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    # cv2.imshow("Image", image)
    # key = cv2.waitKey(0)


def write_image(file_path, image, sub_dir="/report", suffix=""):
    dir_file = os.path.split(file_path)
    dir = dir_file[0]
    file_name = dir_file[1]
    report_dir = dir + sub_dir

    root, ext = os.path.splitext(report_dir + "/" + file_name)
    export_file_path = root + suffix + ext

    os.makedirs(report_dir, exist_ok=True)
    cv2.imwrite(export_file_path, image)

def face_recognition(gray):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

def eye_recognition(gray):
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    return eye_cascade.detectMultiScale(gray)

def crop_faces(gray, faces):
    return [gray[y: y + h, x: x + w] for x,y,w,h in faces]

def resize_image(image):
    height, width = image.shape[:2]
    while width >= 1500:
        image = resize_image_to_harf(image)
        height, width = image.shape[:2]
    else:
        return image

def resize_image_to_harf(image):
    return cv2.resize(image,None,fx=0.5, fy=0.5)

for image_path in paths.list_images(args["images"]):
    original_image = cv2.imread(image_path)
    image = resize_image(original_image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols, colors = image.shape

    hypot = int(math.hypot(rows, cols))
    frame = numpy.zeros((hypot, hypot), numpy.uint8)

    frame[math.ceil((hypot - rows) * 0.5):math.ceil((hypot + rows) * 0.5), math.ceil((hypot - cols) * 0.5):math.ceil((hypot + cols) * 0.5)] = gray

    for deg in range(-50, 51, 5):
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        faces = face_recognition(rotated)

        if len(faces):
            face_images = crop_faces(rotated, faces)
            for index, face_image in enumerate(face_images):
                write_image(image_path, face_image, "/faces", "_" + str(index) + str(deg))


        for (x, y, w, h) in faces:
            cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 0), 2)
            roi = rotated[y:y + h, x:x + w]
            eyes = eye_recognition(roi)
            eyes = list(filter(lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2, eyes))

            if len(eyes) == 2 and abs(eyes[0][0] - eyes[1][0]) > w / 4:
                score = math.atan2(abs(eyes[1][1] - eyes[0][1]), abs(eyes[1][0] - eyes[0][0]))
                if eyes[0][1] == eyes[1][1]:
                    score = 0.0

                cv2.putText(rotated, "{}: {:.2f}".format("score", score), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 2)

        write_image(image_path, rotated, "/rotated", str(deg))
