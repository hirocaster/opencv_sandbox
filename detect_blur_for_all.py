from imutils import paths
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
                help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)


def report_image(image, laplacian, text):
    cv2.putText(image, "{}: {:.2f}".format(text, laplacian.var()), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    # cv2.imshow("Image", image)
    # key = cv2.waitKey(0)


def write_image(file_path, image, sub_dir="/report"):
    dir_file = os.path.split(file_path)
    dir = dir_file[0]
    file_name = dir_file[1]
    report_dir = dir + sub_dir

    os.makedirs(report_dir, exist_ok=True)

    cv2.imwrite(report_dir + "/" + file_name, image)


for image_path in paths.list_images(args["images"]):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = variance_of_laplacian(gray)

    text = "Not Blurry"
    if laplacian.var() < args["threshold"]:
        text = "Blurry"

    report_image(image, laplacian, text)
    write_image(image_path, image)
    write_image(image_path, laplacian, "/laplacian")
