from cv2 import KeyPoint
import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def plot_DOG_images(img_gray, DoG):
    DoG_images = DoG.get_DoG_images(DoG.get_gaussian_images(img_gray))
    for num_octave in range(DoG.num_octaves):
        for num_DoG_image in range(DoG.num_DoG_images_per_octave):
            DoG_image = DoG_images[num_octave*DoG.num_DoG_images_per_octave+num_DoG_image]

            # normalize DoG image to [0, 255]
            norm_DoG_image = 255*(DoG_image - DoG_image.min())/(DoG_image.max() - DoG_image.min())
            cv2.imwrite(f'DoG{num_octave+1}-{num_DoG_image+1}.png', norm_DoG_image)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)
    DoG = Difference_of_Gaussian(args.threshold)

    print('Plotting DOG images ...')
    plot_DOG_images(img, DoG)

    print('Plotting keypoints ...')
    keypoints = DoG.get_keypoints(img)
    plot_keypoints(img, keypoints, f'threshold-{args.threshold}.png')


if __name__ == '__main__':
    main()