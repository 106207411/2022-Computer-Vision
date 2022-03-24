import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    # read config file
    lines = open(args.setting_path, 'r').readlines()
    sigma_s = int(lines[6].split(',')[1])
    sigma_r = float(lines[6].split(',')[3])
    for line in lines[1:6]:
        r, g, b = [float(i) for i in line.split(',')]
        print("r: %s, g: %s, b: %s" % (r, g, b))
        
        # convert to gray scale image according to the config file
        img_gray = r * img_rgb[:, :, 0] + g * img_rgb[:, :, 1] + b * img_rgb[:, :, 2]
        JBF = Joint_bilateral_filter(sigma_s, sigma_r)
        bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)

        # compute error between bf and jbf
        error = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
        print("error: %d" % error)

        # save jbf and img_gray
        cv2.imwrite(f'jbf_{r}_{g}_{b}.png', cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'img_gray_{r}_{g}_{b}.png', img_gray)

    # print("cv2.gr")
    # JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    # bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    # jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)

    # # compute error between bf and jbf
    # error = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
    # print("error: %d" % error)


if __name__ == '__main__':
    main()