import numpy as np
import cv2

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    tiny_images = []
    for image_path in image_paths:
        tiny_img = cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
        tiny_img = tiny_img.flatten()
        # mean, norm = np.mean(tiny_img), np.linalg.norm(tiny_img)
        # tiny_img = (tiny_img - mean) / norm
        tiny_images.append(tiny_img)

    return np.matrix(tiny_images)