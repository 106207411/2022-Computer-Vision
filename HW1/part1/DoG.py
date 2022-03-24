import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image) -> np.ndarray:
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        gaussian_images = self.get_gaussian_images(image)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        Dog_images = self.get_DoG_images(gaussian_images)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        keypoints = []
        for num_octave in range(self.num_octaves):
            for num_group in range(self.num_DoG_images_per_octave - 2):

                DoG1, DoG2, DoG3 = Dog_images[num_octave*self.num_DoG_images_per_octave+num_group:
                                              num_octave*self.num_DoG_images_per_octave+num_group+3]

                # Loop through each pixel in the image from (1, 1) to (width-2, height-2)
                for r in range(1, DoG2.shape[0] - 1):

                    for c in range(1, DoG2.shape[1] - 1):
                        # Get middle value of the 3x3 window
                        middle = DoG2[r, c]
                        
                        # Keep local extremum as a keypoint if the value is larger than threshold
                        if abs(middle) > self.threshold:
                            # Find 26 neighbors of the middle pixel
                            neighbors = np.concatenate([DoG1[r-1:r+2, c-1:c+2], DoG2[r-1:r+2, c-1:c+2], DoG3[r-1:r+2, c-1:c+2]]).flatten()

                            # Check if the middle value is a local extremum
                            if middle >= neighbors.max() or middle <= neighbors.min():
                                scale = 1 / 0.5**num_octave
                                keypoints.append([r*scale, c*scale])  

        # Step 4: Delete duplicate keypoints
        keypoints = np.unique(keypoints, axis=0).astype(int)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints

    def get_gaussian_images(self, image) -> list:
        gaussian_images = []
        for num_octave in range(self.num_octaves):

            # resize the first image in second and higher octave by 50%**octave
            base_image = cv2.resize(gaussian_images[-1], (0, 0), fx=0.5**num_octave, fy=0.5**num_octave, interpolation=cv2.INTER_NEAREST) if num_octave > 0 else image
            gaussian_images.append(base_image)

            # create gaussian images
            for num_guassian_image in range(1, self.num_guassian_images_per_octave):
                sigma = self.sigma**(num_guassian_image)
                # kernel (0, 0) is set to make the size of the kernel automatically calculated by the sigma value
                gaussian_image = cv2.GaussianBlur(base_image, (0, 0), sigma)
                gaussian_images.append(gaussian_image)

        return gaussian_images

    def get_DoG_images(self, gaussian_images) -> list:
        DoG_images = []
        for num_octave in range(self.num_octaves):

            for num_DoG_image in range(self.num_DoG_images_per_octave):
                # substrat the second image (more blurred one) to the first image (less blurred one)
                second_image = gaussian_images[num_octave * self.num_guassian_images_per_octave + num_DoG_image + 1]
                first_image = gaussian_images[num_octave * self.num_guassian_images_per_octave + num_DoG_image]
                DoG_image = cv2.subtract(second_image, first_image)
                DoG_images.append(DoG_image)

        return DoG_images