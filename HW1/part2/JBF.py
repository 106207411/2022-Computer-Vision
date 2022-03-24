import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
        
    def joint_bilateral_filter(self, img, guidance) -> np.ndarray:
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        # normalize the guidance image
        padded_guidance = padded_guidance / 255

        # compute the spatial kernel 
        spatial_kernel = self.compute_spatial_kernel(self.wndw_size, self.pad_w, self.sigma_s)
        output = np.zeros(img.shape)

        for r in range(self.pad_w, padded_guidance.shape[0]-self.pad_w):
            for c in range(self.pad_w, padded_guidance.shape[1]-self.pad_w):

                # compute the range kernel 
                center = padded_guidance[r, c]
                neighbors = padded_guidance[r - self.pad_w:r + self.pad_w + 1, c - self.pad_w:c + self.pad_w + 1]
                range_kernel = self.compute_range_kernel(center, neighbors, self.sigma_r)
                # compute the bilateral filter
                bilateral_filter = np.multiply(spatial_kernel, range_kernel)
                W = bilateral_filter.sum()
                # element-wise multiplication for multiple bands
                for band in range(img.shape[2]):
                    output[r - self.pad_w, c - self.pad_w, band] = (bilateral_filter * padded_img[r - self.pad_w:r + self.pad_w + 1, c - self.pad_w:c + self.pad_w + 1, band]).sum() / W
        
        return np.clip(output, 0, 255).astype(np.uint8)

    def compute_spatial_kernel(self, wndw_size, pad_w, sigma_s) -> np.ndarray:
        kernel = np.zeros((wndw_size, wndw_size))
        for r in range(wndw_size):
            for c in range(wndw_size):
                # L2 distance from the center pixel
                kernel[r, c] = np.exp(-((r-pad_w)**2+(c-pad_w)**2)/(2*sigma_s**2))
                
        return kernel

    def compute_range_kernel(self, center, neighbors, sigma_r) -> np.ndarray:
        # pixel different from the center pixel
        # single band
        if len(neighbors.shape) == 2:
            return np.exp(-np.square(center - neighbors) / (2*sigma_r**2))

        # 3 bands
        elif len(neighbors.shape) == 3:
            return np.exp(-np.square(center - neighbors).sum(axis=2) / (2*sigma_r**2))