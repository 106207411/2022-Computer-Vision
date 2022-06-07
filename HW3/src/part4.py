import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

np.random.seed(999)
random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # init feature detector/descriptor and matcher
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        p1, p2 = filter_matches(kp1, kp2, raw_matches)

        # TODO: 2. apply RANSAC to choose best H
        best_H = run_ransac(p1, p2)

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)

        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, im2.shape[0], 0, im2.shape[1], direction='b')

    return out

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    p1, p2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            p1.append(kp1[m.queryIdx].pt)
            p2.append(kp2[m.trainIdx].pt)

    return np.float32(p1), np.float32(p2)

def run_ransac(p1, p2):
    MAX_ITERS = 10000
    THRESHOLD = 5
    max_inliers = 0
    best_H = np.eye(3)
    for i in range(MAX_ITERS):
        # randomly sample 4 points
        idx = np.random.choice(len(p1), 4, replace=False)
        p1_rand, p2_rand = p1[idx], p2[idx]
        H = solve_homography(p2_rand, p1_rand)
        one_col = np.ones((p2.shape[0], 1))
        U = np.concatenate((p2, one_col), axis=1).T
        V = np.dot(H, U)
        V /= V[2]
        
        # calculate the error distance
        dist = np.linalg.norm(V[:2].T - p1, axis=1)
        n_inliers = np.sum([dist < THRESHOLD])
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_H = H

    return best_H


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)