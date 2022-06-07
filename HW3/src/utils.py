import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    ux = u[:,0].reshape((N,1))
    uy = u[:,1].reshape((N,1))
    vx = v[:,0].reshape((N,1))
    vy = v[:,1].reshape((N,1))
    vx_eq = np.concatenate((ux, uy, np.ones((N,1)), np.zeros((N,3)), -np.multiply(ux,vx), -np.multiply(uy,vx), -vx), axis=1)
    vy_eq = np.concatenate((np.zeros((N,3)), ux, uy, np.ones((N,1)), -np.multiply(ux,vy), -np.multiply(uy,vy), -vy), axis=1)
    A = np.stack((vx_eq, vy_eq), axis=1).reshape((2*N,9))

    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A)
    H = VT[-1,:].reshape((3,3))
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    
    if direction == 'b':
        # TODO: 1.meshgrid the (x,y) coordinate pairs
        x, y = np.arange(xmin, w_dst), np.arange(ymin, h_dst)
        xx, yy = np.meshgrid(x, y)

        # TODO: 2.reshape the destination pixels as 3 x N homogeneous coordinate
        vx_row = xx.reshape((1, xx.size))
        vy_row = yy.reshape((1, yy.size))
        one_row = np.ones((1, xx.size))
        V = np.concatenate((vx_row, vy_row, one_row), axis=0)

        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        H_inv = np.linalg.inv(H)
        U = np.dot(H_inv, V)
        U /= U[2]
        ux = np.round(U[0]).reshape((h_dst-ymin, w_dst-xmin)).astype(np.int)
        uy = np.round(U[1]).reshape((h_dst-ymin, w_dst-xmin)).astype(np.int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # TODO: 6. assign to destination image with proper masking
        mask = np.logical_and(np.logical_and(ux >= 0, ux < w_src), np.logical_and(uy >= 0, uy < h_src))
        dst[yy[mask], xx[mask]] = src[uy[mask], ux[mask]]

    elif direction == 'f':
        # TODO: 1.meshgrid the (x,y) coordinate pairs
        x, y = np.arange(xmin, xmax), np.arange(ymin, ymax)
        xx, yy = np.meshgrid(x, y)

        # TODO: 2.reshape the source pixels as 3 x N homogeneous coordinate
        ux_row = xx.reshape((1, xx.size))
        uy_row = yy.reshape((1, yy.size))
        one_row = np.ones((1, xx.size))
        U = np.concatenate((ux_row, uy_row, one_row), axis=0)

        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H, U)
        V /= V[2]
        vx = np.round(V[0].reshape((ymax-ymin, xmax-xmin))).astype(np.int)
        vy = np.round(V[1].reshape((ymax-ymin, xmax-xmin))).astype(np.int)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        # TODO: 5.filter the valid coordinates using previous obtained mask
        # TODO: 6. assign to destination image using advanced array indicing
        mask = np.logical_and(np.logical_and(vx >= 0, vx < w_dst), np.logical_and(vy >= 0, vy < h_dst))
        dst[vy[mask], vx[mask]] = src[yy[mask], xx[mask]]

    return dst
