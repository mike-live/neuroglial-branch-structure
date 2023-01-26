import numpy as np
import cv2
from time import time

def make_forest(img):
    mask = prepare_mask(img)
    skeleton, dist = make_skeleton(mask)
    return mask, skeleton, dist

def make_skeleton(mask):
    from skimage.morphology import medial_axis, skeletonize
    from skimage import data
    
    cur = mask.copy()
    cur = cv2.GaussianBlur(cur, (7, 7), 1)
    
    # Compute the medial axis (skeleton) and the distance transform
    skel, distance = medial_axis(cur, return_distance=True)
    
    # Compare with other skeletonization algorithms
    #skeleton = skeletonize(cur)
    skeleton_lee = (skeletonize(cur, method='lee') > 0).astype('uint8')
    
    # Distance to the background for pixels of the skeleton
    dist_on_skel = distance * skeleton_lee
    return skeleton_lee, dist_on_skel


def prepare_mask(img):
    cur = np.log(img.astype('float64'))
    cur -= cur.min()
    mx = 2 ** 16 - 1
    cur *= mx / np.quantile(cur.flatten(), 0.99)
    cur = np.minimum(cur, mx)
    cur = cur.astype('uint16')
    cur = cv2.adaptiveThreshold((cur / 256).astype('uint8'), maxValue = 1, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, \
                          thresholdType = cv2.THRESH_BINARY, blockSize = 41, C = -1)
    cur = remove_small(cur, min_size = 50)
    cur = remove_alone(cur, num_iter = 20)
    cur = remove_small(cur, min_size = 300)
    
    return cur

def remove_small(data, min_size = 30):
    mask = (data > 0).astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img = np.zeros((output.shape))
    
    size_img = sizes[output - 1]
    img[size_img >= min_size] = data[size_img >= min_size]
    '''for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img[output == i + 1] = data[output == i + 1]'''
    return img

def remove_alone(img, num_iter = 1, num_neighbors = 4):
    res = img.copy()
    prv = None
    for i in range(num_iter):
        mask = (res > 0).astype('uint8')
        if not prv is None:
            if np.all(prv == mask):
                break
        kernel = np.ones((3, 3), 'uint8')
        neighbors = cv2.filter2D(mask, -1, kernel)
        res[neighbors <= num_neighbors] = 0
        prv = mask.copy()
    return res

def fill_holes(src, hole_size):
    cur = 255 * (src == 0).astype('uint8')

    des = cv2.bitwise_not(cur)
    contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)

    cur = des #cv2.bitwise_not(des)
    return cur