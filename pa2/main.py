import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

K_path = 'data/K.txt'
data_path = ["data/0000.jpg", "data/0003.jpg", "data/0018.jpg"]
#########################################################
# T0
def load_data(data_path, K_path):
    K = np.loadtxt(K_path)
    images = []
    for path in data_path:
        img = cv2.imread(path)
        images.append(img)
    return images, K

#########################################################
# T1
def sift_matching(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    ratio = 0.75
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    visualize_matches(img1, img2, kp1, kp2, good)

    pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good])

    return pts1, pts2

def visualize_matches(img1, img2, kp1, kp2, good):
    sift_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(sift_matches); plt.show()
    result_path = "result"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    cv2.imwrite(os.path.join(result_path, "matches.png"), sift_matches)

#########################################################
# T2
def essential_matrix(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, 
                                   method=cv2.RANSAC, 
                                   prob=0.999, 
                                   threshold=1.0)
    return E, mask

#########################################################
# T3
def pose_disambiguation(E, pts1, pts2, K):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

#########################################################
# T4
def triangulation(R, t, kp1, kp2, good, K):
    pass

def main():
    images, K = load_data(data_path, K_path)
    pts1, pts2 = sift_matching(images[0], images[1])
    E, _ = essential_matrix(pts1, pts2, K)
    R, t = pose_disambiguation(E, pts1, pts2, K)

if __name__ == "__main__":
    main()