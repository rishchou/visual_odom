# import sys,os

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pprint
from scipy.optimize import least_squares
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

def getCameraMatrix(path):
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(path)
    K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]])
    return K, LUT


def undistortImageToGray(img,LUT):
    colorimage = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(colorimage, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(gray1)
    return gray

def features(img1, img2):
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    pts1 = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts1, pts2




def skew(v):
    a=v[0]
    b=v[1]
    c=v[2]
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

def main():
    # sift = cv2.xfeatures2d.SIFT_create()
    BasePath = './stereo/centre/'
    K, LUT = getCameraMatrix('./model')
    images = []
    H1 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
    H1_calc = H1
    P0 = H1[:3]
    cam_pos = np.array([0,0,0])
    cam_pos = np.reshape(cam_pos,(1,3))
    test = os.listdir(BasePath)
    builtin = []
    linear = []
    for image in sorted(test):
       # print(image)
       images.append(image)

    print(len(images[:-2]))
    # cam_pos = np.zeros([1,2])
    # for file in range(len(images)-1):
    for img,_ in enumerate(images[:-2]):
        # print(img)
        img1 = cv2.imread("%s/%s"%(BasePath,images[img]),0)
        img2 = cv2.imread("%s/%s"%(BasePath,images[img+1]),0)
        und1 = undistortImageToGray(img1,LUT)
        und2 = undistortImageToGray(img2,LUT)

        pts1, pts2 = features(und1,und2)
        # print(pts1.shape)
        if pts1.shape[0] <5:
            continue

        F,_ = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        E,_ = cv2.findEssentialMat(pts1,pts2,focal =K[0][0], pp = (K[0,2],K[1,2]), method = cv2.RANSAC,prob=0.999,threshold=0.5)
        
        
        _,R_new,C_new,_=cv2.recoverPose(E, pts1, pts2, focal=K[0,0], pp=(K[0,2],K[1,2]))
       
        if np.linalg.det(R_new)<0:
            R_new = -R_new
            # C_new = -C_new
        # if np.linalg.det(R_calc)<0:
        #     R_calc = -R_calc
        H2 = np.hstack((R_new,np.matmul(-R_new,C_new)))
        H2 = np.vstack((H2,[0,0,0,1]))

        H1 = np.matmul(H1,H2)
       
        xpt = H1[0,3]
        zpt = H1[2,3]
        # builtin.append((xpt,zpt))
        print(img)
        
        plt.plot(-xpt,zpt,'.g')
        # plt.plot(xpt_calc,-zpt_calc,'.r')

        plt.pause(0.01)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

if __name__ == '__main__':
    main()
