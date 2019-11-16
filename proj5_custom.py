import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pprint
from scipy.optimize import least_squares
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
from random import sample


def getCameraMatrix(path):
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(path)
    K = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]])
    return K, LUT

def normalise(points):
    mean1 = np.mean(points[:, 0])
    mean2 = np.mean(points[:, 1])

    d = np.mean(np.sqrt((points[:, 0] - mean1) ** 2 + (points[:, 1] - mean2) ** 2))

    P = np.array([[1 / d, 0, -mean1 / d], [0, 1 / d, -mean2 / d], [0, 0, 1]])
    points = np.insert(points, 2, 1, axis=1)

    P1 = np.dot(P, points.T)
    P1 = P1.T

    return P1, P

def undistortImage(iter,LUT):
    colorimage = cv2.cvtColor(iter, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(colorimage, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(gray1)
    return gray

def keypoints(img1, img2):
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

def get_fundamental_matrix(x1, x2):
    x1, P1 = normalise(x1)
    x2, P2 = normalise(x2)

    n = x1.shape[0]

    if x2.shape[0] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[i, 0] * x2[i, 0], x1[i, 0] * x2[i, 1], x1[i, 0],
                x1[i, 1] * x2[i, 0], x1[i, 1] * x2[i, 1], x1[i, 1], x2[i, 0],
                x2[i, 1], 1]

    # compute linear least square solution

    U, S, V = np.linalg.svd(A)
    F = V[:, 8].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)

    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = np.dot(P1.T, np.dot(F, P2))

    return F / F[2, 2]

def compute_fundamental_ransac(X1, X2):
    # storing the randomly generated 8 correspondences for each iteration in lists
    x1_list = list()
    x2_list = list()

    # F is the best fundamental matrix
    F = np.zeros((3, 3))

    # F_list stores all the F's for various iterations of RANSAC
    F_list = list()
    err_list = list()
    err = list()
    l = X1.shape[0]
    indices = [i for i in range(l)]

    # defining the number of iterations of RANSAC
    M = 40

    # inlier_count_list = [0 for i in range(M)]
    index = 0
    current_inlier_list = list()
    prev_inlier_list = list()
    best_inlier = list()
    for i in range(0, M):
        current_inlier_list.clear()
        # defining random indexes(8 in total) to select 8 random (x,y) from x1 and x2
        indices_8 = sample(indices, 8)

        # choosing 8 random correspondences by giving the position of these points in x1 and
        # x2 using the list of 8 indices created in the prev step

        x1_list.insert(i - 1, X1[indices_8])
        x2_list.insert(i - 1, X2[indices_8])

        # computing the fundamental matrix for the randomly chosen set of x1 and x2 points
        F_list.insert(i - 1, get_fundamental_matrix(x1_list[i], x2_list[i]))

        for j in range(l):

            x1 = np.insert(X1[j], 2, 1, axis=0)
            x2 = np.insert(X2[j], 2, 1, axis=0)
            x1 = x1.reshape(3, 1)
            x2 = x2.reshape(3, 1)

            d1 = np.dot(F_list[i], x1)
            d2 = np.dot(F_list[i].T, x1)

            error = np.linalg.norm(
                np.abs(np.dot(x2.T, np.dot(F_list[i], x1))) / np.sqrt(np.dot(d1.T, d1) + np.dot(d2.T, d2)))

            if error < 0.999:
                current_inlier_list.append([x1.T, x2.T])

        if len(current_inlier_list) > len(prev_inlier_list):
            best_inlier = np.asarray(current_inlier_list)
            F = F_list[i]

        prev_inlier_list = current_inlier_list

    print("best -->", best_inlier.shape)
    x1_in = best_inlier[:, 0, :, :]
    x2_in = best_inlier[:, 1, :, :]

    x1_in = x1_in.reshape(x1_in.shape[0], x1_in.shape[2])
    x2_in = x2_in.reshape(x2_in.shape[0], x2_in.shape[2])

    # # x1 = np.delete(x1, 2, axis=0)
    # x2 = np.delete(x2, 2, axis=0)
    return F, x1_in, x2_in


def extract_camera_pose(E):
    poses = []
    W = np.array(([0,-1,0],[1,0,0],[0,0,1]))
    U,S,V = np.linalg.svd(E)
    
    C1 = -U[:,2].reshape(-1,1)
    C2 = U[:,2].reshape(-1,1)
    C3 = -U[:,2].reshape(-1,1)
    C4 =  U[:,2].reshape(-1,1)


    R1 = np.matmul(np.matmul(U,W),V)
    R2 = np.matmul(np.matmul(U,W),V)
    R3 = np.matmul(np.matmul(U,W.T),V)
    R4 = np.matmul(np.matmul(U,W.T),V)

    if np.linalg.det(R1)<0:
        R1=-R1
    
    if np.linalg.det(R2)<0:
        R2=-R2
    
    if np.linalg.det(R3)<0:
        R3=-R3
    
    if np.linalg.det(R4)<0:
        R4=-R4
    
    P1 = np.concatenate((R1,C1),axis = 1)
    poses.append(P1)
    P2 = np.concatenate((R2,C2),axis = 1)
    poses.append(P2)
    P3 = np.concatenate((R3,C3),axis = 1)
    poses.append(P3)
    P4 = np.concatenate((R4,C4),axis = 1)
    poses.append(P4)

    return poses

def skew(v):
    a=v[0]
    b=v[1]
    c=v[2]
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

def linear_triangulation(K,P0,P1,pt1,pt2):
    pt1 = np.insert(np.float32(pt1),2,1)
    pt2 = np.insert(np.float32(pt2),2,1)
    # print(pt1.shape)
    homo_pt1 = np.matmul(np.linalg.inv(K),pt1.reshape((-1,1)))
    homo_pt2 = np.matmul(np.linalg.inv(K),pt2.reshape((-1,1)))

    skew0 = skew(homo_pt1.reshape((-1,)))
    skew1 = skew(homo_pt2.reshape((-1,)))

    P0 = np.concatenate((P0[:,:3], -np.matmul(P0[:,:3],P0[:,3].reshape(-1,1))),axis=1)
    P1 = np.concatenate((P1[:,:3], -np.matmul(P1[:,:3],P1[:,3].reshape(-1,1))),axis=1)
    # P0 = homogeneousMat(P0)
    # P1 = homogeneousMat(P1)
    pose1 = np.matmul(skew0,P0[:3,:])
    pose2 = np.matmul(skew1,P1[:3,:])

    #Solve the equation Ax=0
    A = np.concatenate((pose1,pose2),axis=0)
    u,s,vt = np.linalg.svd(A)
    X = vt[-1]
    X = X/X[3]
    return X

def find_correct_pose(P0,poses, allPts):
    max = 0
    flag = False
    for i in range(4):
        P = poses[i]
        # print("Each"+str(i),P)
        r3 = P[:,3]
        r3 = np.reshape(r3,(1,3))
        C = P[:,3]
        C = np.reshape(C,(3,1))
        pts_list = allPts[i]
        pts = np.array(pts_list)
        pts = pts[:,0:3].T

        diff = np.subtract(pts,C)
        Z = np.matmul(r3,diff)
        Z = Z>0
        _,idx = np.where(Z==True)
        # print(idx.shape[0])
        if max < idx.shape[0]:
            poseid = i
            correctPose = P
            indices = idx
            max = idx.shape[0]
    if max==0:
        flag = True
        correctPose = None
    return correctPose,flag,poseid

def correspondence_matrix(p1, p2):
    
    A = np.empty((0,9), float)

    for X, Y in zip(p1, p2):
        p1x, p1y = X
        p2x, p2y = Y
        # print(p1x, p1y, p2x, p2y)
        row = np.array([[p1x * p2x, p1x * p2y, p1x, p1y * p2x, p1y * p2y, p1y, p2x, p2y, 1]])
        A = np.append(A, row, axis=0)   
    
    return A

def estimateFundamentalMatrix(x1, x2):
    """ Compute the fundamental or essential matrix from corresponding points
        (x1, x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
   
    F = np.dot(U, np.dot(np.diag(S), V))
    return F

def findEssentialMatrix(K,pts1, pts2):
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC, 1,0.90)

    u,s,vt = np.linalg.svd(F)
    s[2] = 0
    snew = np.diag(s)
    F = np.matmul(np.matmul(u,snew),vt)
    assert np.linalg.matrix_rank(F)==2,"Rank of F not 2"

    E = np.matmul(np.matmul(K.T,F),K)
    U,S,Vt = np.linalg.svd(E)
    S[0] = 1
    S[1] = 1
    S[2] = 0
    Snew = np.diag(S)
    E = np.matmul(U,np.matmul(Snew,Vt))
    return E

def main():

    BasePath = './stereo/centre/'
    K, LUT = getCameraMatrix('./model')
    images = []
    H1 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
    H1_custom = H1
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
    cam_pos = np.zeros([1,2])
    # for file in range(len(images)-1):
    for iter,_ in enumerate(images[:-2]):
        # print(iter)
        img1 = cv2.imread("%s/%s"%(BasePath,images[iter]),0)
        img2 = cv2.imread("%s/%s"%(BasePath,images[iter+1]),0)
        und1 = undistortImage(img1,LUT)
        und2 = undistortImage(img2,LUT)

        pts1, pts2 = keypoints(und1,und2)
        # print(pts1.shape)
        if pts1.shape[0] <8:
            continue

        F_custom, x1_in, x2_in = compute_fundamental_ransac(pts1, pts2)
        E_custom = findEssentialMatrix(K,pts1, pts2)
        poses = extract_camera_pose(E_custom)
        cumulative_points = dict()
        for j in range(4):
           X = []
           for i in range(len(pts1)):
               pt = linear_triangulation(K,P0,poses[j],pts1[i],pts2[i])
               X.append(pt)
           # print("Pose" + str(j))
           cumulative_points.update({j:X})
        correctPose,no_inlier,poseid = find_correct_pose(P0, poses, cumulative_points)
        R_custom = correctPose[:,:3].reshape(3,3)
        C_custom = correctPose[:,3].reshape(3,1)
        
        if np.linalg.det(R_custom)<0:
            R_custom = -R_custom
        
        H2_custom = np.hstack((R_custom,np.matmul(-R_custom,C_custom)))
        H2_custom = np.vstack((H2_custom,[0,0,0,1]))

        H1_custom = np.matmul(H1_custom,H2_custom)
        pos_custom = np.matmul(-H1_custom[:3,:3].T,H1_custom[:3,3].reshape(-1,1))
        
        print("Frame number: ",iter)
        
        x_camera = H1_custom[0,3]
        
        z_camera = H1_custom[2,3]
        
        plt.plot(x_camera,-z_camera,'.r')

        plt.pause(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
