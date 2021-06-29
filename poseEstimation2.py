import numpy as np
import cv2 as cv
import glob
import pickle
import matplotlib.pyplot as plt
import pdb

# Function which takes corners in the chessboard's first corner and uses it to draw a coordinate axis
def draw(img, corners, imgpts):
    corn = np.array(tuple(corners[0].ravel())).astype(int)
    imgPT = np.array(tuple(imgpts[0].ravel())).astype(int)
    img = cv.line(img, corn, imgPT, (255,0,0), 5)

    imgPT = np.array(tuple(imgpts[1].ravel())).astype(int)
    img = cv.line(img, corn, imgPT, (0,255,0), 5)

    imgPT = np.array(tuple(imgpts[2].ravel())).astype(int)
    img = cv.line(img, corn, imgPT, (0,0,255), 5)
    return img

# Load the previous 
squareSZ = 23.876 #Square edge length in mm
camMat = np.array(pickle.load(open('cameraMatrix_GoPro.sav','rb'))) #Loads the camera matrix fromt the calibration
distCoeffs = np.array(pickle.load(open('distCoeff_GoPro.sav','rb')))

# The coordinate axis
axis = squareSZ*np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


# Chessboard Detection termination criteria
# Stops if the specified accuracy (epsilon) is met
# Stops if the max number of iterations is exceededd
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSZ = np.float32(23.876) #Square edge length in mm
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*squareSZ

objpoints = [] # 3d point in real world space (Known)
imgpoints = [] # 2d points in image plane (From Image)

fname = 'goProCal3/GOPR0224.JPG'
img = cv.imread(fname) #Loads the specified image into opencv
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #Changes from BGR colorspace to Gray colorspace
ret, corners = cv.findChessboardCorners(gray, (9,6), None) # Find the chess board corners

#Checks if the chessBoard Pattern was found and if so it refines the corners
if ret == True:
	#Iterates to find the sub-pixel accurate location of corners or radial saddle points
	corners2 = cv.cornerSubPix(gray,corners, winSize=(11,11), zeroZone=(-1,-1), criteria=criteria)

    # Find the rotation and translation vectors.
	ret,rvecs, tvecs = cv.solvePnP(objp, corners2, camMat, distCoeffs)

	# project 3D points to image plane
	imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, camMat, distCoeffs)
    
	img = draw(img,corners2,imgpts)
	plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
	plt.show()
	print("waiting")

	cv.imwrite(fname[:6]+'.png', img)