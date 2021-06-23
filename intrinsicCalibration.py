import numpy as np
import cv2 as cv
import glob
import pickle
import matplotlib.pyplot as plt
import pdb

# termination criteria
# Stops if the specified accuracy (epsilon) is met
# Stops if the max number of iterations is exceededd
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSZ = 23.876 #Square edge length in mm
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*squareSZ

print(objp.shape)

# Plots the object points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(objp[:,0],objp[:,1],objp[:,2])

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

for i in np.arange(0,objp.shape[0]):
	ax.text(objp[i,0],objp[i,1],objp[i,2],i)

#plt.show()

#pdb.set_trace()
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#Gets all the jpg filenames in the callImgIphone directory
images = glob.glob('goProCal3/*.JPG') 

print('Loaded Images')

for fname in images:
	print('Analyzing File: ',fname)
	img = cv.imread(fname)
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #Changes from BGR colorspace to Gray colorspace

	# Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, (9,6), None)

	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp)

		#Iterates to find the sub-pixel accurate location of corners or radial saddle points
		corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)

		# Draw and display the corners
		cv.drawChessboardCorners(img, (9,6), corners2, ret)
		cv.imshow('img', img)
		cv.waitKey(500)

cv.destroyAllWindows()

# Uses object points and image points to determine the camera matrix, distortion coefficients, rotation and translation vectors etc
# ret: The return value tells whether calibration was successful or not
# Mtx: This is the camera matrix it contains the output 3x3 intrinsic camera matrix
# dist: vector of the distortion coefficients
# rvecs: For each image this gives the rotations to go from the object coordinate space to the camera coordinate space
# tvecs: For each image this gives the translations  '      '
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
rvecs = np.array(rvecs)
tvecs = np.array(tvecs)
print(rvecs.shape)
print(tvecs.shape)
#pdb.set_trace()

# Saves the Camera Matrix from Calibration
filename = "cameraMatrix_GoPro.sav"
pickle.dump(mtx,open(filename,'wb'))

# Tests the calibration on a new image
img = cv.imread('GOPR01.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort the test image and save it 
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

# Determine the error in the re-projection using arithmetic mean
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )