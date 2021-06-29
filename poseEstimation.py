import numpy as np
import cv2 as cv
import glob
import pickle
import matplotlib.pyplot as plt
import pdb

#Used for plotting Equal Axis
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Chessboard Detection termination criteria
# Stops if the specified accuracy (epsilon) is met
# Stops if the max number of iterations is exceededd
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
squareSZ = 23.876 #Square edge length in mm
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*squareSZ

objpoints = [] # 3d point in real world space (Known)
imgpoints = [] # 2d points in image plane (From Image)

# Plots the object points
fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(projection='3d')
ax.scatter(objp[:,0],objp[:,1],objp[:,2])

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

for i in np.arange(0,objp.shape[0]):
	ax.text(objp[i,0],objp[i,1],objp[i,2],i)

camMat = np.array(pickle.load(open('cameraMatrix_GoPro.sav','rb'))) #Loads the camera matrix fromt the calibration
distCoeffs = np.array(pickle.load(open('distCoeff_GoPro.sav','rb')))

print(camMat)
print(distCoeffs)

img = cv.imread('goProCal3/GOPR0224.JPG') #Loads the specified image into opencv
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #Changes from BGR colorspace to Gray colorspace
ret, corners = cv.findChessboardCorners(gray, (9,6), None) # Find the chess board corners

#Checks if the chessBoard Pattern was found
if ret == True:
	objpoints.append(objp)

	#Iterates to find the sub-pixel accurate location of corners or radial saddle points
	corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
	imgpoints.append(corners)

	# Draw and display the corners
	cv.drawChessboardCorners(img, (9,6), corners2, ret)
	cv.imshow('img', img)
	cv.waitKey(500)

#cv.destroyAllWindows()

objpoints=np.squeeze(np.array(objpoints))
imgpoints=np.squeeze(np.array(imgpoints))

print("The shape of the imgpoints array is: ", imgpoints.shape)
print("The shape of the objpoints array is: ", objpoints.shape)

# Solves for the rotation and translations between the world frame (objpoints) and camera frame (imgpoints)
retval, rvecs, tvecs = cv.solvePnP(objpoints,imgpoints,camMat,distCoeffs)

#Projects image points into the frame of the 
objPointsProj, jac = cv.projectPoints(objpoints, rvecs, tvecs, camMat, distCoeffs)

#Converts rotation vector into a rotation matrix
rotMat = cv.Rodrigues(rvecs)[0]; 
#To display the camera location in world coordinates we need the inverse
invrotMat = np.linalg.inv(rotMat); 

invtvec = -np.matmul(invrotMat,tvecs)

ax.scatter(invtvec[0],invtvec[1],invtvec[2])
set_axes_equal(ax)

#print("The Rotation Matrix is: ",rotMat)
#print("The Translation Vector is: ", tvecs, " and has a shape of: ",tvecs.shape)
rot_trans_objpoints = np.transpose(tvecs + np.matmul(rotMat,np.transpose(objpoints)))


fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(rot_trans_objpoints[:,0],rot_trans_objpoints[:,1],rot_trans_objpoints[:,2])
ax1.scatter(0,0,0)
ax1.set_xlabel('x-axis')
ax1.set_ylabel('y-axis')
ax1.set_zlabel('z-axis')
set_axes_equal(ax1)
plt.show()
