import cv2
import numpy as np
import math

K = [[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]
K = np.reshape(K,(3,3))
K = K.T

lena = cv2.imread('Lena.png')
ref = cv2.imread('ref_marker.png')

ref_scale = 64
step = int(ref_scale/8)

def processTag(tag):
	qref = np.mean(tag[:step,:step])
		
	quadrants = {}
	count = 0
	for i in range(2*step,6*step,step):
		for j in range(2*step,6*step,step):
			quadrants[(np.mean(tag[i:i+step,j:j+step]))] = [count,i,j]
			count += 1

	keys = sorted(quadrants)
	keys = keys[3:]

	miss = [5,6,9,10]
	avg = 0
	min_white = 255
	for key in keys:
		data = quadrants[key]
		if data[0] not in miss:
			avg += key
			if min_white > key:
				min_white = key
			tag[data[1]:data[1]+step,data[2]:data[2]+step]=[255,255,255]
	avg /= 9

	#cv2.imshow('prerot',tag)
	q1 = np.mean(tag[2*step:3*step,2*step:3*step])
	q2 = np.mean(tag[2*step:3*step,5*step:6*step])
	q4 = np.mean(tag[5*step:6*step,5*step:6*step])
	q3 = np.mean(tag[5*step:6*step,2*step:3*step])
	rev = 0
	if q4>250:
		M = cv2.getRotationMatrix2D((ref_scale/2,ref_scale/2),0,1)
		tag = cv2.warpAffine(tag,M,(0,0))
	elif q3>250:
		M = cv2.getRotationMatrix2D((ref_scale/2,ref_scale/2),90,1)
		tag = cv2.warpAffine(tag,M,(0,0))
		rev = 90
	elif q1>250:
		M = cv2.getRotationMatrix2D((ref_scale/2,ref_scale/2),180,1)
		tag = cv2.warpAffine(tag,M,(0,0))
		rev = 180

	elif q2>250:
		M = cv2.getRotationMatrix2D((ref_scale/2,ref_scale/2),270,1)
		tag = cv2.warpAffine(tag,M,(0,0))
		rev = 270
	#cv2.imshow('postrot',tag)
	q6 = np.mean(tag[3*step:4*step,3*step:4*step])
	q10 = np.mean(tag[4*step:5*step,3*step:4*step])
	q7 = np.mean(tag[3*step:4*step,4*step:5*step])
	q11 = np.mean(tag[4*step:5*step,4*step:5*step])

	data = ''
	t = min_white
	if q10>=t:
		data += '1'
	else:
		data += '0'
	if q11>=t:
		data += '1'
	else:
		data += '0'
	if q7>=t:
		data += '1'
	else:
		data += '0'
	if q6>=t:
		data += '1'
	else:
		data += '0'
	
	#cv2.imshow('tag post rot',tag)
	tag = cv2.resize(lena,(ref_scale,ref_scale))
	M = cv2.getRotationMatrix2D((ref_scale/2,ref_scale/2),-rev,1)
	tag = cv2.warpAffine(tag,M,(0,0))

	return tag,data

def homography(points,last_points):
	stat = 1
	try:
		x1c = points[0,0,0]
		y1c = points[0,0,1]

		x2c = points[1,0,0]
		y2c = points[1,0,1]

		x3c = points[2,0,0]
		y3c = points[2,0,1]

		x4c = points[3,0,0]
		y4c = points[3,0,1]

		if len(points) > 4:
			fail = points[12,0,0]
		if cv2.contourArea(points) >  1658880:
			fail = points[12,0,0]
	except:
		x1c = last_points[0,0,0]
		y1c = last_points[0,0,1]
		
		x2c = last_points[1,0,0]
		y2c = last_points[1,0,1]
		
		x3c = last_points[2,0,0]
		y3c = last_points[2,0,1]

		x4c = last_points[3,0,0]
		y4c = last_points[3,0,1]
		stat = 0
		
	srcPoints = np.array([[x1c,y1c],[x2c,y2c],[x3c,y3c],[x4c,y4c]])
	dstPoints = np.array([[ref_scale,0],[ref_scale,ref_scale],[0,ref_scale],[0,0]])

	x1w = ref_scale
	y1w = 0

	x2w = ref_scale
	y2w = ref_scale

	x3w = 0
	y3w = ref_scale

	x4w = 0
	y4w = 0

	p1 = [x1w,y1w,0,1]
	p2 = [x2w,y2w,0,1]
	p3 = [x3w,y3w,0,1]
	p4 = [x4w,y4w,0,1]

	l1 = [x1w,y1w,1,0,0,0,-x1c*x1w,-x1c*y1w,-x1c]
	l2 = [0,0,0,x1w,y1w,1,-y1c*x1w,-y1c*y1w,-y1c]
	l3 = [x2w,y2w,1,0,0,0,-x2c*x2w,-x2c*y2w,-x2c]
	l4 = [0,0,0,x2w,y2w,1,-y2c*x2w,-y2c*y2w,-y2c]
	l5 = [x3w,y3w,1,0,0,0,-x3c*x3w,-x3c*y3w,-x3c]
	l6 = [0,0,0,x3w,y3w,1,-y3c*x3w,-y3c*y3w,-y3c]
	l7 = [x4w,y4w,1,0,0,0,-x4c*x4w,-x4c*y4w,-x4c]
	l8 = [0,0,0,x4w,y4w,1,-y4c*x4w,-y4c*y4w,-y4c]
	A = np.vstack([l1,l2,l3,l4,l5,l6,l7,l8])
	U,S,V = np.linalg.svd(A)
	V = V[8]
	h = V/V[8]
	H = np.reshape(h,(3,3))

	R = np.dot(np.linalg.inv(K),H)
	c1 = R[:,0]
	c2 = R[:,1]
	c3 = R[:,2]

	h1,h2,h3 = H

	h1 = h1.reshape(3,1)
	h2 = h2.reshape(3,1)
	one = np.dot(np.linalg.inv(K),h1)
	two = np.dot(np.linalg.inv(K),h2)
	top = (np.linalg.norm(c1)+np.linalg.norm(c2))
	lam = (top/2)**(-1)
	
	Bt = lam*R

	if np.linalg.det(Bt)<0:
		Bt = Bt*(-1)
	B = Bt

	b1,b2,b3 = B.T
	b1 = b1.reshape(1,3)
	b2 = b2.reshape(1,3)
	b3 = b3.reshape(1,3)

	r1 = b1
	r2 = b2
	r3 = np.cross(r1,r2)
	t = b3

	T = np.concatenate((r1.T,r2.T,r3.T,t.T),axis=1)
	P = np.matmul(K,T)

	return P,H,stat

def drawCube(img,g,P):
	w_point1 = [0,0,0,1]
	w_point2 = [0,g,0,1]
	w_point3 = [g,g,0,1]
	w_point4 = [g,0,0,1]

	w_point5 = [0,0,-g,1]
	w_point6 = [0,g,-g,1]
	w_point7 = [g,g,-g,1]
	w_point8 = [g,0,-g,1]

	offset = 0


	c_point1 = np.matmul(P,w_point1)
	c_point1 = (c_point1/c_point1[2]).astype(int)
	c_point1 = tuple([c_point1[0]+offset,c_point1[1]+offset])

	c_point2 = np.matmul(P,w_point2)
	c_point2 = (c_point2/c_point2[2]).astype(int)
	c_point2 = tuple([c_point2[0]+offset,c_point2[1]+offset])

	c_point3 = np.matmul(P,w_point3)
	c_point3 = (c_point3/c_point3[2]).astype(int)
	c_point3 = tuple([c_point3[0]+offset,c_point3[1]+offset])

	c_point4 = np.matmul(P,w_point4)
	c_point4 = (c_point4/c_point4[2]).astype(int)
	c_point4 = tuple([c_point4[0]+offset,c_point4[1]+offset])

	c_point5 = np.matmul(P,w_point5)
	c_point5 = (c_point5/(c_point5[2])).astype(int)
	c_point5 = tuple([c_point5[0]+offset,c_point5[1]+offset])

	c_point6 = np.dot(P,w_point6)
	c_point6 = (c_point6/(c_point6[2])).astype(int)
	c_point6 = tuple([c_point6[0]+offset,c_point6[1]+offset])

	c_point7 = np.dot(P,w_point7)
	c_point7 = (c_point7/(c_point7[2])).astype(int)
	c_point7 = tuple([c_point7[0]+offset,c_point7[1]+offset])

	c_point8 = np.dot(P,w_point8)
	c_point8 = (c_point8/(c_point8[2])).astype(int)
	c_point8 = tuple([c_point8[0]+offset,c_point8[1]+offset])

	#bottom
	cv2.line(img,c_point1,c_point2,(0,0,255),3)
	cv2.line(img,c_point2,c_point3,(0,0,255),3)
	cv2.line(img,c_point3,c_point4,(0,0,255),3)
	cv2.line(img,c_point4,c_point1,(0,0,255),3)

	#sides
	cv2.line(img,c_point1,c_point5,(255,0,0),3)
	cv2.line(img,c_point2,c_point6,(255,0,0),3)
	cv2.line(img,c_point3,c_point7,(255,0,0),3)
	cv2.line(img,c_point4,c_point8,(255,0,0),3)

	#top
	cv2.line(img,c_point5,c_point6,(0,255,0),3)
	cv2.line(img,c_point6,c_point7,(0,255,0),3)
	cv2.line(img,c_point7,c_point8,(0,255,0),3)
	cv2.line(img,c_point8,c_point5,(0,255,0),3)

def process_video_from_file(video_file):
	cap = cv2.VideoCapture(video_file)
	last_pnts = []
	previous_points = [[np.array([[[1,1]],[[1,1]],[[1,1]],[[1,1]]]),[510,334]],[np.array([[[1,1]],[[1,1]],[[1,1]],[[1,1]]]),[1000,500]],[np.array([[[1,1]],[[1,1]],[[1,1]],[[1,1]]]),[1349,365]]]
	width = 1920
	height = 1080
	size = (width,height)
	out = cv2.VideoWriter((video_file[:-4]+'output'+'.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
	while cap.isOpened():
		ret,img = cap.read()
		img_copy=img
		# cv2.imwrite('frame_grab.png',img)
		try:
			img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		except:
			print('The video is over')
			break

		# Setup SimpleBlobDetector parameters.
		params = cv2.SimpleBlobDetector_Params()

		# Change thresholds
		params.minThreshold = 50
		params.maxThreshold = 200
		# Filter by Color
		params.filterByColor = 1
		params.blobColor = 255
		# Filter by Area.
		params.filterByArea = True
		params.minArea = 10000
		params.maxArea = 100000000
		# Filter by Circularity
		params.filterByCircularity = True
		params.minCircularity = 0.6
		params.maxCircularity = 0.9
		# Filter by Convexity
		params.filterByConvexity = True
		params.minConvexity = 0.95
		params.maxConvexity = 1
		# Create a detector with the parameters
		detector = cv2.SimpleBlobDetector_create(params)
		# Detect blobs.
		keypoints = detector.detect(img_gray)
		im_with_keypoints = cv2.drawKeypoints(img_copy, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		#cv2.imshow('keypoints',im_with_keypoints)
		i = 0
		for page in keypoints:
			x = np.uint(page.pt[0])
			y = np.uint(page.pt[1])
			r = np.uint((page.size/2)*1.6)
			pt1 = [x-r,y-r]
			pt2 = [x-r,y+r]
			pt3 = [x+r,y+r]
			pt4 = [x+r,y-r]
			box = [np.array([pt1,pt2,pt3,pt4])]
			box = np.int32(box)
			stencil = np.zeros(img.shape).astype(img.dtype)
			black = [255,255,255]
			cv2.fillPoly(stencil,box,black)
			blob = cv2.bitwise_and(img,stencil)
			im_with_keypoints = cv2.drawKeypoints(img_copy, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			blob_gray = cv2.cvtColor(blob,cv2.COLOR_BGR2GRAY)
			ret,thresh = cv2.threshold(blob_gray,200,255,1)
			
			ret,contours,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.RETR_TREE)
			#thresh = cv2.drawContours(thresh,contours,-1,150,3)
			boundary = 0
			dexter = 0
			for c in contours:
				a = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
				if len(a) > 6:
					boundary = contours[dexter-1]
					break
				dexter+=1
			try:
				approx = cv2.approxPolyDP(boundary,0.01*cv2.arcLength(boundary,True),True)
			except:
				min_dist = 100000
				sec = 0
				counter = 0
				for run in previous_points:
					xp = np.float(run[1][0])
					yp = np.float(run[1][1])
					dist = ((x - xp)**2 + (y-yp)**2)**.5;
					if dist < min_dist:
						min_dist = dist
						sec = counter
					counter +=1
				approx = previous_points[sec][0]
				
			min_dist = 100000
			sec = 0
			counter = 0
			for run in previous_points:
				xp = np.float(run[1][0])
				yp = np.float(run[1][1])
				dist = ((x - xp)**2 + (y-yp)**2)**.5;
				if dist < min_dist:
					min_dist = dist
					sec = counter
				counter +=1
			last_pnts = previous_points[sec][0]
			[P,H,stat] = homography(approx,last_pnts)

			if stat == 1:
				previous_points[sec][0]=approx
				previous_points[sec][1][0]=x
				previous_points[sec][1][1]=y
			#show the contour
			

			try:
				blob = cv2.drawContours(blob,boundary,-1,(0,255,0),1)
				thresh = cv2.drawContours(thresh,boundary,-1,150,3)
			except:
				blob = cv2.drawContours(blob,contours,0,(0,255,0),1)
				#im_with_keypoints = cv2.drawContours(im_with_keypoints,contours,0,(0,255,0),3)
			#for p in approx:
				#cv2.circle(img,(p[0,0],p[0,1]),3,(255,0,0),-1)
			#cv2.imshow('blob',blob)
			
			#black out the data
			#cv2.imshow('thresh',thresh)

			index = np.where(blob_gray==200)
			blob_gray[index] = 201
			# x,y,w,h = cv2.boundingRect(contours[-2])
			# blob_gray_double = cv2.rectangle(blob_gray,(x,y),(x+w,y+h),200,-1)
			blob_gray_double = cv2.drawContours(blob_gray,contours,dexter-1,200,-1)
			# blob_gray_double = cv2.drawContours(blob_gray,contours,len(contours)-2,200,-1)

			# cv2.imshow('',blob_gray_double)
			
			# find points within the tag
			tag_pts = np.where(blob_gray_double == 200,)
			one = np.ones(len(tag_pts[0]))
			# create homogeonous coordinates
			tag_pts = np.stack((tag_pts[1],tag_pts[0],one))
			tag = np.zeros((ref_scale,ref_scale,3),np.uint8)
			# multiply for transformation
			trans = np.matmul(np.linalg.inv(H),tag_pts)
			# make homogenous
			trans = trans / trans[2,:]
			
			trans = trans.astype(int)
			tag_pts = tag_pts.astype(int)

			for i in range(len(tag_pts[1])):
				srcx = tag_pts[0][i]
				srcy = tag_pts[1][i]
				destx = trans[0][i]
				desty = trans[1][i]
				try:
					tag[destx,desty] = img[srcy,srcx]
				except:
					pass
			tag = tag[:ref_scale,:ref_scale]
			tag,data = processTag(tag)

			for i in range(len(tag_pts[1])):
				srcx = tag_pts[0][i]
				srcy = tag_pts[1][i]
				destx = trans[0][i]
				desty = trans[1][i]
				try:
					img[srcy,srcx] = tag[destx,desty]
				except:
					pass
			drawCube(img,ref_scale,P)

			font = cv2.FONT_HERSHEY_SIMPLEX
			try:
				points = approx[1,0]
				cv2.putText(img,data,(points[0]+200,points[1]),font,4,(0,0,255),5,cv2.LINE_AA)
			except:
				pass
		
			i+=1
		display = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
		cv2.imshow('final image',display)
		#cv2.imshow('keypoints',im_with_keypoints)
		if cv2.waitKey(1) & 0xFF == ord('p'):
			while cv2.waitKey(1) & 0xFF != ord('p'):
				out.write(img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

process_video_from_file('multipleTags.mp4')
# process_video_from_file('Tag0.mp4')