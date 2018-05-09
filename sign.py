import cv2
import numpy as np
import time

def reject_outliers(points,threshold_x, threshold_y):
    one_point = points[0]
    points = np.array(points)
######################
    threshold_init = 2.2
    u1 = np.mean(points[:,0])
    s1 = np.std(points[:,0])
    filtered1 = [point for point in points if (u1 - threshold_init* s1 <= point[0] <= u1 + threshold_init * s1)]
    points = np.array(filtered1)

    if len(points) == 0:
        return [one_point]
    
    u2 = np.mean(points[:,1])
    s2 = np.std(points[:,1])
    
    filtered2 = [point for point in points if (u2 - threshold_init * s2 <= point[1] <= u2 + threshold_init * s2)]
    
    if len(filtered2) == 0:
        return [one_point]
#####################
    u1 = np.mean(points[:,0])
    s1 = np.std(points[:,0])
    filtered1 = [point for point in points if (u1 - threshold_x * s1 <= point[0] <= u1 + threshold_x * s1)]
    points = np.array(filtered1)

    if len(points) == 0:
        return [one_point]
    
    u2 = np.mean(points[:,1])
    s2 = np.std(points[:,1])
    
    filtered2 = [point for point in points if (u2 - threshold_y * s2 <= point[1] <= u2 + threshold_y * s2)]
    
    if len(filtered2) == 0:
        return [one_point]

    return filtered2

def sign_detect(matches, case, face, kp_case, kp, frame,frame_color):
    matchesMask = [[0,0] for i in range(len(matches))]
    is_detected = 0
    match_points = []
    for i,(m,n) in enumerate(matches):
        
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            
            match_points.append([kp[m.trainIdx].pt[0],kp[m.trainIdx].pt[1]])
            
            is_detected += 1
    draw_params = dict(matchColor = (0,0,255),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = 0)
    img_show = cv2.drawMatchesKnn(face,kp_case,frame,kp,matches,None,**draw_params)


    #print("Time taken: ", (time.time()-init_time))
    #print(case + " : " + str(is_detected))
    if len(match_points) >=2:
        match_points = np.unique(match_points, axis=0)
    

    if len(match_points) >= 3:
        n_origin = len(match_points)
        
        match_points = reject_outliers(match_points,1.7,1.7)
        
        #print "filtered " + str(n_origin - len(match_points)) +"points"

    #print is_detected
    for [x,y] in match_points:
        cv2.circle(frame_color,(int(x),int(y)),5,(0,255,0),2)

    is_detected = len(match_points)
    print is_detected
    if is_detected >= 20 and case == "stop":
        print(case + " SIGN DETECTED : " + str(is_detected))
    if is_detected >= 10 and case == "uturn":
        print(case + " SIGN DETECTED : " + str(is_detected))
    if is_detected >= 15 and case == "parking":
        print(case + " SIGN DETECTED : " + str(is_detected))
    cv2.imshow(case,img_show)
    cv2.imshow(case + ' ',frame_color)

    if(is_detected>=5):

        k = cv2.waitKey(1000) & 0xFF

def findColor(hsv_image, lower, upper):
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask




face_stop = cv2.imread('./sign/stop.jpg',0)
face_parking = cv2.imread('./sign/parking.jpg',0)
face_uturn = cv2.imread('./sign/uturn.jpg',0)

surf = cv2.xfeatures2d.SURF_create(600)
surf_stop = cv2.xfeatures2d.SURF_create(800)
surf_parking = cv2.xfeatures2d.SURF_create(200)
surf_uturn = cv2.xfeatures2d.SURF_create(50)

kp_stop, des_stop = surf_stop.detectAndCompute(face_stop,None)
kp_parking, des_parking = surf_parking.detectAndCompute(face_parking,None)
kp_uturn, des_uturn = surf_uturn.detectAndCompute(face_uturn,None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

count = 1

while True:
    #init_time = time.time()
    path = '../sample/sign/parking4/' + str(count) +  '.jpg'
    frame = cv2.imread(path,0)    
    frame_color = cv2.imread(path,1)    

   
    #frame = frame[40:440, 300:700]
    #M = cv2.getRotationMatrix2D((200,200),90,1)
    #frame = cv2.warpAffine(frame,M,(400,400))

    # frame = cv2.GaussianBlur(frame, (5,5), 0)
    kp, des = surf.detectAndCompute(frame,None)
    

    matches_stop = flann.knnMatch(des_stop,des,k=2)
    matches_parking = flann.knnMatch(des_parking,des,k=2)
    matches_uturn = flann.knnMatch(des_uturn,des,k=2)
    # Need to draw only good matches, so create a mask
    
    #sign_detect(matches_stop, "stop", face_stop, kp_stop, kp, frame, frame_color)
    sign_detect(matches_parking, "parking", face_parking, kp_parking, kp, frame, frame_color)
    #sign_detect(matches_uturn, "uturn", face_uturn, kp_uturn, kp, frame, frame_color)


    #print("Time taken: ", (time.time()-init_time))
    
    k = cv2.waitKey(50) & 0xFF
    
    #break # For Debugging


    if k==27:
        break
    count = count + 1

cv2.destroyAllWindows()
# cap.release()
