import cv2
import numpy as numpy
import time

def sign_detect(matches, case, face, kp_case, kp, frame):
    matchesMask = [[0,0] for i in range(len(matches))]
    is_detected = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            is_detected += 1
    draw_params = dict(matchColor = (0,0,255),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = 0)
    img_show = cv2.drawMatchesKnn(face,kp_case,frame,kp,matches,None,**draw_params)
    #print("Time taken: ", (time.time()-init_time))
    print(case + " : " + str(is_detected))
    if is_detected >= 20 and case == "stop":
        print(case + " SIGN DETECTED : " + str(is_detected))
    if is_detected >= 10 and case == "uturn":
        print(case + " SIGN DETECTED : " + str(is_detected))
    if is_detected >= 15 and case == "parking":
        print(case + " SIGN DETECTED : " + str(is_detected))
    
    cv2.imshow(case,img_show)






face_stop = cv2.imread('./sign/stop.jpg',0)
face_parking = cv2.imread('./sign/parking.jpg',0)
face_uturn = cv2.imread('./sign/uturn.jpg',0)

surf = cv2.xfeatures2d.SURF_create(400)
surf_stop = cv2.xfeatures2d.SURF_create(800)
surf_parking = cv2.xfeatures2d.SURF_create(200)
surf_uturn = cv2.xfeatures2d.SURF_create(200)

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

count = 10

while True:
    #init_time = time.time()
    frame = cv2.imread('../sample/sign/parking/' + str(count) +  '.jpg',0)    
    frame = frame[40:440, 300:700]
    M = cv2.getRotationMatrix2D((200,200),90,1)
    frame = cv2.warpAffine(frame,M,(400,400))

    # frame = cv2.GaussianBlur(frame, (5,5), 0)
    kp, des = surf.detectAndCompute(frame,None)
    
    matches_stop = flann.knnMatch(des_stop,des,k=2)
    matches_parking = flann.knnMatch(des_parking,des,k=2)
    matches_uturn = flann.knnMatch(des_uturn,des,k=2)
    # Need to draw only good matches, so create a mask
    
    sign_detect(matches_stop, "stop", face_stop, kp_stop, kp, frame)
    sign_detect(matches_parking, "parking", face_parking, kp_parking, kp, frame)
    sign_detect(matches_uturn, "uturn", face_uturn, kp_uturn, kp, frame)


    k = cv2.waitKey(60) & 0xFF
    
    #break # For Debugging


    if k==27:
        break
    count = count + 1

cv2.destroyAllWindows()
# cap.release()
