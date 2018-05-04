import cv2
import numpy as numpy
import time


face = cv2.imread('../sample/sign/stop.jpg',0)
#cap = cv2.VideoCapture(0)

surf = cv2.xfeatures2d.SURF_create(2000)
kp1, des1 = surf.detectAndCompute(face,None)
print len(kp1)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=30)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

count = 1

while True:
    try:
        init_time = time.time()
        #_,frame = cap.read()
        frame = cv2.imread('../sample/sign/stop/' + str(count) +  '.jpg',0)    
        frame = frame[40:440, 300:700]
        M = cv2.getRotationMatrix2D((200,200),90,1)
        frame = cv2.warpAffine(frame,M,(400,400))
        # frame = cv2.GaussianBlur(frame, (5,5), 0)
        kp2, des2 = surf.detectAndCompute(frame,None)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
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
        img_show1 = cv2.drawMatchesKnn(face,kp1,frame,kp2,matches,None,**draw_params)
        print("Time taken: ", (time.time()-init_time))
        print(is_detected)
        if is_detected >= 20:
            print("@@SIGN DETECTED : " + str(is_detected))
        
        cv2.imshow('face',img_show1)
    except:
        print("Error")
        pass
    k = cv2.waitKey(60) & 0xFF
    
    #break # For Debugging


    if k==27:
        break
    count = count + 1
    
cv2.destroyAllWindows()
# cap.release()


