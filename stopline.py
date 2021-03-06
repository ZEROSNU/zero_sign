import cv2
import numpy as np
import time

#DEBUG MODE : True / False
Z_DEBUG = True
USE_HoughLines = False #HoughLines : True / HoughLinesP : False

lower_white_hsv = np.array([0,0,220], np.uint8)
upper_white_hsv = np.array([255,50,255], np.uint8)

ANGLE_THRESHOLD = 20 #angle difference (DEGREE)

MIN_DETECTED_THRESHOLD = 3 #for HoughLines
MIN_DETECTED_THRESHOLD_P = 3  

detect_count = 0
#number of detected frames in STOPLINE_BUFFER >= DETECT_THRESHOLD : STOPLINE DETECTED!  
BUFFER_SIZE = 5
DETECT_THRESHOLD = 3
STOPLINE_BUFFER = np.zeros(BUFFER_SIZE)
###########################################


def GaussianBlur(img, kernel_size):
    blur_n = 5 # blur_n : 5 or higher
    blurred_img = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    return blurred_img

# Masking color
def findColor(hsv_image, lower, upper):
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask

def DrawHoughLines(edge, img): #edge: edge input, img : image on which draw lines
    

    lines = cv2.HoughLines(edge,1,np.pi/180,70)  
   
    detect_count = 0
    try:
        for temp in lines:
            
            rho = temp[0][0]
            theta = temp[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
          
            if np.sin(theta) < np.sin(float(ANGLE_THRESHOLD) / float(180) * np.pi):
                detect_count = detect_count + 1
                if Z_DEBUG:
                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                    pass
            else:
                if Z_DEBUG:
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                    pass


        if Z_DEBUG:
            print detect_count

        if detect_count >= MIN_DETECTED_THRESHOLD:
            detected = True
        else:
            detected = False

        #buffer
        STOPLINE_BUFFER[0:BUFFER_SIZE-1] = STOPLINE_BUFFER[1:BUFFER_SIZE]
        STOPLINE_BUFFER[BUFFER_SIZE-1] = detected

        return img
    except:
        return img

def DrawHoughLinesP(edge, img): #edge: edge input, img : image on which draw lines

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=50, maxLineGap=20)
    detect_count = 0
    try:
        for x1, y1, x2, y2 in lines[:, 0]:
            
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(np.cos(angle*np.pi/float(180))) < np.sin(float(ANGLE_THRESHOLD) / float(180) * np.pi):
                detect_count = detect_count + 1
                if Z_DEBUG:
                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                    pass
            else:
                if Z_DEBUG:
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                    pass

        if Z_DEBUG:
            print detect_count
            pass
        if detect_count >= MIN_DETECTED_THRESHOLD_P:
            detected = True
        else:
            detected = False

        #buffer
        STOPLINE_BUFFER[0:BUFFER_SIZE-1] = STOPLINE_BUFFER[1:BUFFER_SIZE]
        STOPLINE_BUFFER[BUFFER_SIZE-1] = detected

        return img
    except:
        return img
###########################################

count = 1

while True:
    init_time = time.time()

    #Image input
    path = '../sample/stopline1/' + str(count) +  '.jpg'
    img = cv2.imread(path,1)    
    
    init_time = time.time()
        
    img = GaussianBlur(img, 5)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv_img, lower_white_hsv,upper_white_hsv)
    edges = cv2.Canny(white_mask, 200, 800)

    #HoughLines vs HoughLinesP
    if USE_HoughLines:
        DrawHoughLines(edges, img)
    else:
        DrawHoughLinesP(edges,img)

    #BUFFER
    if np.size(np.where(STOPLINE_BUFFER == True)) >= DETECT_THRESHOLD:
        print("stopline detected!")
        '''
        Do something!
        '''

    if Z_DEBUG:
        cv2.imshow('img', img)
        cv2.imshow('white mask', white_mask)
        k = cv2.waitKey(50) & 0xFF
        print("Time taken: ", (time.time()-init_time))
        if k==27:
            break
    count = count + 1

if Z_DEBUG:
    cv2.destroyAllWindows()

