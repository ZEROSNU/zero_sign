import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from imutils import *

# define values boundaries for color
lower_yellow = np.array([15, 50, 100], np.uint8)
upper_yellow = np.array([40, 255, 255], np.uint8)
lower_white_hsv = np.array([0, 0, 200], np.uint8)
upper_white_hsv = np.array([255, 100, 255], np.uint8)

lower_white_rgb = np.array([190, 190, 190], np.uint8)
upper_white_rgb = np.array([255, 255, 255], np.uint8)


# hls_lower = np.array([0, 200, 0], np.uint8)
# hls_upper = np.array([255,255, 150], np.uint8)
def reject_outliers(data):
    m = 2
    u = np.mean(data[:,0])
    s = np.std(data[:,0])
    filtered = [f for f in data if (u - 1 * s < f[0] < u + 1 * s)]

    return np.array(filtered)



def process_image():
    count = 2
    x_buffer = []
    y_buffer = []

    angle_range = 10
    angle_additional = 0
    angle_range_max = angle_range + angle_additional

    while True:
        path = './sample/s/' + str(count) + '.jpg'
        img = cv2.imread(path)

        init_time = time.time()

        img = GaussianBlur(img, 5)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = GaussianBlur(hsv_img, 5)
        print("Blur and color time: ", time.time() - init_time)
        yellow_mask = findColor(hsv_img, lower_yellow, upper_yellow)


        full_mask = yellow_mask  # only YELLOW For now

        # DILATE
        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.dilate(full_mask, kernel, iterations=1)

        edges = cv2.Canny(img, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=30, maxLineGap=20)
        angles = []

        y_0_arr = []
        y_300_arr = []
        angle_0_arr = []
        angle_300_arr = []

        y_angle_array = []

        # sort line list by x value
        lines = lines[lines[:, 0, 0].argsort()]

        # yellow only
        prev_y1 = lines[0, 0, 1]
        prev_y2 = lines[0, 0, 3]

        #Polynomial fitting points buffer : previous 8 points



        for x1, y1, x2, y2 in lines[:, 0]:
            if ((full_mask[y1, x1] == 255) and (full_mask[y2, x2] == 255)):
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
                prev_y1 = y1
                prev_y2 = y2



                # filter if y1-prev_y is greater than 100
                angle_range_max = angle_range + angle_additional
                if (angle < angle_range_max and angle > - angle_range_max and abs(y1 - prev_y1) < 100 and abs(y2 - prev_y2) < 100):
                    # print(y1-prev_y)
                    # cv2.line(img, (x1,y1), (x2,y2),(0,0,255),2)
                    cv2.circle(img, (x1, y1), 2, (255, 0, 0), 2)
                    # cv2.circle(img, (x2,y2), 2, (255,0,0),2)



                    if x1 < 300:
                        y_0 = int(np.tan(angle * np.pi / 180) * (0 - x1) + y1)
                        y_0_arr.append(y_0)
                        angle_0_arr.append(angle)

                        y_angle_array.append([y_0,angle])

                    else:
                        y_0 = int(np.tan(angle * np.pi / 180) * (0 - x1) + y1)
                        y_300 = int(np.tan(angle * np.pi / 180) * (300 - x1) + y1)
                        y_300_arr.append(y_300)
                        angle_300_arr.append(angle)

                        y_angle_array.append([y_0, angle])

        # POLYNOMIAL FITTING 4 POINTS
        if (np.size(y_0_arr) > 4 and np.size(y_300_arr) > 4):


            x = [0, 300, 300, 600]
            y = []

            y.append((int(np.median(y_0_arr)) + int(np.mean(y_0_arr))) / 2)
            y.append((int(np.tan(np.median(angle_0_arr) * np.pi / 180) * (300 - 0) + y[0]) + int(
                np.tan(np.mean(angle_0_arr) * np.pi / 180) * (300 - 0) + y[0])) / 2)
            y.append((int(np.median(y_300_arr)) + int(np.mean(y_300_arr))) / 2)
            y.append((int(np.tan(np.median(angle_300_arr) * np.pi / 180) * (600 - 300) + y[2]) + int(
                np.tan(np.mean(angle_300_arr) * np.pi / 180) * (600 - 300) + y[2])) / 2)

            x_buffer = x_buffer + x
            y_buffer = y_buffer + y


            if (np.size(x_buffer) > 8):
                x_buffer[0:4] = []
                y_buffer[0:4] = []

            x_points = x_buffer
            y_points = y_buffer

            angle_additional = max(angle_additional - 3, 0)

            degree = 2

        elif(np.size(y_0_arr) + np.size(y_300_arr) > 2):

            filtered_y_angle = reject_outliers(np.array(y_angle_array))

            degree = 1
        else:

            angle_additional = min(35,angle_additional + 5)
            x_points = x_buffer
            y_points = y_buffer

            degree = 2

        if(np.size(x_points) > 0 or degree == 1):
            if(degree == 2):
                coefficients = np.polyfit(x_points, y_points, degree)
                color = (255,0,0)
            elif(degree == 1):
                y_line = np.mean(filtered_y_angle[:,0])
                slope = np.tan( np.mean(filtered_y_angle[:,1]) * np.pi / 180)
                coefficients = [slope, y_line]
                color = (0,255,0)

            polypoints = np.zeros((600, 2))
            f = np.poly1d(coefficients)
            t = np.arange(0, 600, 1)
            polypoints[:, 0] = t
            polypoints[:, 1] = f(t)
            cv2.polylines(img, np.int32([polypoints]), False, color, 2)
            cv2.circle(img, (599, int(f(600))), 2, (0, 0, 255), 2)

            # MASKING OUTPUT IMAGE
            mask = np.arange(0, 600, 1)
            if (degree == 2):
                mask = coefficients[0] * mask * mask + coefficients[1] * mask + coefficients[2]
            elif (degree == 1):
                mask = coefficients[0] * mask  + coefficients[1]
            mask = np.zeros((600, 600)) + mask  # broadcast into 2d

            y_vals = np.arange(0, 600, 1)
            y_vals = np.broadcast_to(y_vals, (600, 600)).T

            masked_img = np.zeros((600, 600), dtype='uint8')
            masked_img[mask > y_vals] = 255
            cv2.imshow('masked', masked_img)
        print("Time taken: ", time.time() - init_time)

        # cv2.imshow('yellow', full_mask)
        cv2.imshow('original', img)
        count += 1
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    process_image()