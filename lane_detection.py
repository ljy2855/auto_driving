import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LaneDetection:
    # '''
    # Lane detection module using edge detection and b-spline fitting

    # args: 
    #     cut_size (cut_size=68) cut the image at the front of the car
    #     spline_smoothness (default=10)
    #     gradient_threshold (default=14)
    #     distance_maxima_gradient (default=3)

    # '''

    def __init__(self, cut_size=68, spline_smoothness=30, gradient_threshold=18, distance_maxima_gradient=3):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0


    def cut_gray(self, state_image_full):
        # '''
        # ##### TODO #####
        # This function should cut the imagen at the front end of the car (e.g. pixel row 68) 
        # and translate to grey scale

        # input:
        #     state_image_full 96x96x3

        # output:
        #     gray_state_image 68x96x1

        # '''
        
        # 이미지 가로축 자르기
        cropped_image = state_image_full[:self.cut_size, :, :]

        
        #red layer : 29%, green layer : 58%, blue layer : 11%
        gray_state_image = np.dot(cropped_image[...,:3], [0.2989, 0.5870, 0.1140])
        
        return gray_state_image


    def edge_detection(self, gray_image):
        # '''
        # ##### TODO #####
        # In order to find edges in the gray state image, 
        # this function should derive the absolute gradients of the gray state image.
        # Derive the absolute gradients using numpy for each pixel. 
        # To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        # input:
        #     gray_state_image 68x96x1

        # output:
        #     gradient_sum 68x96x1

        # '''
        
        # Calculate the gradient in x and y direction
        gradient_x, gradient_y = np.gradient(gray_image)

        # Calculate the absolute value of the gradient
        abs_gradient_x = np.abs(gradient_x)
        abs_gradient_y = np.abs(gradient_y)

        # Sum the absolute gradients
        gradient_sum = abs_gradient_x + abs_gradient_y

        # Apply thresholding
        gradient_sum[gradient_sum < self.gradient_threshold] = 0

        x_min, x_max = 45, 50
        y_min, y_max = 0, 2
        gradient_sum = np.flipud(gradient_sum)

        # x_min, x_max = 45, 50
        # y_min, y_max = 0, 2 사이의 그라디언트를 0으로 설정
        gradient_sum[y_min:y_max+1, x_min:x_max+1] = 0
        
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        # '''
        # ##### TODO #####
        # This function should output arguments of local maxima for each row of the gradient image.
        # You can use scipy.signal.find_peaks to detect maxima. 
        # Hint: Use distance argument for a better robustness.

        # input:
        #     gradient_sum 68x96x1

        # output:
        #     maxima (np.array) 2x Number_maxima

        # '''



        maxima = []

        for row in range(gradient_sum.shape[0]):
            # 각 행에서 국소 최대값 찾기
            peaks, _ = find_peaks(gradient_sum[row,:,0], distance=self.distance_maxima_gradient)

            # 각 최대값의 위치 저장
            for peak in peaks:
                maxima.append([peak, row])

        # numpy 배열로 변환
        argmaxima = np.array(maxima)
        # argmaxima[:, 1] = self.cut_size - argmaxima[:, 1]

        # x_min, x_max = 45, 50
        # y_min, y_max = 0, 2

        # mask = ~((argmaxima[:, 0] >= x_min) & (argmaxima[:, 0] <= x_max) & (argmaxima[:, 1] >= y_min) & (argmaxima[:, 1] <= y_max))

        # # Apply the mask to the array to filter out the points
        # argmaxima = argmaxima[mask]

        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        # '''
        # Find the first lane_boundaries points above the car.
        # Special cases like just detecting one lane_boundary or more than two are considered. 
        # Even though there is space for improvement ;) 

        # input:
        #     gradient_sum 68x96x1

        # output: 
        #     lane_boundary1_startpoint
        #     lane_boundary2_startpoint
        #     lanes_found  true if lane_boundaries were found
        # '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found and row < gradient_sum.shape[0]:
            argmaxima = find_peaks(gradient_sum[row,:,0], distance=5)[0]

            if argmaxima.size == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])
                lane_boundary2_startpoint = np.array([[0 if argmaxima[0] < 48 else 96, row]])
                lanes_found = True

            elif argmaxima.size == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1], row]])
                lanes_found = True

            elif argmaxima.size > 2:
                sorted_indices = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[sorted_indices[0]], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[sorted_indices[1]], row]])
                lanes_found = True

            row += 1

            if row == self.cut_size:
                print("cut_size 끝에 도달하여 차선을 찾지 못했습니다.")
                lane_boundary1_startpoint = np.array([[0, 0]])
                lane_boundary2_startpoint = np.array([[0, 0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found



    def lane_detection(self, state_image_full):
        # '''
        # ##### TODO #####
        # This function should perform the road detection 

        # args:
        #     state_image_full [96, 96, 3]

        # out:
        #     lane_boundary1 spline
        #     lane_boundary2 spline
        # '''

        # to gray
        gray_state = self.cut_gray(state_image_full)
       
        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        gradient_sum = np.expand_dims(gradient_sum,axis=2)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)
        
        # print(maxima)
        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)
        
        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)

            # lane_boundary 1

            
            
            while True:
                # 마지막 차선 경계점과 가장 가까운 최대 기울기 찾기
                nearest_point = self.find_nearest_point(maxima, lane_boundary1_points[-1])

                # 최대 기울기가 없거나 거리가 너무 멀면 반복 중단
                if nearest_point is None:
                    break

                if not any((lane_boundary1_points == nearest_point).all(1)):
                    lane_boundary1_points = np.vstack((lane_boundary1_points, nearest_point))
                    # 해당 점을 maxima 배열에서 삭제
                maxima = np.delete(maxima, np.where((maxima == nearest_point).all(axis=1))[0], axis=0)

            # lane_boundary 2
            while True:
                # 마지막 차선 경계점과 가장 가까운 최대 기울기 찾기
                nearest_point = self.find_nearest_point(maxima, lane_boundary2_points[-1])

                # 최대 기울기가 없거나 거리가 너무 멀면 반복 중단
                if nearest_point is None:
                    break

                if not any((lane_boundary2_points == nearest_point).all(1)):
                    lane_boundary2_points = np.vstack((lane_boundary2_points, nearest_point))
                    # 해당 점을 maxima 배열에서 삭제
                maxima = np.delete(maxima, np.where((maxima == nearest_point).all(axis=1))[0], axis=0)
            # plt.scatter(lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]+28, c='red', marker='o', s=10)
            # plt.scatter(lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]+28, c='red', marker='o', s=10)
        
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
           
                # 차선 1 스플라인
                tck1, _ = splprep([lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]], s=self.spline_smoothness, k=2)
                lane_boundary1 = tck1
                
                # 차선 2 스플라인
                tck2, _ = splprep([lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]], s=self.spline_smoothness , k=2)
                lane_boundary2 = tck2
                
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2
    
    def find_nearest_point(self, maxima, current_point, threshold=8):
        min_distance = float('inf')
        nearest_point = None

        for point in maxima:
            
            distance = np.sqrt((current_point[0] - point[0])**2 + (current_point[1] - point[1])**2)

            # 가장 가까운 점 갱신
            if distance < min_distance:
                min_distance = distance
                nearest_point = point

        # 임계값을 초과하는 경우 None 반환
        if min_distance > threshold:
            return None

        return nearest_point

    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        # '''
        # Plot lanes and way points
        # '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+96-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
