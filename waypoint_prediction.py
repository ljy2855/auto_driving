import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import sys


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    # Compute vectors between consecutive waypoints and normalize them
    vecs = np.diff(waypoints, axis=1)
    norms = np.linalg.norm(vecs, axis=0) + 1e-5  # Prevent division by zero
    normalized_vecs = vecs / norms

    # Compute the curvature as the sum of the dot product of consecutive vectors
    curvatures = np.einsum('ij,ij->j', normalized_vecs[:, :-1], normalized_vecs[:, 1:])
    return np.sum(curvatures)



def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    # Directly compute mean squared error without reshaping
    ls_term = np.mean((waypoints - waypoints_center) ** 2)

    # Curvature term
    curv_term = curvature(waypoints.reshape(2, -1))

    # Objective function
    return ls_term + weight_curvature * curv_term

def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    param_range = np.linspace(0, 1, num_waypoints)

    def get_spline_points(spline, param_range):
        return np.array(splev(param_range, spline))


    
    roadside1_points = get_spline_points(roadside1_spline, param_range)
    roadside2_points = get_spline_points(roadside2_spline, param_range)
    waypoints_center = (roadside1_points + roadside2_points) / 2

    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments

        # derive roadside points from spline

        # derive center between corresponding roadside points

        # output way_points with shape(2 x Num_waypoints)

    
        return waypoints_center
    
    elif way_type == "smooth":
        ##### TODO #####

        # create spline points

        # roadside points from spline

        # center between corresponding roadside points

        # init optimized waypoints
        
        # optimization


        def optimization_func(way_points_flat):
            return smoothing_objective(way_points_flat, waypoints_center.flatten())

        # Initial guess for the optimization: the center waypoints
        initial_guess = waypoints_center.flatten()

        # Perform the optimization
        result = minimize(optimization_func, initial_guess)
        optimized_waypoints = result.x.reshape(2, -1)
        return optimized_waypoints



def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''

    waypoints_subset = waypoints[:, :num_waypoints_used]
    curv = curvature(waypoints_subset)

    # Calculate target speed
    target_speed = max_speed * np.exp(-exp_constant * curv) + offset_speed
    return np.clip(target_speed, 0, max_speed)