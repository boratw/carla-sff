import numpy as np
import cv2
import random
import tensorflow.compat.v1 as tf
import sys
import os
import glob
import time
import math
try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from actor import Actor
from shapely.geometry import Polygon
from controller import PIDLongitudinalController, PIDLateralController

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)



def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Check if a location is both within a certain distance from a reference object.
    By using 'angle_interval', the angle between the location and reference transform
    will also be tkaen into account, being 0 a location in front and 180, one behind.

    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    # Further than the max distance
    if norm_target > max_distance:
        return False

    # We don't care about the angle, nothing else to check
    if not angle_interval:
        return True

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return min_angle < angle < max_angle



def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def detect_vehicles(actor, max_distance, up_angle_th=90, low_angle_th=0, lane_offset=0):

    

    ego_transform = actor.player.get_transform()
    ego_wpt = actor.map.get_waypoint(actor.player.get_location())

    # Get the right offset
    if ego_wpt.lane_id < 0 and lane_offset != 0:
        lane_offset *= -1

    # Get the transform of the front of the ego
    ego_forward_vector = ego_transform.get_forward_vector()
    ego_extent = actor.player.bounding_box.extent.x
    ego_front_transform = ego_transform
    ego_front_transform.location += carla.Location(
        x=ego_extent * ego_forward_vector.x,
        y=ego_extent * ego_forward_vector.y,
    )

    for target_vehicle in actor.npc_vehicle_actors:
        target_transform = target_vehicle.get_transform()
        target_wpt = actor.map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

        # Simplified version for outside junctions
        if not ego_wpt.is_junction or not target_wpt.is_junction:

            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                next_wpt =  actor.route[3][0]
                if not next_wpt:
                    continue
                if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                    continue

            target_forward_vector = target_transform.get_forward_vector()
            target_extent = target_vehicle.bounding_box.extent.x
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                x=target_extent * target_forward_vector.x,
                y=target_extent * target_forward_vector.y,
            )

            if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        # Waypoints aren't reliable, check the proximity of the vehicle to the route
        else:
            route_bb = []
            ego_location = ego_transform.location
            extent_y = actor.player.bounding_box.extent.y
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            route_bb.append([p1.x, p1.y, p1.z])
            route_bb.append([p2.x, p2.y, p2.z])

            for wp, _ in actor.route:
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

            if len(route_bb) < 3:
                # 2 points don't create a polygon, nothing to check
                return (False, None, -1)
            ego_polygon = Polygon(route_bb)

            # Compare the two polygons
            for target_vehicle in actor.npc_vehicle_actors:
                target_extent = target_vehicle.bounding_box.extent.x
                if target_vehicle.id == actor.player.id:
                    continue
                if ego_location.distance(target_vehicle.get_location()) > max_distance:
                    continue

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if ego_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            return (False, None, -1)

    return (False, None, -1)



try:
    world = client.get_world()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = True
    world.apply_settings(settings)

    actor = Actor(world, client)
    latcontroller = PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05)
    loncontroller = PIDLongitudinalController(K_P = 1.0, K_I = 0.05, K_D = 0., dt=0.05)

    tf.disable_eager_execution()
    sess = tf.Session()
    with sess.as_default():
        for exp in [0, 1, 2, 4]:
            log_file = open("policy_test_log/default_policy2_" + str(exp) + ".txt", "wt")
            log_file.write("Iteration\tSurvive_Time\tScore\n")
            for iteration in range(50):
                actor.reset()
                for a in actor.npc_vehicle_actors:
                    traffic_manager.ignore_lights_percentage(a, 5.0 * exp)
                    traffic_manager.ignore_vehicles_percentage(a, 0.5 * exp)
                world.tick()
                world.tick()
                world.tick()
                world.tick()
                world.tick()
                success = 0
                accel, brake, steer = 1.0, 0.0, 0.0
                for step in range(5000):
                    ret = actor.step([accel, brake, steer])
                    detect_result = detect_vehicles(actor, 8.0 + 0.5 * ret["velocity"])
                    world.tick()
                    if ret["collision"]:
                        break
                    if ret["success_dest"]:
                        success += 1

                    target_velocity = 8.0
                    if(detect_result[0]):
                        target_velocity = detect_result[2] * 0.5 - 1.5
                        if target_velocity < 0.:
                            target_velocity = 0.
                        elif target_velocity > 8.:
                            target_velocity = 8. 

                    acceleration = loncontroller.run_step(target_velocity, ret["velocity"]) 
                    if acceleration >= 0.0:
                        accel = min(acceleration, 0.75)
                        brake = 0.0
                    else:
                        accel = 0.0
                        brake = min(abs(acceleration), 0.3)


                    steer = latcontroller.run_step(actor.route[2][0].transform, actor.player.get_transform())

                    for a in actor.npc_vehicle_actors:
                        r = random.random()
                        if r < 0.001 * exp * exp:
                            traffic_manager.force_lane_change(a, True)
                        elif r < 0.002 * exp * exp:
                            traffic_manager.force_lane_change(a, False)

                print(str(iteration) + "\t" + str(step + 1) + "\t" + str(success) + "\n")
                log_file.write(str(iteration) + "\t" + str(step + 1) + "\t" + str(success) + "\n")
finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    actor.destroy()

    time.sleep(0.5)


