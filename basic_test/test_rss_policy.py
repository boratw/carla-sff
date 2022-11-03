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
from rss_sensor import RssSensor # pylint: disable=relative-import

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)


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
    rss_sensor = None
    rss_restrictor = carla.RssRestrictor()
    with sess.as_default():
        for exp in [4]:
            log_file = open("policy_test_log/rss_policy_" + str(exp) + ".txt", "wt")
            log_file.write("Iteration\tSurvive_Time\tScore\n")
            for iteration in range(50):
                if rss_sensor != None:
                    rss_sensor.destroy()
                actor.reset()
                rss_sensor = RssSensor(actor.player, world, None, None, None)

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
                    world.tick()
                    if ret["collision"]:
                        break
                    if ret["success_dest"]:
                        success += 1


                    acceleration = loncontroller.run_step(8.0, ret["velocity"]) 
                    if acceleration >= 0.0:
                        accel = min(acceleration, 0.75)
                        brake = 0.0
                    else:
                        accel = 0.0
                        brake = min(abs(acceleration), 0.3)


                    steer = latcontroller.run_step(actor.route[2][0].transform, actor.player.get_transform())

                    rss_proper_response = rss_sensor.proper_response if rss_sensor.response_valid else None
                    if rss_proper_response:
                        control = carla.VehicleControl()
                        control.throttle = accel
                        control.brake = brake
                        control.steer = steer

                        control = rss_restrictor.restrict_vehicle_control(control, rss_proper_response, rss_sensor.ego_dynamics_on_route, actor.player.get_physics_control())

                        accel = control.throttle
                        brake = control.brake
                        steer = control.steer
                    

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


