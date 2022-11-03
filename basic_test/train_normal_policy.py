import numpy as np
import cv2
import random
import tensorflow.compat.v1 as tf
import sys
import os
import glob
import time
try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from actor import Actor
from network.predict_behavior2 import PredictBehavior


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
    #settings.no_rendering_mode = True
    world.apply_settings(settings)

    actor = Actor(world, client)

    tf.disable_eager_execution()
    sess = tf.Session()
    with sess.as_default():

        actor.reset()

        for step in range(1000):
            actor.step([0.2, 0.0, 0.0])

            world.tick()

finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    actor.destroy()

    time.sleep(0.5)