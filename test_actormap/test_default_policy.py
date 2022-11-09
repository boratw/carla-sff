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

envdir = os.path.dirname(os.getcwd())
sys.path.append(envdir)
sys.path.append(envdir + "/algorithm")

import carla
from actormap import ActorMap
from controller import PIDLongitudinalController, PIDLateralController

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

    actormap = ActorMap(world, client, 50)
    actormap.reset()
    world.tick()
    world.tick()
    world.tick()

    for step in range(5000):
        actormap.step()
        world.tick()
        
finally:
    actormap.destroy()
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)


    time.sleep(0.5)


