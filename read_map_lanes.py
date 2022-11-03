#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np


from numpy import random

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

try:
    world = client.get_world()

    lanes = []

    topologies = world.get_map().get_topology()
    for wp1, wp2 in topologies:
        lane = []
        curwp = wp1
        endloc = wp2.transform.location


        while True:
            lane.append([curwp.transform.location.x, curwp.transform.location.y])
            nextlist = curwp.next(2.0)
            if len(nextlist) == 0:
                break
            curwp = nextlist[0]
            if curwp.transform.location.distance(endloc) < 2.0:
                break
        lane.append([curwp.transform.location.x, curwp.transform.location.y])
        lanes.append(np.array(lane))

    wf = open("lane.txt", "wt")
    for lane in lanes:
        for item in lane:
            wf.write(str(item[0]) + "," + str(item[1]) + "\t")
        wf.write("\n")
except:
    pass