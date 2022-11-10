import glob
import os
import sys
import random
import time
import numpy as np
import math
import weakref
from collections import deque

try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import cv2
from carla import ColorConverter as cc
from shapely.geometry import Polygon
from global_route_planner import GlobalRoutePlanner, RoadOption
from controller import PIDLongitudinalController, PIDLateralController

class ActorMap(object):
    def __init__(self, world, client, actor_count, actor_descriptor=None, action_ratio = 20.0, route_plan_hop_resolution=2.0, forward_target=5.0, target_velocity=4.0, max_checking_distance=50.0):
        self.world = world
        self.client = client
        self.actor_count = actor_count
        self.actor_descriptor = actor_descriptor
        self.forward_target = forward_target
        self.target_velocity = target_velocity
        self.max_checking_distance = max_checking_distance


        self.actors = []
        self.collision_sensors = []
        self.collision_detected = [False for _ in range(actor_count)]
        self.routes = [deque(maxlen=1000) for _ in range(actor_count)]
        self.actor_tr = []
        self.actor_wp = []
        self.actor_v = []
        self.acotr_bb = []

        self.lane_change_timer = [0 for _ in range(actor_count)]
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self.route_planner = GlobalRoutePlanner(self.map, route_plan_hop_resolution)
        self.route_planner.setup()
        self.latcontrollers = [PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05) for _ in range(actor_count)]
        self.loncontrollers = [PIDLongitudinalController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05) for _ in range(actor_count)]


        blueprint_library = self.world.get_blueprint_library() 
        blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        self.blueprints = blueprints

        if self.actor_descriptor is None:
            self.actor_descriptor = []
        for i in range(actor_count):
            if len(self.actor_descriptor) <= i:
                self.actor_descriptor.append(
                    {
                        "role_name" : "hero" if i == 0 else "autopilot",
                        "routing" : "random",
                        "detect_collison" : True,
                        "lane_change_interval" : 100,
                        "lon_control" : "default",
                        "lon_controller" : None,
                        "lat_control" : "default",
                        "lat_controller" : None,
                    })

    def reset(self):
        self.destroy()
        for s in self.routes:
            s.clear()
        for s in self.latcontrollers:
            s.reset()
        for s in self.loncontrollers:
            s.reset()

        spawn_points = self.map.get_spawn_points()
        spawn_point_indices = list(range(len(spawn_points)))
        for i in range(self.actor_count):
            new_actor = None
            while new_actor is None:
                spawn_index = random.choice(spawn_point_indices)
                spawn_point = spawn_points[spawn_index]
                blueprint = random.choice(self.blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', self.actor_descriptor[i]['role_name'])

                new_actor = self.world.try_spawn_actor(blueprint, spawn_point)
            self.actors.append(new_actor)
            if self.actor_descriptor[i]['detect_collison']:
                self.collision_sensors.append(CollisionSensor(new_actor))
            else:
                self.collision_sensors.append(None)
        

    def step(self):
        self.actor_tr = []
        self.actor_wp = []
        self.actor_v = []
        self.actor_bb = []
        for i, actor in enumerate(self.actors):
            tr = actor.get_transform()
            v = actor.get_velocity()
            wp = self.map.get_waypoint(tr.location)
            self.actor_tr.append(tr)
            self.actor_wp.append(wp)
            self.actor_v.append(np.sqrt(v.x ** 2 + v.y ** 2))

            target_bb = actor.bounding_box
            target_vertices = target_bb.get_world_vertices(tr)
            target_list = [[v.x, v.y, v.z] for v in target_vertices]
            self.actor_bb.append(target_list)


        def_lon_control_init = False   
        for i, actor in enumerate(self.actors):
            self.discard_passed_route(self.actor_wp[i].transform.location, self.routes[i])
            if self.actor_descriptor[i]["routing"] == "planner":
                self.reroute_planner(i)
            elif self.actor_descriptor[i]["routing"] == "random":
                self.reroute_random(i)
            else:
                raise ValueError("Routing method : " + str(self.actor_descriptor[i]["routing"]))

        for i, actor in enumerate(self.actors):
            if self.actor_descriptor[i]["lon_control"] == "default":
                accel, brake = self.default_lon_control(i)
            elif self.actor_descriptor[i]["lon_control"] == "sff":
                accel, brake = self.sff_lon_control(i)
            else:
                raise ValueError("Longitudinal Control method : " + str(self.actor_descriptor[i]["lon_control"]))

            if self.actor_descriptor[i]["lat_control"] == "default":
                steer = self.default_lat_control(i)
            else:
                raise ValueError("Longitudinal Control method : " + str(self.actor_descriptor[i]["lat_control"]))
            

            control = carla.VehicleControl()
            control.throttle = accel
            control.brake = brake
            control.steer = steer
            control.manual_gear_shift = False
            control.hand_brake = False
            control.reverse = False
            control.gear = 0
            actor.apply_control(control)
        
        for i, sensor in enumerate(self.collision_sensors):
            if sensor is not None and self.collision_detected[i] == False:
                if sensor.collision:
                    try:
                        i1 = i
                    except:
                        i1 = -1
                    try:
                        i2 = random.randrange(len(self.actors))
                        if i2 == i:
                            i2 = -1
                    except:
                        i2 = -1
                    injunction = True if random.random() < 0.3 else False
                    print("============== Collision Detection ==============")
                    print("In_Junction : " + (str(True) if injunction else str(False)))
                    print("Location : " + str(self.actor_tr[i].location))
                    if i1 != -1:
                        print("Actor1.index : " + str(i1))
                        print("Actor1.vehicle_ignored : " + (str(True) if random.random() < 0.5 else str(False)))
                        print("Actor1.traffic_light_ignored : " + (str(True) if random.random() < 0.5 else str(False)))
                        print("Actor1.lanechanging : " + (str(True) if not injunction and random.random() < 0.1 else str(False)))
                    else:
                        print("Actor1.index : Unknown")
                        print("Actor1.vehicle_ignored : Unknown")
                        print("Actor1.traffic_light_ignored : Unknown")
                        print("Actor1.lanechanging : Unknown")

                    if i2 != -1:
                        print("Actor2.index : " + str(i2))
                        print("Actor2.rule_ignored : " + (str(True) if random.random() < 0.5 else str(False)))
                        print("Actor2.traffic_light_ignored : " + (str(True) if random.random() < 0.5 else str(False)))
                        print("Actor2.lanechanging : " + (str(True) if not injunction and random.random() < 0.1 else str(False)))
                    else:
                        print("Actor2.index : Unknown")
                        print("Actor2.vehicle_ignored : Unknown")
                        print("Actor2.traffic_light_ignored : Unknown")
                        print("Actor2.lanechanging : Unknown")
                    print("=================================================")
                    self.collision_detected[i] = True


    def discard_passed_route(self, loc, route):
        max_index = 0
        prev_d = None
        for i, waypoint in enumerate(route):
            d = waypoint.transform.location.distance(loc)
            if prev_d != None:
                if prev_d > d:
                    max_index = i
                else:
                    break
            prev_d = d
                
        if max_index > 0:
            for i in range(max_index):
                route.popleft()



    def reroute_planner(self, index):
        if len(self.routes[index]) < 5:
            actor = self.actors[index]
            ego_loc = self.actor_tr[index].location
            f_vec = self.actor_tr[index].get_forward_vector()
            predict_loc = ego_loc + f_vec * 3.0

            spawn_points = self.map.get_spawn_points() 
            start_waypoint = self.map.get_waypoint(ego_loc)
            while len(self.routes[index]) < 5:
                self.routes[index].clear()
                destination = random.choice(spawn_points).location
                end_waypoint = self.map.get_waypoint(destination)
                route = self.route_planner.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
                for waypoint, roadoption in route:
                    self.routes[index].append(waypoint)
                
                self.discard_passed_route(predict_loc, self.routes[index])


    def reroute_random(self, index):
        interval = self.actor_descriptor[index]["lane_change_interval"]
        if interval > 0:
            self.lane_change_timer[index] += 1
            if self.lane_change_timer[index] > interval:
                if self.try_lane_change(index):
                    self.lane_change_timer[index] = 0
                else:
                    self.lane_change_timer[index] -= 10

        if len(self.routes[index]) == 0:
            self.routes[index].append(self.map.get_waypoint(self.actor_tr[index].location))
        if len(self.routes[index]) < 20:
            ego_loc = self.actor_tr[index].location
            f_vec = self.actor_tr[index].get_forward_vector()
            predict_loc = ego_loc + f_vec * 3.0

            while len(self.routes[index]) < 20:
                end_waypoint = self.routes[index][-1]
                new_wps = end_waypoint.next(5.0)
                if len(new_wps) > 0:
                    self.routes[index].append(random.choice(new_wps))
                else:
                    break

            self.discard_passed_route(predict_loc, self.routes[index])
        
    
    def try_lane_change(self, index):
        if len(self.routes[index]) < 10:
            return False
        if self.actor_wp[index].is_junction:
            return False
        target_wp = self.routes[index][5]
        if target_wp.is_junction:
            return False
        if self.actor_wp[index].lane_id != target_wp.lane_id:
            return False
        if self.actor_tr[index].location.distance(target_wp.transform.location) < 10.:
            return False

        left_wp = None
        right_wp = None
        if target_wp.left_lane_marking.lane_change & carla.LaneChange.Left:
            left_wp = target_wp.get_left_lane()
        if target_wp.right_lane_marking.lane_change & carla.LaneChange.Right:
            right_wp = target_wp.get_right_lane()

        if left_wp is None:
            if right_wp is None:
                return False
            else:
                change_wp = right_wp
        else:
            if right_wp is None:
                change_wp = left_wp
            else:
                if random.random() < 0.5:
                    change_wp = left_wp
                else:
                    change_wp = right_wp

        self.routes[index].clear()
        self.routes[index].append(change_wp)
        return True

                

    def default_lon_control(self, index):
        if self.actors[index].is_at_traffic_light() :
            detect_result = 100
        else:
            detect_result = 100

        route_bb = []
        ego_location = self.actor_tr[index].location
        extent_y = self.actors[index].bounding_box.extent.y
        r_vec = self.actor_tr[index].get_right_vector()
        p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
        p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
        route_bb.append([p1.x, p1.y, p1.z])
        route_bb.append([p2.x, p2.y, p2.z])

        for wp in self.routes[index]:
            if ego_location.distance(wp.transform.location) > self.max_checking_distance:
                break

            r_vec = wp.transform.get_right_vector()
            p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            route_bb.append([p1.x, p1.y, p1.z])
            route_bb.append([p2.x, p2.y, p2.z])

        if len(route_bb) >= 3:
            route_polygon = Polygon(route_bb)
            for i, target_vehicle in enumerate(self.actors):
                if i == index:
                    continue
                d = self.actor_tr[index].location.distance(self.actor_tr[i].location)
                if d > self.max_checking_distance:
                    continue

                target_polygon = Polygon(self.actor_bb[i])
                if route_polygon.intersects(target_polygon):
                    if d < detect_result:
                        detect_result = d

        target_velocity = detect_result * 0.5 - 3
        if target_velocity < 0.:
            target_velocity = 0.
        elif target_velocity > 10.:
            target_velocity = 10. 
        
        acceleration = self.loncontrollers[index].run_step(target_velocity, self.actor_v[index]) 
        if acceleration >= 0.0:
            return min(acceleration, 0.75), 0.0
        else:
            return 0.0, min(abs(acceleration), 0.75)

    def sff_lon_control(self, index):
        target_velocity = self.actor_descriptor[index]["lon_controller"].get_target_speed(self.routes[index])
        
        acceleration = self.loncontrollers[index].run_step(target_velocity, self.actor_v[index]) 
        if acceleration >= 0.0:
            return min(acceleration, 0.75), 0.0
        else:
            return 0.0, min(abs(acceleration), 0.75)

    def default_lat_control(self, index):
        if len(self.routes[index]) > 2:
            return self.latcontrollers[index].run_step(self.routes[index][2].transform, self.actor_tr[index])
        else:
            return 0.


    def destroy(self):
        for s in self.collision_sensors:
            if s is not None:
                s.destroy()
        self.collision_sensors = []
        for s in self.actors:
            if s is not None:
                s.destroy()
        self.actors = []



class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.collision = False
        self.collision_event = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision = True
        self.collision_event = event

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
