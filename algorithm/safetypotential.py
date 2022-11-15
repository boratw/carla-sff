
import glob
import os
import sys
try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import cv2
import numpy as np
import datetime

from network.predict_behavior3 import PredictBehavior
import tensorflow.compat.v1 as tf

class SafetyPotential:
    def __init__(self, lane_txt, visualize=False, record_video=False):
        self.player = None
        self.visualize = visualize
        self.record_video = record_video

        self.lanes = []
        with open(lane_txt, "rt") as rf:
            for line in rf.readlines():
                lane = []
                for s in line.split("\t"):
                    v = s.split(",")
                    if len(v) == 2:
                        lane.append([float(v[0]), float(v[1])])
                if len(lane) > 0:
                    self.lanes.append(np.array(lane))


        self.network_input_map = np.full((4096, 4096, 3), 128, np.uint8)
        self.network_input_loctr = np.array([256, 216])
        for lane in self.lanes:
            for i, _ in enumerate(lane[:-1]):
                dx = lane[i+1][0] - lane[i][0]
                dy = lane[i+1][1] - lane[i][1]
                r = np.sqrt(dx * dx + dy * dy)
                if r > 0.1:
                    color = ( int(dx * 127 / r + 128), 128, int(dy * 127 / r + 128) )
                    cv2.line(self.network_input_map, tuple(((lane[i] + self.network_input_loctr) * 8.).astype(np.int32)), tuple(((lane[i+1] + self.network_input_loctr) * 8.).astype(np.int32)), color, 4)
        
        self.cam_topview = None
        self.cam_frontview = None
        self.img_topview = None
        self.img_frontview = None

        tf.disable_eager_execution()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.learner = PredictBehavior()
            learner_saver = tf.train.Saver(var_list=self.learner.trainable_dict, max_to_keep=0)
            learner_saver.restore(self.sess, "/home/user/Documents/Taewoo/carla-sff/iter_5200.ckpt")

    def Assign_Player(self, player):
        self.player = player
        if self.visualize:
            world = player.get_world()

            bp = world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '1024')
            bp.set_attribute('image_size_y', '1024')
            self.cam_topview = world.spawn_actor(bp, carla.Transform(
                carla.Location(x=24.0, z=32.0), carla.Rotation(pitch=-90, yaw=0)), attach_to=player)
            self.cam_topview.listen(lambda image: self.on_cam_topview_update(image))

            bp = world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '1024')
            bp.set_attribute('image_size_y', '512')
            self.cam_frontview = world.spawn_actor(bp, carla.Transform(
                carla.Location(x=-7.5, z=2.5)), attach_to=player) # 2.3 1.0
            self.cam_frontview.listen(lambda image: self.on_cam_frontview_update(image))

            if self.record_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video = cv2.VideoWriter('recorded_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.avi', fourcc, 13, (2048, 1024))

    def Assign_NPCS(self, npcs):
        self.npcs = npcs
        self.acc_buffer = np.array( [ [ [0., 0.] for _ in range(4)] for _ in self.npcs]  )

    def Get_Predict_Result(self):
        screen_copied = self.network_input_map.copy()
        for npc in self.close_npcs:
            tr = npc.get_transform()
            v = npc.get_velocity()
            loc = np.array([tr.location.x, tr.location.y])
            cv2.circle(screen_copied, tuple(((loc + self.network_input_loctr) * 8.).astype(int)), 12, (128, 255, 128), -1)

        screen_array = []
        cur_record = []

        with self.sess.as_default():
            for npc in self.close_npcs:
                tr = npc.get_transform()
                v = npc.get_velocity()
                
                pos = (np.array([tr.location.x, tr.location.y]) + self.network_input_loctr) * 8.

                M1 = np.float32( [ [1, 0, -pos[0]], [0, 1, -pos[1]], [0, 0, 1] ] )
                M2 = cv2.getRotationMatrix2D((0, 0), tr.rotation.yaw + 90, 1.0)
                M2 = np.append(M2, np.float32([[0, 0, 1]]), axis=0)
                M3 = np.float32( [ [1, 0, 128], [0, 1, 192], [0, 0, 1] ] )
                M = np.matmul(np.matmul(M3, M2), M1)
                rotated = cv2.warpAffine(screen_copied, M[:2], (256, 256))

                screen_array.append(rotated.astype(np.float32) / 128. - 1.)
                cur_record.append([tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y])

            prob, accel = self.learner.get_result(screen_array, cur_record)
            self.prob = prob
            self.accel = accel

    def get_target_speed(self, target_velocity_in_scenario, route, visualize=False):
        target_velocity = target_velocity_in_scenario # HO ADDED 20.0
        sff_potential = 0.0
        final_sff = None

        if self.player != None:
            agent_tr = self.player.get_transform()
            agent_v = self.player.get_velocity()
            M = cv2.getRotationMatrix2D((512, 512), agent_tr.rotation.yaw + 90, 1.0)

            locx = 512 - int(agent_tr.location.x * 8)
            locy = 512 - int(agent_tr.location.y * 8)
            loctr = np.array([locx, locy], np.int32)

            self.close_npcs = []
            for npc in self.npcs:
                loc = npc.get_transform().location
                front_loc = loc + npc.get_transform().get_forward_vector() * 5.
                if np.sqrt( (agent_tr.location.x - loc.x) ** 2 + (agent_tr.location.y - loc.y) ** 2 ) < 256: ##DISTANCE
                    if np.sqrt( (agent_tr.location.x - front_loc.x) ** 2 + (agent_tr.location.y - front_loc.y) ** 2 ) > 3:
                        self.close_npcs.append(npc)

            if len(self.close_npcs) > 0:
                self.Get_Predict_Result()

                screen = np.zeros((1024, 1024), np.uint8)
                line_screen = np.zeros((1024, 1024), np.uint8)

                route_line = [[512, 512]]
                for i, waypoint in enumerate(route[1:]):
                    route_line.append([locx + int(waypoint.location.x * 8), locy + int(waypoint.location.y * 8)])
                    if i == 20:
                        break
                route_line = np.array([route_line], dtype=np.int32)
                cv2.polylines(line_screen, route_line, False, (255,), 20)

                #vx_array = -record[step:step+50, :, 3] * sin_array + record[step:step+50, :, 4] * cos_array
                #vy_array = -record[step:step+50, :, 3] * cos_array - record[step:step+50, :, 4] * sin_array

                new_screen = np.zeros((4, 1024, 1024), np.uint8)
                for npci, npc in enumerate(self.close_npcs):
                    tr = npc.get_transform()
                    v = npc.get_velocity()
                
                    lx = locx + int(tr.location.x * 8)
                    ly = locy + int(tr.location.y * 8)
                    f = tr.get_forward_vector()

                    for i in range(4):
                        accx = self.accel[npci][i * 2]
                        accy = self.accel[npci][i * 2 + 1]

                        yaw = tr.rotation.yaw * 0.017453293
                        ax = accx * np.sin(yaw) + accy * np.cos(yaw)
                        ay = accx * np.cos(yaw) - accy * np.sin(yaw)
                        
                        x = lx + f.x * 12
                        y = ly + f.x * 12
                        vx = float(v.x)
                        vy = float(v.y)
                        line = [[lx - f.x * 12, ly - f.x * 12]]
                        for j in range(11):
                            x += vx / 5 * 8
                            y += vy / 5 * 8
                            line.append([x, y])

                            vx = vx * 0.99 + ax * 4 / 10
                            vy = vy * 0.99 + ay * 4 / 10

                        x += vx / 20 * 128
                        y += vy / 20 * 128 

                        color = 64
                        cv2.polylines(new_screen[i], np.array([line], dtype=np.int32), False, (color,), 20)
                for i in range(4):
                    blurred1 = cv2.GaussianBlur(new_screen[i], (0, 0), 11)
                    screen = cv2.add(screen, blurred1)

                final_sff = cv2.warpAffine(screen, M, (1024,1024))
                final_line = cv2.warpAffine(line_screen, M, (1024,1024))
                final_sff = final_sff[64:576, 256:768]
                final_line = final_line[64:576, 256:768]
                #cv2.imshow("final", final)
                #cv2.imshow("final_line", final_line)

                final = cv2.resize(final_sff[192:448, 128:384], (64, 64), interpolation=cv2.INTER_AREA)
                final_line = cv2.resize(final_line[192:448, 128:384], (64, 64), interpolation=cv2.INTER_AREA)

                #final2 = final[48:108, 118:138]
                #cv2.imshow("SafetyPotential", final2)
                #cv2.waitKey(1)
                final_mean = np.clip(np.max(final_line.astype(np.float32) * final.astype(np.float32) / 25600., axis=1), 0., 1.)
                sff_potential = np.mean(final_mean) 

                for i in range(60):
                    new_velocity = 20. - 0.35 * i
                    target_velocity = target_velocity * (1 - final_mean[i]) + new_velocity * final_mean[i]

                if target_velocity < 1: # HO ADDED
                    target_velocity = 0.

                if target_velocity < 0.:
                    target_velocity = 0.
                
            if self.visualize:
                visual_output = np.zeros((1024, 2048, 3), np.uint8)
                actor_speed = np.sqrt(agent_v.x ** 2 + agent_v.y ** 2)
                if self.img_topview is not None:
                    sff_visual = np.zeros((512, 512, 3), np.uint8)
                    line_visual = np.zeros((1024, 1024, 3), np.uint8)

                    my_sff_visual = np.zeros((1024, 1024, 3), np.uint8)
                    f = agent_tr.get_forward_vector()

                    expected_distance = 0
                    v = actor_speed
                    for j in range(11):
                        expected_distance += v / 5
                        if v > target_velocity:
                            v = v * 0.9 - 11.0 * 0.4
                        else:
                            v = v * 0.9 + 1.7 * 0.4
                    if expected_distance < 3:
                        expected_distance = 3

                    cv2.line(my_sff_visual, (512 - int(f.x * 12), 512 - int(f.y * 12)), 
                        (512 + int(f.x * 8 * (1.5 + expected_distance)), 512 + int(f.y * 8 * (1.5 + expected_distance))), (255, 0, 0), 20)
                    my_sff_visual = cv2.GaussianBlur(my_sff_visual, (0, 0), 11)
                    my_sff_visual = cv2.warpAffine(my_sff_visual, M, (1024, 1024))
                    my_sff_visual = my_sff_visual[64:576, 256:768]
                    my_sff_visual = cv2.resize(my_sff_visual, (1024, 1024), interpolation=cv2.INTER_LINEAR)

                    #cv2.polylines(line_visual, route_line, False, (0, 255, 0), 2)

                    bb = self.player.bounding_box.get_world_vertices(self.player.get_transform())
                    bb_list = [[locx + int(bb[0].x * 8), locy + int(bb[0].y * 8)], [locx + int(bb[2].x * 8), locy + int(bb[2].y * 8)], 
                            [locx + int(bb[6].x * 8), locy + int(bb[6].y * 8)], [locx + int(bb[4].x * 8), locy + int(bb[4].y * 8)]]
                    cv2.polylines(line_visual, np.array([bb_list], dtype=np.int32), True, (255, 0, 0), 2)
                    for npci, npc in enumerate(self.npcs):
                        bb = npc.bounding_box.get_world_vertices(npc.get_transform())
                        bb_list = [[locx + int(bb[0].x * 8), locy + int(bb[0].y * 8)], [locx + int(bb[2].x * 8), locy + int(bb[2].y * 8)], 
                                [locx + int(bb[6].x * 8), locy + int(bb[6].y * 8)], [locx + int(bb[4].x * 8), locy + int(bb[4].y * 8)]]
                        cv2.polylines(line_visual, np.array([bb_list], dtype=np.int32), True, (0, 0, 255), 2)
                    
                    if final_sff is not None:
                        sff_visual[:, :, 2] = final_sff
                    sff_visual = cv2.resize(sff_visual, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    line_visual = cv2.warpAffine(line_visual, M, (1024, 1024))
                    line_visual = line_visual[64:576, 256:768]
                    line_visual = cv2.resize(line_visual, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    mask = np.mean(line_visual, axis=2, dtype=np.uint8)

                    final_visual = cv2.addWeighted(self.img_topview, 0.5, sff_visual, 1.0, 0)
                    final_visual = cv2.add(final_visual, my_sff_visual)
                    cv2.copyTo(line_visual, mask, final_visual)
                    visual_output[:, :1024] = final_visual
                if self.img_frontview is not None:
                    visual_output[:512, 1024:] = self.img_frontview
                cv2.putText(visual_output, "Current Speed : %dkm/h" % int(actor_speed * 3.6), (1040, 560), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), thickness=2)
                cv2.putText(visual_output, "Target Speed : %dkm/h" % int(round(target_velocity * 3.6)), (1040, 608), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), thickness=2)
                cv2.putText(visual_output, "Safety Potential : %.3f" % sff_potential, (1040, 656), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), thickness=2)
                cv2.imshow("visual_output", visual_output)

                if self.record_video:
                    self.video.write(visual_output)

            cv2.waitKey(1)

        if target_velocity < 0.:
            target_velocity = 0.

        #print(target_velocity)
        return target_velocity
            #cv2.imshow("SafetyPotential2", final2)
            #cv2.waitKey(1)

    def on_cam_topview_update(self, image):
        if not image:
            return

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_image = np.reshape(image_data, (image.height, image.width, 4))
        np_image = np_image[:, :, :3]
        np_image = np_image[:, :, ::-1]
        #np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        #self.img_topview = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        self.img_topview = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    def on_cam_frontview_update(self, image):
        if not image:
            return

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_image = np.reshape(image_data, (image.height, image.width, 4))
        np_image = np_image[:, :, :3]
        np_image = np_image[:, :, ::-1]
        self.img_frontview = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    def destroy(self):
        if self.visualize:
            if self.cam_topview:
                self.cam_topview.stop()
                self.cam_topview.destroy()
            if self.cam_frontview:
                self.cam_frontview.stop()
                self.cam_frontview.destroy()
            if self.record_video:
                self.video.release()