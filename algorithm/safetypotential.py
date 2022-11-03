
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

from network.predict_behavior3 import PredictBehavior
import tensorflow.compat.v1 as tf

class SafetyPotential:
    def __init__(self, lane_txt):
        self.player = None

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


        self.screen = np.full((4096, 4096, 3), 128, np.uint8)
        self.zero_screen = np.full((256, 256, 3), 0., np.float32)
        self.screen_loctr = np.array([256, 216])
        for lane in self.lanes:
            for i, _ in enumerate(lane[:-1]):
                dx = lane[i+1][0] - lane[i][0]
                dy = lane[i+1][1] - lane[i][1]
                r = np.sqrt(dx * dx + dy * dy)
                if r > 0.1:
                    color = ( int(dx * 127 / r + 128), 128, int(dy * 127 / r + 128) )
                    cv2.line(self.screen, tuple(((lane[i] + self.screen_loctr) * 8.).astype(np.int32)), tuple(((lane[i+1] + self.screen_loctr) * 8.).astype(np.int32)), color, 4)
            #cv2.polylines(self.screen, [((lane + self.screen_loctr) * 8.).astype(np.int32)], False, (255, 0, 0), 4)

        tf.disable_eager_execution()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.learner = PredictBehavior()
            learner_saver = tf.train.Saver(var_list=self.learner.trainable_dict, max_to_keep=0)
            learner_saver.restore(self.sess, "/home/user/Documents/Taewoo/carla-sff/train_network/log_train/log6_2/iter_5200.ckpt")

    def Assign_Player(self, player):
        self.player = player

    def Assign_NPCS(self, npcs):
        self.npcs = npcs
        self.acc_buffer = np.array( [ [ [0., 0.] for _ in range(4)] for _ in self.npcs]  )

    def Get_Predict_Result(self):
        screen_copied = self.screen.copy()
        for npc in self.close_npcs:
            tr = npc.get_transform()
            v = npc.get_velocity()
            loc = np.array([tr.location.x, tr.location.y])
            cv2.circle(screen_copied, tuple(((loc + self.screen_loctr) * 8.).astype(int)), 12, (128, 255, 128), -1)

        #screen_array = []
        #cur_record = []

        self.prob = []
        self.accel = []
        with self.sess.as_default():
            for npc in self.close_npcs:
                tr = npc.get_transform()
                v = npc.get_velocity()
                

                pos = (np.array([tr.location.x, tr.location.y]) + self.screen_loctr) * 8.

                M1 = np.float32( [ [1, 0, -pos[0]], [0, 1, -pos[1]], [0, 0, 1] ] )
                M2 = cv2.getRotationMatrix2D((0, 0), tr.rotation.yaw + 90, 1.0)
                M2 = np.append(M2, np.float32([[0, 0, 1]]), axis=0)
                M3 = np.float32( [ [1, 0, 128], [0, 1, 192], [0, 0, 1] ] )
                M = np.matmul(np.matmul(M3, M2), M1)
                rotated = cv2.warpAffine(screen_copied, M[:2], (256, 256))

                #screen_array.append(rotated.astype(np.float32) / 128. - 1.)
                #cur_record.append([tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y])
                #else:
                #    screen_array.append(self.zero_screen)
                #    cur_record.append([0., 0., 0., 0., 0.])
                screen_array = rotated.astype(np.float32) / 128. - 1.
                cur_record = [tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y]
                prob, accel = self.learner.get_result([screen_array], [cur_record])
                self.prob.append(prob[0])
                self.accel.append(accel[0])

            
        #with self.sess.as_default():
        #    self.prob, self.accel = self.learner.get_result(screen_array, cur_record)
         

    def tick(self):
        if self.player != None:
            agent_tr = self.player.get_transform()
            agent_v = self.player.get_velocity()

            self.close_npcs = []
            for npci, npc in enumerate(self.npcs):
                loc = npc.get_transform().location
                front_loc = loc + npc.get_transform().get_forward_vector() * 5.
                if np.sqrt( (agent_tr.location.x - loc.x) ** 2 + (agent_tr.location.y - loc.y) ** 2 ) < 64:
                    if np.sqrt( (agent_tr.location.x - front_loc.x) ** 2 + (agent_tr.location.y - front_loc.y) ** 2 ) > 3:
                        self.close_npcs[npci] = True

            self.Get_Predict_Result()

            screen = np.zeros((2048, 2048, 3), np.uint8)

            locx = 1024 - int(agent_tr.location.x * 16)
            locy = 1024 - int(agent_tr.location.y * 16)
            loctr = np.array([locx, locy], np.int32)
            for lane in self.lanes:
                cv2.polylines(screen, [(lane * 16.).astype(np.int32) + loctr], False, (128, 128, 128), 1)

            #vx_array = -record[step:step+50, :, 3] * sin_array + record[step:step+50, :, 4] * cos_array
            #vy_array = -record[step:step+50, :, 3] * cos_array - record[step:step+50, :, 4] * sin_array

            new_screen = np.zeros((4, 2048, 2048, 3), np.uint8)
            for npci, npc in enumerate(self.close_npcs):
                if self.close_npcs[npci] :
                    tr = npc.get_transform()
                    v = npc.get_velocity()
                
                    lx = locx + int(tr.location.x * 16)
                    ly = locy + int(tr.location.y * 16)
                    cv2.circle(screen, (lx, ly), 16, (0, 255, 255), -1)
                    cv2.arrowedLine(screen, (lx, ly), (lx + int(v.x * 12), ly + int(v.y * 12)), (0, 255, 255), 2)

                    for i in range(4):
                        accx = self.accel[npci][i * 2]
                        accy = self.accel[npci][i * 2 + 1]

                        yaw = tr.rotation.yaw * 0.017453293
                        ax = accx * np.sin(yaw) + accy * np.cos(yaw)
                        ay = accx * np.cos(yaw) - accy * np.sin(yaw)
                        
                        x = lx
                        y = ly
                        vx = float(v.x)
                        vy = float(v.y)
                        line = [[x, y]]
                        for j in range(11):
                            x += vx / 5 * 16
                            y += vy / 5 * 16 
                            line.append([x, y])

                            vx = vx * 0.99 + ax * 4 / 10
                            vy = vy * 0.99 + ay * 4 / 10
                            #ax += jx
                            #ay += jy
                        #color = self.prob[npci][i] * 3.5 * 256 - 128
                        #if color > 255:
                        #    color = 255
                        #elif color < 0:
                        #    color = 0
                        x += vx / 20 * 128
                        y += vy / 20 * 128 
                        #line.append([x, y])

                        #color = self.prob[npci][i] * 3.5 * 128 - 64
                        #if color > 128:
                        #    color = 128
                        #elif color < 0:
                        #    color = 0
                        color = 64
                        cv2.polylines(new_screen[i], np.array([line], dtype=np.int32), False, (0, 0, color), 40)
            for i in range(4):
                blurred1 = cv2.GaussianBlur(new_screen[i], (0, 0), 16)
                blurred2 = cv2.resize(blurred1, (512, 512), interpolation=cv2.INTER_AREA)
                blurred2 = cv2.resize(blurred2, (2048, 2048), interpolation=cv2.INTER_LINEAR)
                #blurred2 = cv2.GaussianBlur(new_screen[i], (0, 0), 36)
                #screen = cv2.add(screen, new_screen[i])
                screen = cv2.add(screen, blurred1)
                screen = cv2.add(screen, blurred2)
                

            cv2.circle(screen, (1024, 1024), 16, (0, 255, 0), -1)
            cv2.arrowedLine(screen, (1024, 1024), (1024 + int(agent_v.x * 12), 1024 + int(agent_v.y * 12)), (0, 255, 0), 4)

            M = cv2.getRotationMatrix2D((1024, 1024), agent_tr.rotation.yaw + 90, 1.0)
            rotated = cv2.warpAffine(screen, M, (2048, 2048))

            final = rotated[512:1280, 640:1408]
            cv2.imshow("SafetyPotential", final)
            cv2.waitKey(1)

    def get_target_velocity(self, route):
        target_velocity = 8.0
        if self.player != None:
            agent_tr = self.player.get_transform()
            agent_v = self.player.get_velocity()

            self.close_npcs = []
            for npc in self.npcs:
                loc = npc.get_transform().location
                front_loc = loc + npc.get_transform().get_forward_vector() * 5.
                if np.sqrt( (agent_tr.location.x - loc.x) ** 2 + (agent_tr.location.y - loc.y) ** 2 ) < 64:
                    if np.sqrt( (agent_tr.location.x - front_loc.x) ** 2 + (agent_tr.location.y - front_loc.y) ** 2 ) > 3:
                        self.close_npcs.append(npc)

            if len(self.close_npcs) > 0:
                self.Get_Predict_Result()

                screen = np.zeros((512, 512), np.uint8)
                line_screen = np.zeros((512, 512), np.uint8)

                locx = 256 - int(agent_tr.location.x * 16)
                locy = 256 - int(agent_tr.location.y * 16)
                loctr = np.array([locx, locy], np.int32)
                
                line = [[256, 256]]
                for i, (waypoint, roadoption) in enumerate(route):
                    line.append([locx + int(waypoint.transform.location.x * 16), locy + int(waypoint.transform.location.y * 16)])
                    if i == 10:
                        break
                cv2.polylines(line_screen, np.array([line], dtype=np.int32), False, (255,), 20)
                

                #vx_array = -record[step:step+50, :, 3] * sin_array + record[step:step+50, :, 4] * cos_array
                #vy_array = -record[step:step+50, :, 3] * cos_array - record[step:step+50, :, 4] * sin_array

                new_screen = np.zeros((4, 512, 512), np.uint8)
                for npci, npc in enumerate(self.close_npcs):
                    tr = npc.get_transform()
                    v = npc.get_velocity()
                
                    lx = locx + int(tr.location.x * 16)
                    ly = locy + int(tr.location.y * 16)

                    for i in range(4):
                        accx = self.accel[npci][i * 2]
                        accy = self.accel[npci][i * 2 + 1]

                        yaw = tr.rotation.yaw * 0.017453293
                        ax = accx * np.sin(yaw) + accy * np.cos(yaw)
                        ay = accx * np.cos(yaw) - accy * np.sin(yaw)
                        
                        x = lx
                        y = ly
                        vx = float(v.x)
                        vy = float(v.y)
                        line = [[x, y]]
                        for j in range(11):
                            x += vx / 5 * 16
                            y += vy / 5 * 16 
                            line.append([x, y])

                            vx = vx * 0.99 + ax * 4 / 10
                            vy = vy * 0.99 + ay * 4 / 10

                        x += vx / 20 * 128
                        y += vy / 20 * 128 

                        color = 64
                        cv2.polylines(new_screen[i], np.array([line], dtype=np.int32), False, (color,), 20)
                for i in range(4):
                    blurred1 = cv2.GaussianBlur(new_screen[i], (0, 0), 11)
                    #blurred1 = cv2.resize(blurred1, (512, 512), interpolation=cv2.INTER_AREA)
                    #blurred1 = cv2.resize(new_screen[i], (256, 256), interpolation=cv2.INTER_AREA)
                    #blurred2 = cv2.GaussianBlur(blurred1, (0, 0), 7)
                    #screen = cv2.add(screen, blurred1)
                    screen = cv2.add(screen, blurred1)
                    

                M = cv2.getRotationMatrix2D((256, 256), agent_tr.rotation.yaw + 90, 1.0)
                final = cv2.warpAffine(screen, M, (512,512))
                final_line = cv2.warpAffine(line_screen, M, (512,512))

                final = cv2.resize(final[:256, 128:384], (64, 64), interpolation=cv2.INTER_AREA)
                final_line = cv2.resize(final_line[:256, 128:384], (64, 64), interpolation=cv2.INTER_AREA)
                #cv2.imshow("Line", final_line)
                #cv2.imshow("SafetyPotential", final)
                #cv2.waitKey(1)

                #final2 = final[48:108, 118:138]
                #cv2.imshow("SafetyPotential", final2)
                #cv2.waitKey(1)
                final_mean = np.clip(np.max(final_line.astype(np.float32) * final.astype(np.float32) / 25600., axis=1), 0., 1.)
                #print(final_mean)

                for i in range(60):
                    new_velocity = 12. - (0.25 * i * final_mean[i])
                    if target_velocity > new_velocity:
                        target_velocity = new_velocity

        if target_velocity < 0.:
            target_velocity = 0.
        #print(target_velocity)
        return target_velocity
            #cv2.imshow("SafetyPotential2", final2)
            #cv2.waitKey(1)

    def get_target_speed(self, route):
        target_velocity = 20.0
        if self.player != None:
            agent_tr = self.player.get_transform()
            agent_v = self.player.get_velocity()
            agent_reltr = agent_tr.location + agent_tr.get_forward_vector() * 3.


            self.close_npcs = []
            for npc in self.npcs:
                loc = npc.get_transform().location
                front_loc = loc + npc.get_transform().get_forward_vector() * 5.
                if np.sqrt( (agent_tr.location.x - loc.x) ** 2 + (agent_tr.location.y - loc.y) ** 2 ) < 256:
                    if np.sqrt( (agent_tr.location.x - front_loc.x) ** 2 + (agent_tr.location.y - front_loc.y) ** 2 ) > 3:
                        self.close_npcs.append(npc)

            if len(self.close_npcs) > 0:
                self.Get_Predict_Result()

                screen = np.zeros((512, 512), np.uint8)
                line_screen = np.zeros((512, 512), np.uint8)

                locx = 256 - int(agent_reltr.x * 8)
                locy = 256 - int(agent_reltr.y * 8)
                loctr = np.array([locx, locy], np.int32)
                
                line = [[256, 256]]
                for i, waypoint in enumerate(route):
                    line.append([locx + int(waypoint.location.x * 8), locy + int(waypoint.location.y * 8)])
                    if i == 20:
                        break
                cv2.polylines(line_screen, np.array([line], dtype=np.int32), False, (255,), 20)
                

                #vx_array = -record[step:step+50, :, 3] * sin_array + record[step:step+50, :, 4] * cos_array
                #vy_array = -record[step:step+50, :, 3] * cos_array - record[step:step+50, :, 4] * sin_array

                new_screen = np.zeros((4, 512, 512), np.uint8)
                for npci, npc in enumerate(self.close_npcs):
                    tr = npc.get_transform()
                    v = npc.get_velocity()
                
                    lx = locx + int(tr.location.x * 8)
                    ly = locy + int(tr.location.y * 8)

                    for i in range(4):
                        accx = self.accel[npci][i * 2]
                        accy = self.accel[npci][i * 2 + 1]

                        yaw = tr.rotation.yaw * 0.017453293
                        ax = accx * np.sin(yaw) + accy * np.cos(yaw)
                        ay = accx * np.cos(yaw) - accy * np.sin(yaw)
                        
                        x = lx
                        y = ly
                        vx = float(v.x)
                        vy = float(v.y)
                        line = [[x, y]]
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
                    #blurred1 = cv2.resize(blurred1, (512, 512), interpolation=cv2.INTER_AREA)
                    #blurred1 = cv2.resize(new_screen[i], (256, 256), interpolation=cv2.INTER_AREA)
                    #blurred2 = cv2.GaussianBlur(blurred1, (0, 0), 7)
                    #screen = cv2.add(screen, blurred1)
                    screen = cv2.add(screen, blurred1)
                    

                M = cv2.getRotationMatrix2D((256, 256), agent_tr.rotation.yaw + 90, 1.0)
                final = cv2.warpAffine(screen, M, (512,512))
                final_line = cv2.warpAffine(line_screen, M, (512,512))
                cv2.imshow("SafetyPotential", final)
                cv2.imshow("SafetyPotential", final_line)

                final = cv2.resize(final[:256, 128:384], (64, 64), interpolation=cv2.INTER_AREA)
                final_line = cv2.resize(final_line[:256, 128:384], (64, 64), interpolation=cv2.INTER_AREA)

                cv2.waitKey(1)

                #final2 = final[48:108, 118:138]
                #cv2.imshow("SafetyPotential", final2)
                #cv2.waitKey(1)
                final_mean = np.clip(np.max(final_line.astype(np.float32) * final.astype(np.float32) / 25600., axis=1), 0., 1.)

                for i in range(60):
                    new_velocity = 20. - 0.35 * i
                    target_velocity = target_velocity * (1 - final_mean[i]) + new_velocity * final_mean[i]

        if target_velocity < 0.:
            target_velocity = 0.
        #print(target_velocity)
        return target_velocity
            #cv2.imshow("SafetyPotential2", final2)
            #cv2.waitKey(1)
