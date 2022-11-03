import numpy as np
import cv2
import random
from network.predict_behavior3 import PredictBehavior
import tensorflow.compat.v1 as tf


lanes = []
with open("lane_town10.txt", "rt") as rf:
    for line in rf.readlines():
        lane = []
        for s in line.split("\t"):
            v = s.split(",")
            if len(v) == 2:
                lane.append([float(v[0]), float(v[1])])
        if len(lane) > 0:
            lanes.append(np.array(lane))


screen = np.full((4096, 4096, 3), 128, np.uint8)
loctr = np.array([256, 216])
for lane in lanes:
    for i, _ in enumerate(lane[:-1]):
        dx = lane[i+1][0] - lane[i][0]
        dy = lane[i+1][1] - lane[i][1]
        r = np.sqrt(dx * dx + dy * dy)
        if r > 0.1:
            color = ( int(dx * 127 / r + 128), 128, int(dy * 127 / r + 128) )
            cv2.line(screen, ((lane[i] + loctr) * 8.).astype(np.int32), ((lane[i+1] + loctr) * 8.).astype(np.int32), color, 4)
            #cv2.polylines(screen, [((lane + loctr) * 8.).astype(np.int32)], False, (255, 0, 0), 4)


tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = PredictBehavior()
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    log_file = open("train_log/log6_2/log.txt", "wt")
    log_file.write("Iteration" + learner.log_caption())
    sess.run(tf.global_variables_initializer())
    learner.network_initialize()
    learner_saver.restore(sess, "train_log/log6/iter_1500.ckpt")
        
    for iteration in range(1501, 100000):
        record = np.load("gathered/log1/" + str(random.randrange(1000)) + ".npy")
        record_index = list(range(1, np.shape(record)[0] - 50))
        random.shuffle(record_index)
        for step in record_index[:100]:
            print("%04d" % step)
            screen_copied = screen.copy()
            cur_record = record[step]
            cos_array = np.cos(cur_record[:, 2] * 0.017453293)
            sin_array = np.sin(cur_record[:, 2] * 0.017453293)

            vx_array = -record[step:step+50, :, 3] * sin_array + record[step:step+50, :, 4] * cos_array
            vy_array = record[step:step+50, :, 3] * cos_array + record[step:step+50, :, 4] * sin_array
            
            ax = np.mean(vx_array[1:] - vx_array[:-1], axis=0) * 10.
            ay = np.mean(vy_array[1:] - vy_array[:-1], axis=0) * 10.

            
            target_array = np.stack((ax, ay), axis=1)

            for s in cur_record:
                cv2.circle(screen_copied, tuple(((s[:2] + loctr) * 8.).astype(int)), 12, (128, 255, 128), -1)

            screen_array = []
            for s in cur_record:
                pos = (s[:2] + loctr) * 8.
                #pos = (int(pos[0]), int(pos[1]))
                M1 = np.float32( [ [1, 0, -pos[0]], [0, 1, -pos[1]], [0, 0, 1] ] )
                M2 = cv2.getRotationMatrix2D((0, 0), s[2] + 90, 1.0)
                M2 = np.append(M2, np.float32([[0, 0, 1]]), axis=0)
                M3 = np.float32( [ [1, 0, 128], [0, 1, 192], [0, 0, 1] ] )
                M = np.matmul(np.matmul(M3, M2), M1)
                rotated = cv2.warpAffine(screen_copied, M[:2], (256, 256))
                
                #cv2.imshow("rotated", rotated)
                #print("%2.6f %2.6f" % (ax[i], ay[i]))
                #cv2.waitKey(0)

                #rotated = rotated[pos[1]-192:pos[1]+64, pos[0]-128:pos[0]+128]
                screen_array.append(rotated.astype(np.float32) / 128. - 1.)

            learner.optimize_batch(screen_array, cur_record, target_array)
            learner.log_print()
            learner.network_update()

        
        log_file.write(str(iteration) + learner.current_log() + "\n")
        log_file.flush()
        if iteration % 100 == 0:
            learner_saver.save(sess, "train_log/log6_2/iter_" + str(iteration) + ".ckpt")

