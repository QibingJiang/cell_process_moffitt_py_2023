#!/usr/bin/env python3
import cv2
import copy
from cell_detect import CellDetector
from cell_classify import CellClassifier
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
import tifffile as tiff

from phagocytosis_detect import PhagocytosisDetector

import multiprocessing 

import time
import imageio
from itertools import chain
from operator import add
from statistics import mean

debug = 0
# crop_width = 1328
# crop_height = 1048
# crop_width = 512
# crop_height = 512
# crop_width = 256
# crop_height = 256
# crop_width = 128
# crop_height = 128
crop_width = 1360
crop_height = 1024

scale = 1
# scale = 8
line_thick = 1
debug = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

parameters = [["Pt204", [1, 73], "{0:0=1d}", "f00d1.PNG"],
                ["Pt211", [41], "{0:0=2d}", "f00d0.PNG"]]

sample_index = 1

def main():

    # beacons = chain(range(1, 6, 1), range(11, 16, 1), range(73, 78, 1), range(83, 88, 1), range(54, 59, 1), range(64, 69, 1))
    # beacons = chain(range(1, 6, 1), range(11, 16, 1), range(73, 78, 1), range(83, 88, 1))
    beacons = parameters[sample_index][1]

    path = "/home/qibing/disk_t/"
    processes = []

    for pt in [parameters[sample_index][0]]:#"Pt211", "Pt210"

        beacon_count = 0
        for beacon in beacons:
            input_path = path + pt + "/RawData/Beacon-" + str(beacon) + "/"
            process_one_video_main(input_path, beacon, 1)

            # try:
            #     p = multiprocessing.Process(target=process_one_video_main, args=(input_path, beacon, 1))
            #     p.start()
            #     processes.append(p)
            #
            # except Exception as e:  # work on python 3.x
            #     print('Exception: ' + str(e))

            beacon_count = beacon_count + 1

        print(len(processes), " processes are running.")
        for p in processes:
            p.join()
        print("All processes ended.")


def process_one_video_main(path, Beacon, data_type):

    t0 = time.time()
    t0_str = time.ctime(t0)

    scale = 1
    # path = "/home/qibing/Work2/dataset1/"
    path = "/home/qibing/Work2/OneDrive_1_11-2-2020/"
    prepro_frames_iowa_1(path, Beacon)
    # if(os.path.exists(path + "Thumbs.db")):
    #     os.remove(path + "Thumbs.db")
    #     print(path + "Thumbs.db has been removed.")
    #
    # image_num = len(os.listdir(path))
    #
    # print(image_num, path)

    detector = CellDetector()
    classifier = CellClassifier(50, 0, 5, 0)
    # ph_detector = PhagocytosisDetector(10, 30, 5, 0)
    frame_count = 0

    out = None
    det_out = None
    tra_out = None
    # make_video = False
    # make_video = True
    make_video = False

    file = open(path + "detect_result.txt", "w")

    while True:
        # print(str(Beacon) + "_" + str(frame_count), end = " ")
        ret, frame = read_frame(path, Beacon, frame_count, data_type, scale)
        # ret, frame_org = read_frame(path, Beacon, frame_count, 2, scale)
        # frame_org = frame.astype(np.uint8)

        # if(frame_count > 10):
        #     break

        if(ret == False):
            print("done")
            break

        frame_org = frame.copy()

        # print((frame_org.shape[1], frame_org.shape[0]))

        frame_det, centers = detector.detect_by_white_core(frame, scale, frame_count)
        # centers = detector.detect_hybrid(frame, scale)
        # centers = detector.detect_edge_test(frame, scale)



        cell_count = 0
        print(frame_count, len(centers))

        # path = "/home/qibing/disk_t/dataset/Fluo-N2DL-HeLa/"

        if len(centers) > 0:

            # def match_and_mitosis(self, centers_s, frame_prev, frame, frame_index, scale)

            frame_tra = classifier.match_and_mitosis(centers, None, frame, frame_count, scale)
            # ph_detector.match_track(centers, frame, frame_count)

            # cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Tracking', 900, 900)
            # cv2.imshow('Tracking', frame_tra)
            # cv2.waitKey()

            # for point in centers:
            #     cv2.circle(frame, (point[0] * scale, point[1] * scale), 5 * scale, (255, 255, 0), 4)

            colors = [(255, 255, 0), (255, 0, 255), (0, 255, 0)]
            # for i in range(len(centers)):
                # print(len(arr), arr[:, 3].sum(), end=" ")
                # cell_count = cell_count + len(centers[i])
                # file.write(str(len(centers[i])) + " " + str(centers[i][:, 3].sum()) + " ")
                # cv2.circle(frame, (int(centers[i][0]), int(centers[i][1])), 25, (255, 255, 0), 1)
                # for point in centers[i]:
                #     cv2.circle(frame_org, (int(point[0]), int(point[1])), 6, colors[i], 1)

            print("\n")
            file.write("\n")

            pass
        else:
            print("not detected")

        # cv2.putText(frame, str(frame_count) + " " + str(cell_count), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))
        cv2.putText(frame_org, str(frame_count) + " " + str(cell_count), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), int(0.3))
        # cv2.putText(frame, str(len(centers)), (30*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))

        if make_video == True:
            if det_out is None:
                det_out = cv2.VideoWriter(path + "cell_detect_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 1.0, (frame_det.shape[1], frame_det.shape[0]), isColor=True)
            if tra_out is None:
                tra_out = cv2.VideoWriter(path + "cell_track_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 1.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)

        if (det_out != None):
            det_out.write(frame_det)
        if (tra_out != None):
            tra_out.write(frame_tra)

        frame_count = frame_count + 1


    file.close()

    print("Done!")

    if (out != None):
        out.release()

    # classifier.save_track_cell_track_challenge(path)

    # classifier.analyse_classification(path, frame_count)
    # ph_detector.analyse(path)

    classifier.analyse_mitosis(path, frame_count)
    mark(classifier, path, None, scale)

    cv2.destroyAllWindows()

    t1 = time.time()
    t1_str = time.ctime(t1)

    if (not os.path.exists(path)):
        os.makedirs(path)

    log = t0_str + "\n" + t1_str + "\n" + str((t1 - t0)) + " seconds.\n"

    log_file = open(path + "/" + t0_str + "_log.txt", 'w')
    log_file.write(log)
    # print(log)
    log_file.close()
    # np.savetxt("/home/qibing/disk_t/" + pt + "/log", log)


def mark(worker, path, Beacon, scale):

    out2 = None
    frame_count = 0

    while True:
        ret, frame = read_frame(path, Beacon, frame_count, 1, 1)
        if(ret == False):
            print("done")
            return

        print("make cells frame_count:" + str(frame_count))

        if(len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame = worker.mark_mitosis(frame, frame_count, 1, path, frame_count)
        # ph_detector.mark_cells(frame, frame_count)


        if(not os.path.exists(path)):
            os.makedirs(path)

        if(out2 is None):
            out2 = cv2.VideoWriter(path + "cell_marked_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4",fourcc, 1.0, (frame.shape[1], frame.shape[0]), isColor=True)
            # print("VideoWriter", frame.shape[1], frame.shape[0])

        # print(frame.shape[1], frame.shape[0])
        # cv2.putText(frame, str(frame_count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
        out2.write(frame)

        frame_count = frame_count + 1

    print("Done!")
    # cap2.release()
    if(out2):
        out2.release()
    cv2.destroyAllWindows()

# def read_frame(path, Beacon, frame_count, data_type, scale):
#
#     if (data_type == 0):#read from raw image data
#         video_full_path = path + "input_images/"
#         image_path = video_full_path + str(frame_count) + ".npy" #In default case, npy files have been scaled 8 times.
#         ret = os.path.exists(image_path)
#         if ret != True:
#             print("File not exists, ", image_path)
#             return False, None
#
#         frame = np.load(image_path)
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         frame = frame[0:crop_height * scale, 0:crop_width * scale]
#         return True, frame
#     elif (data_type == 1 or data_type == 2):  # read from jpg png tiff
#         input_path = path
#
#         index_0 = chr(ord('A') + int((Beacon - 1) / 24))
#         index_1 = (Beacon - 1) % 24 + 1
#
#         index_1_str = "{0:0=2d}".format(index_1)
#
#         frame_count_str = parameters[sample_index][2].format(frame_count)
#
#         # image_path = input_path + "scan_Plate_D_p" + frame_count_str + "_0_" + index_0 + index_1_str + parameters[sample_index][3]
#
#         # image_path = "/home/qibing/disk_t/dataset/Fluo-N2DL-HeLa/01/t" + "{0:0=3d}".format(frame_count) + ".tif"
#         image_path = "r02c01f01p01-ch1sk" + "{0:0=1d}".format(frame_count + 1) + "fk1fl1.tiff"
#         # image_path = "/home/qibing/disk_t/dataset/Fluo-N2DH-GOWT1/Fluo-N2DH-GOWT1/01/t" + "{0:0=3d}".format(frame_count) + ".tif"
#
#         if (not os.path.exists(image_path)):
#             print("file not exist: ", image_path)
#             return False, None
#         else:
#             # print(frame_count, image_path)
#             pass
#
#         frame = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
#         # return True, frame
#         # frame = cv2.imread(image_path)
#         # frame = imageio.imread(image_path)
#
#         # equ = cv2.equalizeHist(frame)
#         # frame = np.hstack((frame, equ))
#
#         clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(32, 32))
#         # clahe = cv2.createCLAHE(clipLimit=500.0, tileGridSize=(8, 8))
#         frame = clahe.apply(frame)
#         # frame = cv2.blur(frame, (10, 10))
#
#         # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('frame', 900, 900)
#         # cv2.imshow('frame', frame)
#         # cv2.waitKey()
#
#         return True, frame
#
#         # frame = imageio.imread(image_path)
#         out_path = path + "input_images/"
#         if(not os.path.exists(out_path)):
#             os.makedirs(out_path)
#
#         frame = frame[0:crop_height, 0:crop_width]
#         if(data_type == 1):
#             frame = preprocess_2(frame, out_path + str(frame_count), scale)
#         return True, frame
#
#     # frame = frame[0:crop_height * scale, 0:crop_width * scale]
#     # return True, frame

# def read_frame(path, Beacon, frame_count, data_type, scale):
#
#     if (data_type == 0):#read from raw image data
#         video_full_path = path + "input_images/"
#         image_path = video_full_path + str(frame_count) + ".npy" #In default case, npy files have been scaled 8 times.
#         ret = os.path.exists(image_path)
#         if ret != True:
#             print("File not exists, ", image_path)
#             return False, None
#
#         frame = np.load(image_path)
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         frame = frame[0:crop_height * scale, 0:crop_width * scale]
#         return True, frame
#     elif (data_type == 1 or data_type == 2):  # read from jpg png tiff
#         input_path = path
#
#         index_0 = chr(ord('A') + int((Beacon - 1) / 24))
#         index_1 = (Beacon - 1) % 24 + 1
#
#         index_1_str = "{0:0=2d}".format(index_1)
#
#         frame_count_str = parameters[sample_index][2].format(frame_count)
#
#         # image_path = input_path + "scan_Plate_D_p" + frame_count_str + "_0_" + index_0 + index_1_str + parameters[sample_index][3]
#
#         image_path = "/home/qibing/disk_t/dataset/Fluo-N2DH-GOWT1/Fluo-N2DH-GOWT1/01/t" + "{0:0=3d}".format(frame_count) + ".tif"
#
#         if (not os.path.exists(image_path)):
#             print("file not exist: ", image_path)
#             return False, None
#         else:
#             # print(frame_count, image_path)
#             pass
#
#         frame = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
#
#         cv2.namedWindow('org', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('org', 900, 900)
#         cv2.imshow("org", frame)
#
#
#         # return True, frame
#         # cv2.imshow("frame", frame)
#         # cv2.waitKey()
#
#
#         # frame = cv2.imread(image_path)
#         # frame = imageio.imread(image_path)
#
#         # equ = cv2.equalizeHist(frame)
#         # frame = np.hstack((frame, equ))
#
#         # print((frame))
#
#         # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
#         # # clahe = cv2.createCLAHE(clipLimit=500.0, tileGridSize=(8, 8))
#         # frame = clahe.apply(frame)
#
#         grid_size = (24, 24)
#         # grid_size = (100, 100)
#         clip_limit = 1000.0/256
#         # clip_limit = 256
#
#         clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
#         frame = clahe.apply(frame)
#
#         # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
#         # frame = clahe.apply(frame)
#         #
#         # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
#         # frame = clahe.apply(frame)
#
#         frame = cv2.blur(frame, (10, 10))
#
#         cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('frame', 900, 900)
#         cv2.imshow("frame", frame)
#         # cv2.waitKey()
#
#         return True, frame
#
#         # frame = imageio.imread(image_path)
#         out_path = path + "input_images/"
#         if(not os.path.exists(out_path)):
#             os.makedirs(out_path)
#
#         frame = frame[0:crop_height, 0:crop_width]
#         if(data_type == 1):
#             frame = preprocess_2(frame, out_path + str(frame_count), scale)
#         return True, frame
#
#     # frame = frame[0:crop_height * scale, 0:crop_width * scale]
#     # return True, frame

def read_frame(path, Beacon, frame_count, data_type, scale):

    if (data_type == 0):#read from raw image data
        video_full_path = path + "input_images/"
        image_path = video_full_path + str(frame_count) + ".npy" #In default case, npy files have been scaled 8 times.
        ret = os.path.exists(image_path)
        if ret != True:
            print("File not exists, ", image_path)
            return False, None

        frame = np.load(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = frame[0:crop_height * scale, 0:crop_width * scale]
        return True, frame
    elif (data_type == 1 or data_type == 2):  # read from jpg png tiff
        image_path = path + "input_images/" + "t" + "{0:0=3d}".format(frame_count) + ".tif"
        if ((not os.path.exists(image_path))): #  or frame_count > 30
            print("file not exist: ", image_path)
            return False, None
        else:
            # print(frame_count, image_path)
            pass

        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # x = 619
        # y = 321
        # frame = frame[y:y + crop_height, x:x + crop_width]
        frame = frame[0:crop_height, 0:crop_width]
        if (scale > 1):
            frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        return True, frame


def preprocess(frame, save_image_path):

    # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('preprocess_0', 900, 900)
    # cv2.imshow('preprocess_0', frame)

    if (len(frame.shape) > 2):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = frame

    width = frame.shape[1]
    height = frame.shape[0]
    box_side = 64
    vertical = int(height / box_side + 1)
    horizontal = int(width / box_side + 1)

    for i in range(vertical):
        for j in range(horizontal):
            start_x = box_side * j
            start_y = box_side * i

            end_x = 0
            end_y = 0

            if(i < vertical and j < horizontal):
                end_x = start_x + box_side
                end_y = start_y + box_side
            elif(j ==  horizontal):
                end_x = 1328
                end_y = start_y + box_side
            elif(i == vertical):
                end_x = start_x + box_side
                end_y = 1048


            window = frame[start_y:end_y, start_x:end_x]

            histSize = 256
            histRange = (0, 256)  # the upper boundary is exclusive
            accumulate = False
            window_b_hist = cv2.calcHist([window], [0], None, [histSize], histRange, accumulate=accumulate)

            window_hist_max_index = np.argmax(window_b_hist)

            hist_max_index = window_hist_max_index
            hist_trans = np.zeros(256, dtype=np.uint8)
            target_peak = 100

            for k in range(hist_max_index):
                hist_trans[k] = target_peak * k / hist_max_index + 0.5

            for m in range(hist_max_index, 256):
                hist_trans[m] = ((255 - target_peak) * m - 255 * hist_max_index + target_peak * 255) / (255 - hist_max_index) + 0.5

            window = hist_trans[window]
            frame[start_y:end_y, start_x:end_x] = window

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))#clipLimit=2.0,
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))#clipLimit=2.0,
    # gray = clahe.apply(gray)
    frame = clahe2.apply(frame)

    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking', 900, 900)
    cv2.imshow('Tracking', frame)
    cv2.waitKey()

    np.save(save_image_path, frame)
    new_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return new_frame

def preprocess_2(frame, save_image_path, scale = 1):
    # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('preprocess_0', 900, 900)
    # cv2.imshow('preprocess_0', frame)

    if (len(frame.shape) > 2):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_org = frame.copy()

    t0 = time.time()
    frame = cv2.medianBlur(frame_org, 81)#There is an unexpected effect when ksize as 81, applied to 8 times scaled image.

    frame = frame / 100
    frame = frame_org / frame
    np.clip(frame, 0, 255, out = frame)
    frame = frame.astype(np.uint8)

    if(scale > 1):
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation = cv2.INTER_CUBIC)

    t1 = time.time()

    # print("time: ", t1 - t0)

    # np.save(save_image_path, frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # cv2.namedWindow('preprocess_2', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('preprocess_2', 900, 900)
    # cv2.imshow('preprocess_2', frame)
    #
    # cv2.waitKey()
    return frame

def prepro_frames_iowa_0(path, Beacon):

    frame_count = 0
    image_a = None
    image_b = None

    input_path = path
    last_vec = None
    motion_vectors = []
    out_path = path + "input_images/"
    if (not os.path.exists(out_path)):
        os.makedirs(out_path)

    print("ajust luminance")
    while True: # adjust luminance and record cam move
    # while frame_count < 10:
    #     print(frame_count, end=' ', flush=True)
        index_0 = chr(ord('A') + int((Beacon - 1) / 24))
        index_1 = (Beacon - 1) % 24 + 1
        index_1_str = "{0:0=2d}".format(index_1)
        frame_count_str = parameters[sample_index][2].format(frame_count)
        # image_path = input_path + parameters[sample_index][4] + frame_count_str + "_0_" + index_0 + index_1_str + parameters[sample_index][3]
        image_path = input_path + "r02c01f01p01-ch1sk" + "{0:0=1d}".format(frame_count + 1) + "fk1fl1.tiff"

        if (not os.path.exists(image_path)):
            print("file not exist: ", image_path)
            break
        else:
            # print(frame_count, image_path)
            pass

        # frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        # frame = cv2.imread(image_path)
        # frame = imageio.imread(image_path)
        # cv2.imshow("frame", frame)
        # cv2.waitKey()
        # adjust luminance

        # if (len(frame.shape) > 2):
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_org = frame.copy()
        # frame = cv2.medianBlur(frame_org, 81)  # There is an unexpected effect when ksize as 81, applied to 8 times scaled image.
        # frame = frame / 100
        # frame = frame_org / frame
        # np.clip(frame, 0, 255, out=frame)
        # frame = frame.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(32, 32))
        # clahe = cv2.createCLAHE(clipLimit=30000.0, tileGridSize=(16, 16))
        frame = clahe.apply(frame)
        frame = frame/256
        frame = frame.astype(np.uint8)
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 900, 900)
        # cv2.imshow('frame', frame)
        # cv2.waitKey()

        out_image_path = out_path + "t" + "{0:0=3d}".format(frame_count) + ".tif"
        cv2.imwrite(out_image_path, frame)

        # calculate camera movement for all frames.
        if(frame_count == 0):
            image_b = frame
            last_vec = [0, 0]
        else:
            image_a = image_b
            image_b = frame

            d0 = image_a.shape[0] >> 2
            d1 = image_a.shape[1] >> 2
            template = image_a[d0:3 * d0, d1:3 * d1]
            ret = cv2.matchTemplate(image_b, template, cv2.TM_SQDIFF)
            resu = cv2.minMaxLoc(ret)

            if(frame_count == 1):
                last_vec = [resu[2][1] - d0, resu[2][0] - d1]
            else:
                # last_vec = last_vec + [resu[2][1] - d0, resu[2][0] - d1]
                last_vec = list(map(add, last_vec, [resu[2][1] - d0, resu[2][0] - d1]))

        motion_vectors.append(last_vec)
            # print(last_vec, end = " ")
        #
        frame_count = frame_count + 1
    print()

    # print("motion_vectors", motion_vectors)
    motion_vectors_arr = np.asarray(motion_vectors)
    average = [mean(motion_vectors_arr[:,0]), mean(motion_vectors_arr[:,1])]
    # print("average", average)
    motion_vectors_arr = motion_vectors_arr - average
    # print("motion_vectors_arr", motion_vectors_arr)

    np.savetxt(out_path + "motion_vectors.txt", motion_vectors_arr, fmt='%d')

    ret = cv2.minMaxLoc(motion_vectors_arr)
    pad_wid = int(max(abs(ret[0]), abs(ret[1])))

    print("stable images")
    for i in range(frame_count):
        # print(i, end=' ', flush=True)
        image_path = out_path + "t" + "{0:0=3d}".format(i) + ".tif"
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        frame_pad = cv2.copyMakeBorder(frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
        # new_frame = np.zeros((frame.shape[0] + motion_vectors_arr[i][0], frame.shape[1] + motion_vectors_arr[i][1]), np.uint8)
        new_frame = frame_pad[pad_wid + motion_vectors_arr[i][0]:pad_wid + motion_vectors_arr[i][0] + frame.shape[0], pad_wid + motion_vectors_arr[i][1]:pad_wid + motion_vectors_arr[i][1] + frame.shape[1]]
        cv2.imwrite(image_path, new_frame)
    # return True, frame
    print()

def prepro_frames_iowa_1(path, Beacon):

    frame_count = 0
    image_a = None
    image_b = None

    input_path = path
    last_vec = None
    motion_vectors = []
    out_path = path + "input_images/"
    if (not os.path.exists(out_path)):
        os.makedirs(out_path)

    print("ajust luminance")
    while True: # adjust luminance and record cam move
        image_path = input_path + "r02c01f01p01-ch2sk" + "{0:0=1d}".format(frame_count + 1) + "fk1fl1.tiff"

        if (not os.path.exists(image_path)):
            print("file not exist: ", image_path)
            break
        else:
            # print(frame_count, image_path)
            pass

        # frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        # frame = frame % 256
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 900, 900)
        # cv2.imshow("frame", frame)
        # cv2.waitKey()

        # hi = plt.hist(frame.flatten(), 256*256 - 1, [0, 256*256 - 1], alpha=0.5, label='Image a')
        # print(frame.shape)
        # a = np.sum(hi[0][0:1000])
        # b = np.sum(hi[0])
        # print(a, b, float(a) / float(b))
        # plt.xlim(0, 1000)
        # plt.show()

        clahe = cv2.createCLAHE(clipLimit=30000.0, tileGridSize=(16, 16))
        frame = clahe.apply(frame)
        frame = cv2.blur(frame, (5, 5))
        frame = frame / 256
        frame = frame.astype(np.uint8)

        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 900, 900)
        # cv2.imshow("frame", frame)
        # cv2.waitKey()

        out_image_path = out_path + "t" + "{0:0=3d}".format(frame_count) + ".tif"
        cv2.imwrite(out_image_path, frame)

        # calculate camera movement for all frames.
        if(frame_count == 0):
            image_b = frame
            last_vec = [0, 0]
        else:
            image_a = image_b
            image_b = frame

            d0 = image_a.shape[0] >> 2
            d1 = image_a.shape[1] >> 2
            template = image_a[d0:3 * d0, d1:3 * d1]
            ret = cv2.matchTemplate(image_b, template, cv2.TM_SQDIFF)
            resu = cv2.minMaxLoc(ret)

            if(frame_count == 1):
                last_vec = [resu[2][1] - d0, resu[2][0] - d1]
            else:
                # last_vec = last_vec + [resu[2][1] - d0, resu[2][0] - d1]
                last_vec = list(map(add, last_vec, [resu[2][1] - d0, resu[2][0] - d1]))

        motion_vectors.append(last_vec)
            # print(last_vec, end = " ")
        #
        frame_count = frame_count + 1
    print()

    # print("motion_vectors", motion_vectors)
    motion_vectors_arr = np.asarray(motion_vectors)
    average = [mean(motion_vectors_arr[:,0]), mean(motion_vectors_arr[:,1])]
    # print("average", average)
    motion_vectors_arr = motion_vectors_arr - average
    # print("motion_vectors_arr", motion_vectors_arr)

    np.savetxt(out_path + "motion_vectors.txt", motion_vectors_arr, fmt='%d')

    ret = cv2.minMaxLoc(motion_vectors_arr)
    pad_wid = int(max(abs(ret[0]), abs(ret[1])))

    print("stable images")
    for i in range(frame_count):
        # print(i, end=' ', flush=True)
        image_path = out_path + "t" + "{0:0=3d}".format(i) + ".tif"
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        frame_pad = cv2.copyMakeBorder(frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
        # new_frame = np.zeros((frame.shape[0] + motion_vectors_arr[i][0], frame.shape[1] + motion_vectors_arr[i][1]), np.uint8)
        new_frame = frame_pad[pad_wid + motion_vectors_arr[i][0]:pad_wid + motion_vectors_arr[i][0] + frame.shape[0], pad_wid + motion_vectors_arr[i][1]:pad_wid + motion_vectors_arr[i][1] + frame.shape[1]]
        cv2.imwrite(image_path, new_frame)
    # return True, frame
    print()

if __name__ == "__main__":
    # execute main
    main()
