import cv2
import copy
from cell_detect import CellDetector
from cell_classify import CellClassifier
import os
# from matplotlib import pyplot as plt
import numpy as np
# from phagocytosis_detect import PhagocytosisDetector
# import multiprocessing
# import time
# import imageio
# from itertools import chain
# from operator import add
# from statistics import mean
import sys
import re
from util import read_frame
import shutil

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# home_dir = os.path.expanduser("~") + "/"

debug = 0

crop_width = 0
crop_height = 0

# crop_width = 1328
# crop_height = 1048

# crop_width = 512
# crop_height = 512

# crop_width = 256
# crop_height = 256

# crop_width = 730
# crop_height = 1024


scale = 8

line_thick = 1
debug = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# configure_path = sys.argv[1]
# path = sys.argv[2]

# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_t/Pt210/RawData/Beacon-164", out_path = "/home/qibing/disk_t/Pt210/Beacon-164_test_3"):
# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt204/RawData/Beacon-73", out_path = "Default"):
# def main(configure_path = "./configure.txt", path="/home/qibing/disk_t/Pt282_SOCCO/RawData/Beacon-267", out_path="Hello"):
# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt210/RawData/Beacon-83", out_path = "/home/qibing/disk_16t/Pt210/pad_area_2"):
# def main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt204/RawData/Beacon-2", out_path="/home/qibing/disk_16t/Pt204/pad_area_2"):
# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt400_SOCCO/RawData/Beacon-29", out_path = "Default"):

# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt174/RawData/Beacon-74", out_path = "Default"):
# def main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt210/RawData/Beacon-73", out_path="Default"):
# def main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt199/RawData/Beacon-181", out_path="Default"):
# def main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt199/RawData/Beacon-172", out_path="Default"):
# def main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt323_SOCCO/RawData/Beacon-193", out_path="Default"):
# def main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt323_SOCCO/RawData/Beacon-110", out_path="/home/qibing/disk_16t/Pt323_SOCCO/debug/"):
# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt174/RawData/Beacon-121", out_path = "Default"):
# def main(configure_path = "./configure.txt", path = "Work/ground_truth/preprocess", out_path = "Work/ground_truth/output_red_level_20_green_max_pixel_130_radius_20"):
def main(configure_path = "./configure.txt", path = "Work/ground_truth/preprocess", out_path = "Default"):
# def main(configure_path = "./configure.txt", path = "/home/qibing/Work/ground_truth/M C3 10nM BORT ACTUAL 2012-12-21-15-28-13/bright", out_path = "/home/qibing/Work/ground_truth/output"):
# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt196/RawData/Beacon-77", out_path = "/home/qibing/disk_16t/Pt196/Beacon-77_512"):
# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt196/RawData/Beacon-77", out_path = "/home/qibing/disk_16t/Pt196/Beacon-77_512_old_preprocess"):
# if(len(sys.argv) < 3):
    #     print("configure.txt path and input path are not specified.")
    #     return
    # configure_path = sys.argv[1]
    # path = sys.argv[2]
    #
    # if (len(sys.argv) >= 4):
    #     out_path = sys.argv[3]

    # path = home_dir + path

    if(out_path == "Default"):
        out_path = re.sub(r'RawData.*', 'TimeLapseVideos_cbb/', path)
        print("Output directory is: ", out_path)
    # else:
    #     out_path = home_dir + out_path

    ret = re.sub(r'.*Beacon-', '', path)
    Beacon = re.sub(r'/.*', '', ret)

    if(Beacon == ''):
        Beacon = 0
    else:
        Beacon = int(Beacon)

    paras = []
    with open(configure_path) as f:
        for l in f:
            l = re.sub(r'#.*', '', l)# replace the comments with ''
            l = l.replace(" ", "")
            l = l.replace("\n", "")
            if(len(l) > 0):
                paras.append(l.split("="))

    paras_dict = {p[0]:p[1] for p in paras}

    for key in ["cell_core_radius_range_2", "cell_core_radius_range_3"]:
        radius_interval = paras_dict[key]
        radius_interval = radius_interval.replace("(", "")
        radius_interval = radius_interval.replace(")", "")
        radius_interval = radius_interval.split(",")
        radius_interval = [float(radius_interval[0]), float(radius_interval[1])]
        paras_dict[key] = radius_interval

    for key in ["cell_max_1", "black_edge_2", "white_core_2", "white_core_3", "cell_max_3"]:
        paras_dict[key] = float(paras_dict[key])

    print("Mode: ", paras_dict["Mode"])

    if (path[-1] != '/'):
        path += '/'

    if (out_path[-1] != '/'):
        out_path += '/'

    os.makedirs(out_path, exist_ok = True)

    for new_folder in ["images_ucf/Beacon_" + str(Beacon) + "/", "Cell_tracks/", "Results/"]:
        os.makedirs(out_path + new_folder, exist_ok = True)

    process_one_video_main(path, Beacon, 1, None, out_path, paras_dict)

def process_one_video_main(path, Beacon, data_type, pt, out_path, paras_dict):
    global crop_width, crop_height
    # t0 = time.time()
    # t0_str = time.ctime(t0)

    detector = CellDetector()
    classifier = CellClassifier(8, 20, 5, 0)
    # ph_detector = PhagocytosisDetector(10, 30, 5, 0)
    image_path = path
    develop = 0
    if(develop == 0):
        detector.prepro_frames_2(image_path, out_path + "images_ucf/Beacon_" + str(Beacon) + "/")
        classifier.image_amount = detector.image_amount

        if(paras_dict["Mode"] == '2'):
            detector.edge_thr = paras_dict["black_edge_2"]
            detector.core_thr = paras_dict["white_core_2"]
            detector.radius_thr = paras_dict["cell_core_radius_range_2"]
            pass
        elif (paras_dict["Mode"] == '3'):
            detector.core_thr = paras_dict["white_core_3"]
            detector.radius_thr = paras_dict["cell_core_radius_range_3"]
            detector.max_pixel = paras_dict["cell_max_3"]
            pass
        else:#this is mode 0 and 1
            # detector.edge_thr = detector.background_pixel_mean
            # detector.core_thr = detector.bg_gau_mean + 3.0 * detector.bg_gau_std
            # detector.radius_thr = [detector.cell_core_r - 3 * detector.cell_core_r_std, detector.cell_core_r + 3 * detector.cell_core_r_std]

            if (paras_dict["Mode"] == '1'):
                detector.max_pixel = paras_dict["cell_max_1"]
                pass
            # detector.max_pixel = 200
    else:
        classifier.image_amount = detector.image_amount = 30
        detector.background_pixel = 100
        detector.edge_thr = detector.background_pixel_mean = 99.57
        detector.background_pixel_std = 18.02
        detector.bg_gau_mean = 100.24
        detector.bg_gau_std = 4.67
        detector.cell_core_r = 2.88
        detector.cell_core_r_std = 0.46
        detector.noise_radius_thresh = 1.0
        detector.core_thr = 120

        detector.radius_thr = [1, 10]
        # detector.image_amount = 10000
        paras_dict["Mode"] = '3'
        print("Developing Mode using Detection Mode: ", paras_dict["Mode"])

    frame_count = 0

    det_out = None
    tra_out = None
    # make_video = True
    make_video = False

    # file = open(image_path + "detect_result_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")

    frame_prev = None
    image_amount_str = str(detector.image_amount)
    print("detect and track:")
    for frame_count in range(detector.image_amount):
        # print(str(Beacon) + "_" + str(frame_count), end = " ", flush=True)
        print("\r", frame_count, end = "/" + image_amount_str, flush=True)
        ret, frame_org = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/", frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)

        if(crop_width == 0 and crop_height == 0):
            crop_width = int(frame_org.shape[1]/scale)
            crop_height = int(frame_org.shape[0]/scale)

        # frame = cv2.imread("/home/qibing/Work/ground_truth/output/images_ucf/Beacon_0/t096.tif", cv2.IMREAD_GRAYSCALE)
        # frame_org = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        # ret = True

        if(ret == False):
            detector.image_amount = frame_count
            print("done")
            break

        frame_det = frame_org.copy()

        # centers = detector.detect_by_white_core(frame_det, scale, frame_count)
        # centers = detector.detect_hybrid(frame, scale)
        # centers = detector.detect_by_contour_ex(frame_det, scale)
        # temp_t = time.time()
        # frame_det, centers = detector.detect_edge_test(frame_det, frame_count, scale)
        if(paras_dict['Mode'] == '0' or paras_dict['Mode'] == '2'):
            # frame_det, centers = detector.detect_by_edge_core_and_level_RFP(out_path, frame_det, frame_count, scale)
            frame_det, centers = detector.detect_by_edge_core_and_level(out_path, frame_det, frame_count, scale)
        elif(paras_dict['Mode'] == '1' or paras_dict['Mode'] == '3'):
            # frame_det, centers = detector.detect_by_edge_core_and_level_RFP(out_path, frame_det, frame_count, scale)
            frame_det, centers = detector.detect_by_white_core_and_level(frame_det, frame_count, scale)
        else:
            print("Mode is not defined: ", paras_dict['Mode'])
            pass
        # print("det t:", time.time() - temp_t)

        cell_count = 0
        if len(centers) > 0:
            # centers_cat = np.vstack(centers)
            frame_tra = frame_org.copy()
            # frame_tra = classifier.match_track(centers_cat, frame_prev, frame_tra, frame_count, scale)
            # temp_t = time.time()
            frame_tra = classifier.match_track_3_times(centers, frame_prev, frame_tra, frame_count, scale)
            # print("match t:", time.time() - temp_t)

            # ph_detector.match_track(centers, frame, frame_count)

            for i in range(len(centers)):
                # print(len(arr), arr[:, 3].sum(), end=" ")
                cell_count = cell_count + len(centers[i])
                # file.write(str(len(centers[i])) + " " + str(centers[i][:, 3].sum()) + " ")

                # if(centers_cat == None):
                #     centers_cat = centers[i]
                # else:
                #     centers_cat = np.concatenate((centers_cat, centers[i]), axis=0)

                # for point in centers[i]:
                    # cv2.circle(frame_org, (int(point[0]), int(point[1])), 6, colors[i], 1)

            # print(centers_cat)
            pass
        else:
            print("not detected")

        # cv2.putText(frame, str(frame_count) + " " + str(cell_count), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))
        # cv2.putText(frame_org, str(frame_count) + " " + str(cell_count), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), int(0.3))
        # cv2.putText(frame, str(len(centers)), (30*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))

        if make_video == True:
            if det_out is None:
                # det_out = cv2.VideoWriter(out_path + "cell_detect_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_det.shape[1], frame_det.shape[0]), isColor=True)
                det_out = cv2.VideoWriter(out_path + "cell_detect.mp4", fourcc, 3.0, (frame_det.shape[1], frame_det.shape[0]), isColor=True)
            if tra_out is None:
                # tra_out = cv2.VideoWriter(out_path + "cell_track_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)
                tra_out = cv2.VideoWriter(out_path + "cell_track.mp4", fourcc, 3.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)

        if (det_out != None):
            det_out.write(frame_det)
        if (tra_out != None):
            tra_out.write(frame_tra)

        # cv2.namedWindow('det', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('det', 900, 900)
        # cv2.imshow('det', frame_det)
        # cv2.imwrite(image_path + "det_" + str(frame_count) + ".png", frame_det[154:625, 748:1340, :])
        # cv2.waitKey()

        # cv2.namedWindow('tra', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('tra', 900, 900)
        # cv2.imshow('tra', frame_tra)
        # cv2.imwrite(image_path + "tra_" + str(frame_count) + ".png", frame_tra[154:625, 748:1340, :])
        # cv2.waitKey()

        # cv2.imwrite(out_path + "det_" + str(frame_count) + ".png", frame_det)
        # cv2.imwrite(out_path + "tra_" + str(frame_count) + ".png", frame_tra)

        frame_prev = frame_org.copy()
    print()


    # print("Done!")

    if (det_out != None):
        det_out.release()

    if (tra_out != None):
        tra_out.release()

    classifier.background_pixel = detector.background_pixel
    classifier.cell_core_r = detector.cell_core_r
    classifier.cell_core_r_mean = detector.cell_core_r_mean


    # classifier.analyse_classification_3(image_path, frame_count)
    gt_video_path = ""
    gt_video_path = re.sub(r'RawData.*', 'TimeLapseVideos/', path) + "Beacon-" + str(Beacon) + "processed.avi"
    # gt_video_path = home_dir + "Work/ground_truth/RFP.mp4"
    gt = False
    classifier.analyse_classification_8_win(out_path, detector.image_amount, gt_video_path, scale, Beacon, gt)
    mark_ground_truth(classifier, out_path + "images_ucf/Beacon_" + str(Beacon) + "/", Beacon, data_type, 8, detector.image_amount, out_path, gt_video_path, gt)
    # mark(classifier, path, Beacon, data_type, scale)


    dir_path = out_path + "images_ucf/Beacon_" + str(Beacon) + "/"
    try:
        shutil.rmtree(dir_path)
        # os.remove(dir_path)
        # if(len(os.listdir(out_path + "images_ucf")) == 0):
        #     os.remove(out_path + "images_ucf")
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


    cv2.destroyAllWindows()

    # t1 = time.time()
    # t1_str = time.ctime(t1)
    #
    # if (not os.path.exists(image_path)):
    #     os.makedirs(image_path)
    # log = t0_str + "\n" + t1_str + "\n" + str((t1 - t0)) + " seconds.\n"
    # log_file = open(image_path + "/" + t0_str + "_log.txt", 'w')
    # log_file.write(log)
    # print(log)
    # log_file.close()
    # np.savetxt("/home/qibing/disk_t/" + pt + "/log", log)


# def mark(worker, path, Beacon, data_type, scale):
#
#     out2 = None
#     frame_count = 0
#
#     image_path = path + "/RawData/Beacon-" + str(Beacon) + "/"
#
#     while True:
#         ret, frame = read_frame(image_path, frame_count, data_type, scale)
#         if(ret == False):
#             print("done")
#             return
#
#         # print("mark cells frame_count:" + str(frame_count))
#
#         if(len(frame.shape) == 2):
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#
#         # frame = worker.mark(frame, frame_count, scale)
#         # ph_detector.mark_cells(frame, frame_count)
#         frame = worker.mark_mitosis(frame, frame_count, scale, path, frame_count)
#
#         if(not os.path.exists(image_path)):
#             os.makedirs(image_path)
#
#         if(out2 is None):
#             out2 = cv2.VideoWriter(image_path + "cell_classified_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4",fourcc, 3.0, (crop_width * 3, crop_height * 3), isColor=True)
#
#         if out2:
#             frame_vid = cv2.resize(frame, (crop_width * 3, crop_height * 3), interpolation=cv2.INTER_CUBIC)
#             out2.write(frame_vid)
#
#         # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('Tracking', 900,900)
#         # cv2.imshow('Tracking', frame)
#         # cv2.waitKey()
#
#         frame_count = frame_count + 1
#
#     print("Done!")
#     # cap2.release()
#     if(out2):
#         out2.release()
#     cv2.destroyAllWindows()

def mark_ground_truth(worker, image_path, Beacon, data_type, scale, frame_amount, out_path, gt_video_path, gt = False):

    out2 = None
    vid = None
    frame_count = 0

    save_img = False
    # get_cells = True
    get_cells = False
    f_det_txt = None

    if(gt == True and os.path.exists(gt_video_path)):
        # print(out_path + "Beacon-" + str(Beacon) + "processed.avi")
        vid = cv2.VideoCapture(gt_video_path)
        if(vid):
            skip = frame_amount - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            print("mark cells skip: ", skip)

    print("mark cells:")
    image_amount_str = str(frame_amount)

    if (save_img):
        os.makedirs(out_path + "tmp/" + "Beacon-" + str(Beacon) + "/", exist_ok=True)

    if(get_cells):
        os.makedirs(out_path + "/ML/", exist_ok=True)
        os.makedirs(out_path + "/ML/images/", exist_ok=True)
        cells_path = out_path + "/ML/cells/Beacon_" + str(Beacon) + "/"
        os.makedirs(cells_path, exist_ok=True)
        f_det_txt = open(out_path + "/ML/det.txt", "w")

    for frame_count in range(frame_amount):
        ret, frame = read_frame(image_path, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)
        if(ret == False):
            print("done")
            return

        print("\r", frame_count, end="/" + image_amount_str, flush=True)

        if(len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        gt_frame = np.zeros(0)
        ret = False
        if(gt and vid and frame_count >= skip):
            ret, gt_frame = vid.read()

        # frame = worker.mark(frame, frame_count, scale)
        # ret_1, frame_1 = read_frame(image_path, Beacon, frame_count, data_type, 1)

        # print("gt and ret", gt, ret)


        frame, frame_red = worker.mark_gt(frame, frame_count, scale, gt_frame, crop_height, crop_width, out_path, Beacon, gt and ret, get_cells, f_det_txt)
        # ph_detector.mark_cells(frame, frame_count)

        size = (crop_width * 3, crop_height * 3)
        # size = (crop_width, crop_height)
        if(out2 is None):
            # out2 = cv2.VideoWriter(out_path + "cell_classified_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4",fourcc, 3.0, (crop_width * 3, crop_height * 3), isColor=True)
            out2 = cv2.VideoWriter(out_path + "Beacon-" + str(Beacon) + "processed.mp4",fourcc, 3.0, size, isColor=True)

        if out2:
            if(save_img):
                cv2.imwrite(out_path + "tmp/Beacon-" + str(Beacon) + "/mark_img_" + str(frame_count) + ".png", frame)

            frame_vid = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
            out2.write(frame_vid)

            if(gt and ret and save_img):
                frame_red = cv2.resize(frame_red, (crop_width * 8, crop_height * 8), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(out_path + "tmp/Beacon-" + str(Beacon) + "/imj_j" + str(frame_count) + ".png", frame_red)

    if(get_cells):
        f_det_txt.close()

    # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Tracking', 900,900)
    # cv2.imshow('Tracking', frame)
    # cv2.waitKey()

    print("\n")

    print("Done!")
    # cap2.release()
    if(out2):
        out2.release()
    cv2.destroyAllWindows()


# def preprocess(frame, save_image_path):
#
#     # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('preprocess_0', 900, 900)
#     # cv2.imshow('preprocess_0', frame)
#
#     if (len(frame.shape) > 2):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     else:
#         frame = frame
#
#     width = frame.shape[1]
#     height = frame.shape[0]
#     box_side = 64
#     vertical = int(height / box_side + 1)
#     horizontal = int(width / box_side + 1)
#
#     for i in range(vertical):
#         for j in range(horizontal):
#             start_x = box_side * j
#             start_y = box_side * i
#
#             end_x = 0
#             end_y = 0
#
#             if(i < vertical and j < horizontal):
#                 end_x = start_x + box_side
#                 end_y = start_y + box_side
#             elif(j ==  horizontal):
#                 end_x = 1328
#                 end_y = start_y + box_side
#             elif(i == vertical):
#                 end_x = start_x + box_side
#                 end_y = 1048
#
#
#             window = frame[start_y:end_y, start_x:end_x]
#
#             histSize = 256
#             histRange = (0, 256)  # the upper boundary is exclusive
#             accumulate = False
#             window_b_hist = cv2.calcHist([window], [0], None, [histSize], histRange, accumulate=accumulate)
#
#             window_hist_max_index = np.argmax(window_b_hist)
#
#             hist_max_index = window_hist_max_index
#             hist_trans = np.zeros(256, dtype=np.uint8)
#             target_peak = 100
#
#             for k in range(hist_max_index):
#                 hist_trans[k] = target_peak * k / hist_max_index + 0.5
#
#             for m in range(hist_max_index, 256):
#                 hist_trans[m] = ((255 - target_peak) * m - 255 * hist_max_index + target_peak * 255) / (255 - hist_max_index) + 0.5
#
#             window = hist_trans[window]
#             frame[start_y:end_y, start_x:end_x] = window
#
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))#clipLimit=2.0,
#     clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))#clipLimit=2.0,
#     # gray = clahe.apply(gray)
#     frame = clahe2.apply(frame)
#
#     cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Tracking', 900, 900)
#     cv2.imshow('Tracking', frame)
#     cv2.waitKey()
#
#     np.save(save_image_path, frame)
#     new_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     return new_frame

def preprocess_2(frame, save_image_path, scale = 1):
    # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('preprocess_0', 900, 900)
    # cv2.imshow('preprocess_0', frame)

    if (len(frame.shape) > 2):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_org = frame.copy()

    # t0 = time.time()
    frame = cv2.medianBlur(frame_org, 81)#There is an unexpected effect when ksize as 81, applied to 8 times scaled image.

    frame = frame / 100
    frame = frame_org / frame
    np.clip(frame, 0, 255, out = frame)
    frame = frame.astype(np.uint8)

    if(scale > 1):
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation = cv2.INTER_CUBIC)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


if __name__ == "__main__":
    # execute main
    # main(sys.argv[1], sys.argv[2])

    print(sys.argv)

    if(len(sys.argv[1:]) > 1):
        main(*sys.argv[1:])
    else:
        home_dir = os.path.expanduser("~") + "/"
        # main(configure_path="./configure.txt", path=home_dir + "Work/ground_truth/preprocess", out_path=home_dir + "Work/ground_truth/test")
        # main(configure_path = "./configure.txt", path = home_dir + "disk_16t/Pt196/RawData/Beacon-153", out_path = home_dir + "disk_16t/Pt196/output/Beacon-153")
        # main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt210/RawData/Beacon-73", out_path="Default")
        # main(configure_path="./configure.txt", path="/home/qibing/disk_16t/qibing/Pt298_SOCCO/RawData/Beacon-124", out_path="Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt174/RawData/Beacon-44", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt210/RawData/Beacon-73", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt180/RawData/Beacon-77", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt180/RawData/Beacon-32", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt204/RawData/Beacon-73/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt174/RawData/Beacon-1/", out_path = "Default") The beacon only has a couple of cells.
        pass

