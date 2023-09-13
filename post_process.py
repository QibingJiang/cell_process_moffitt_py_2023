#!/usr/bin/env python3
import cv2
import copy
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing
import re
import time

import multiprocessing as mp
import random
from hungarian_algorithm import algorithm
import pandas as pd
from scipy.signal import find_peaks
from io import StringIO

labels_0 = ["SSR", "R"]
labels_1 = ["∆Area", "T"]

col = 1
row = 2

path_0 = "/home/qibing/disk_16t/qibing/output_max_min_2_feat/"
path_1 = ""

pt_sel = ['Pt174', 'Pt180', 'Pt181', 'Pt196', 'Pt199', 'Pt210', 'Pt220', 'Pt224', 'Pt230', 'Pt298_SOCCO', 'Pt323_SOCCO', 'Pt400_SOCCO', 'Pt415_SOCCO']


def main():
    for pt in pt_sel:
        files = os.listdir(path_0 + pt + "/info_ucf/")
        files = [x for x in files if("file3_" in x)]
        for file in files:
            ti_c = os.path.getctime(path_0 + pt + "/info_ucf/" + file)
            c_ti = time.ctime(ti_c)
            if("Aug" in c_ti):
                print(path_0 + pt + "/info_ucf/" + file, c_ti)
                exit()
    exit()











    # path = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/info_ucf/file3_73_02_21_43.txt"
    path = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/info_ucf/file3_73_06_19_18.txt"

    with open(path, "r") as f:
        data = f.read()
    lines = data.split("\n")
    show_all_cells(lines)

def show_all_cells(lines):
    tracks = []
    image_amount = 0
    # f_die_time = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_die_time.txt", 'w')
    f_die_time = open("./die_time.txt", 'w')

    for i in range(len(lines)):
        cell = None
        death_t = np.nan
        if (lines[i].startswith("track: ")):
            cell = lines[i][7:-1]
            plt.clf()
            plt.figure(0, figsize=(6 * col, 4.5 * row))
            # print("cell: ", cell)
            plt.subplot(2, 1, 1)
            for j in range(3):
                i += 1
                l_list = lines[i].split(',')
                l_f = [float(e) for e in l_list[1:-1]]
                # if (j == 0):
                #     tmp = l_f.copy()
                #     print("cell_diff: ", tmp)
                #     pass
                plt.plot(l_f, label=l_list[0])
            i += 1
            death_t_0 = int(lines[i].split(',')[0])
            plt.title("cell: " + cell + " feature 1 cell diff, death: " + str(death_t_0))
            plt.plot(death_t_0, 0, '^')
            plt.xlim(0, 300)
            plt.legend()

            plt.subplot(2, 1, 2)
            for j in range(4):
                i += 1
                l_list = lines[i].split(',')
                l_f = [float(e) for e in l_list[1:-1]]

                if(j == 0):
                    area_tmp = l_f.copy()
                if (j > 0):
                    plt.plot(l_f, label=l_list[0])
                # if (j == 2):
                #     print("cell_area_diff_max_min: ", l_f)
                #     pass

            i += 1
            if(image_amount == 0):
                image_amount = len(l_f)

            death_t_1 = int(lines[i].split(',')[0])
            plt.title("cell: " + cell + " feature 2 cell area, death: " + str(death_t_1))
            plt.plot(death_t_1, 0, '^')
            plt.xlim(0, 300)
            plt.legend()
            plt.savefig("./cells/" + cell + ".pdf")
            # plt.show()

            death_t = min(death_t_0, death_t_1)
            track = np.zeros(len(l_f))
            track[:death_t] = 1
            track[death_t:] = 0
            # print(track)
            # nan_loc = np.argwhere(np.isnan(tmp))
            # track[nan_loc] = np.nan
            # print(track)
            track = track * area_tmp
            # print(area_tmp, track)
            tracks.append(track)
            # exit()

            if(death_t > image_amount - 5):
                f_die_time.write(cell + " " + str(1000) + "\n")
            elif(0 <= death_t <= image_amount - 5):
                f_die_time.write(cell + " " + str(death_t) + "\n")
            elif(death_t < 0 or np.isnan(death_t)):
                f_die_time.write(cell + " " + str(-1) + "\n")
            else:
                pass

    live_state = np.zeros((len(l_f), 4))

    for i in range(image_amount):
        for j in range(len(tracks)):
            if(not np.isnan(tracks[j][i])):
                if(tracks[j][i] > 0):
                    live_state[i][0] += 1
                    live_state[i][2] += tracks[j][i]
                else:
                    live_state[i][1] += 1
                    live_state[i][3] += 0 # I did not calculate the area sum of dead cells.

    np.savetxt("./live_state.txt", live_state)



if __name__ == "__main__":
    main()