#!/usr/bin/env python3
import cv2
import copy
from cell_detect import CellDetector
from cell_classify import CellClassifier
import os
# from matplotlib import pyplot as plt
import numpy as np
# from phagocytosis_detect import PhagocytosisDetector
import multiprocessing
import time
# import imageio
from itertools import chain
from operator import add
# from statistics import mean
import cell_process
from cell_process import *

home_dir = os.path.expanduser("~") + "/"

debug = 0
crop_width = 1328
crop_height = 1048
# crop_width = 512
# crop_height = 512
# crop_width = 256
# crop_height = 256
# crop_width = 128
# crop_height = 128

scale = 8
line_thick = 1
debug = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# parameters = [  ["Pt180", [54], "{0:0=1d}", "f00d1.PNG", "scan@1_Plate_D_p"],
#                 ["Pt204", [2], "{0:0=1d}", "f00d1.PNG", "scan_Plate_D_p"],
#                 ["Pt204", [93], "{0:0=1d}", "f00d1.PNG", "scan_Plate_D_p"],
#                 ["Pt210", [73], "{0:0=1d}", "f00d0.TIF", "scan_Plate_D_p"],
#                 ["Pt211", [41], "{0:0=2d}", "f00d0.PNG", "scan_Plate_D_p"]]

# drug_abbr = {
# "BTZ": "Bortezomib"
# "CFZ": "Carfilzomib"
# "DEX": "DEXAMETHASONE"
# "IXA": "IXAZOMIB"
# "LEN": "LENALIDOMIDE"
# "PAN": "PANOBINOSTAT"
# "POM": "POMALIDOMIDE"
# "DAR": "DARATUMUMAB"
# "ADR": "ADRIAMYCIN"
# }

# drug_abbr_dict = {
# "BORTEZOMIB": "BTZ",
# "CARFILZOMIB": "CFZ",
# "DEXAMETHASONE": "DEX",
# "IXAZOMIB": "IXA",
# "LENALIDOMIDE": "LEN",
# "PANOBINOSTAT": "PAN",
# "POMALIDOMIDE": "POM",
# "DARATUMUMAB": "DAR",
# "ADRIAMYCIN": "ADR",
# }

drug_names = [['111'], ['1112', '111_2'], ['113'], ['1132', '113_2'], ['A-1155463'], ['A-1210477'], ['ABBVIE1'], ['ABBVIE10'], ['ABBVIE11'], ['ABBVIE12'], ['ABBVIE13'], ['ABBVIE14'], ['ABBVIE15'], ['ABBVIE16'], ['ABBVIE2'], ['ABBVIE3'], ['ABBVIE4'], ['ABBVIE5'], ['ABBVIE6'], ['ABBVIE7'], ['ABBVIE8'], ['ABBVIE9'], ['ABT-199', 'VEN'], ['ADAVOSERTIB'], ['ALISERTIB', 'Ali', 'ali', 'alisertib', 'Alisertib'], ['AR-A014418'], ['AR-A014418_1'], ['ASO'], ['ASO_CONTROL', 'ASO CONTROL'], ['ASO_CONTROL1', 'ASO CONTROL1'], ['ASO_CONTROL2', 'ASO CONTROL2'], ['ASO_IRF4_1', 'ASO IRF4(1)'], ['ASO_IRF4_2', 'ASO IRF4(2)'], ['ASO_IRF4x2', 'ASO IRF4x2'], ['ASO_PYK2', 'ASO PYK2', 'PYK2 ASO'], ['ASO_STAT3', 'ASO STAT3', 'STAT3 ASO'], ['AZ-628'], ['AZD1208', 'ASD1208'], ['AZD1480'], ['AZD7762'], ['BARASERTIB'], ['BARASERTIB_1'], ['BAY1816032'], ['BAY1816032_1'], ['BI2536', 'BI2536 ', 'Bl2536', 'Bl2536 '], ['BMS-265246'], ['BMS-265246(high)'], ['BMS-265246(low)'], ['BMS-265246_1'], ['BMS754807', 'BMS-754807', 'BMS754807 '], ['BMS777607'], ['BORTEZOMIB_BAD'], ['BTZ', 'BORTEZOMIB', 'Bortezomib'], ['BTZ(24h)', '24h+BTZ'], ['BTZ1', 'BORTEZOMIB1'], ['BTZ2', 'BORTEZOMIB2'], ['BTZ3', 'BORTEZOMIB3'], ['BTZ4', 'BORTEZOMIB4'], ['BTZ5', 'BORTEZOMIB5'], ['BTZ6', 'BORTEZOMIB6'], ['BTZ7', 'BORTEZOMIB7'], ['BTZ8', 'BORTEZOMIB8'], ['BTZ_1'], ['CB2'], ['CEP-33779'], ['CEP-33779_1'], ['CFZ', 'CARFILZOMIB', 'Carfilzomib', 'CRAFILZOMIB'], ['CFZ1', 'CARFILZOMIB1'], ['CFZ2', 'CARFILZOMIB2'], ['CFZ3', 'CARFILZOMIB3'], ['CFZ4', 'CARFILZOMIB4'], ['CFZ5', 'CARFILZOMIB5'], ['CFZ6', 'CARFILZOMIB6'], ['CFZ7', 'CARFILZOMIB7'], ['CFZ70', 'CARFILZOMIB70'], ['CFZ8', 'CARFILZOMIB8'], ['CFZ_1'], ['CFZ_2'], ['CFZ_3'], ['CFZ_4'], ['CFZ_5'], ['CFZ_6'], ['CFZ_7'], ['CFZ_8'], ['CGP-60474'], ['CISPLATINUM', 'CIS'], ['CLE'], ['COBIMETINIB', 'COB'], ['COLCHICINE'], ['CONTROL'], ['CONTROL1'], ['CONTROL2'], ['CONTROL3'], ['CONTROL4'], ['CONTROL5'], ['CONTROL6'], ['CONTROL7'], ['CONTROL_1'], ['CONTROL_10'], ['CONTROL_11'], ['CONTROL_12'], ['CONTROL_13'], ['CONTROL_14'], ['CONTROL_15'], ['CONTROL_16'], ['CONTROL_17'], ['CONTROL_18'], ['CONTROL_19'], ['CONTROL_2'], ['CONTROL_20'], ['CONTROL_21'], ['CONTROL_22'], ['CONTROL_23'], ['CONTROL_24'], ['CONTROL_3'], ['CONTROL_4'], ['CONTROL_5'], ['CONTROL_6'], ['CONTROL_7'], ['CONTROL_8'], ['CONTROL_9'], ['CP-724714'], ['CP43'], ['CP43_1'], ['CPD22', 'Cpd22'], ['CPD22_1'], ['CRIZOTINIB', 'crizotinib', 'Crizotinib', 'rizotinib'], ['CURCUMIN'], ['CYCLOPHOSPHAMIDE', 'CYC'], ['DABRAFENIB', 'Da', 'da', 'Dabrafeni', 'Dabrafenib', 'dabrafenib'], ['DARATUMUMAB', 'DAR'], ['DARATUMUMAB-80C'], ['DARATUMUMAB-80F'], ['DASATINIB', 'Dasatanib', 'dasatinib', 'Dasatinib'], ['DEFACTINIB', 'DEF', 'Defa', 'Defactinib', 'DEFACTINIB(VS6063)', 'Defactinib(VS-6063)', 'Defactinib (VS-6063)'], ['DEFACTINIB1'], ['DEFACTINIB2'], ['DEFACTINIB3'], ['DEX', 'Dexametha', 'Dexamethasone', 'DEXAMETHASONE'], ['DEX2'], ['DINACICLIB', 'Dinaciclib'], ['DINACICLIB_1'], ['DMSO'], ['DMSO2'], ['DORSOMORPHIN'], ['DORSOMORPHIN_1'], ['DOVITINIB'], ['DOX', 'ADR', 'Adriamycin', 'Adriamycin (Dox)', 'Adriamy', 'ADRIAMYCIN', 'Doxorubi', 'Doxorubicin', 'DOXORUBICIN'], ['DOX1', 'ADR1', 'ADRIAMYCIN1'], ['DOX2', 'ADR2', 'ADRIAMYCIN2'], ['DOX3', 'ADR3', 'ADRIAMYCIN3'], ['DS'], ['DUMMY1'], ['DUMMY10'], ['DUMMY11'], ['DUMMY12'], ['DUMMY13'], ['DUMMY14'], ['DUMMY15'], ['DUMMY16'], ['DUMMY2'], ['DUMMY3'], ['DUMMY4'], ['DUMMY5'], ['DUMMY6'], ['DUMMY7'], ['DUMMY8'], ['DUMMY9'], ['ELEVENOSTAT'], ['ELOTUZUMAB'], ['EN460'], ['ENZASTAURIN'], ['ENZASTAURIN_1'], ['ERLOTINIB', 'Erlotinib'], ['ETOPOSIDE'], ['F8'], ['FOR'], ['FOR10'], ['FOR2'], ['FOR28'], ['FOR29'], ['FRAX597'], ['FRAX597_1'], ['FXM'], ['GDC-0980'], ['GGTI', 'GGT'], ['GGTI(SIM)'], ['GGTI2'], ['GSK461364'], ['GSK461364_1'], ['HDAC11 ASO'], ['HP9060'], ['HSP90'], ['I-BET-762'], ['IBRUTINIB', 'Ibrutinib', 'ibrutinib'], ['IDELALISIB', 'idealisib', 'Idelali', 'idelali', 'idelalisib', 'Idelalsib', 'Idelalisib', 'IDEALISIB'], ['INCB054329'], ['INK128', 'INK 128'], ['INK128_1'], ['IRAK1-4(INHIBITOR)407601'], ['IRAK1-4(INHIBITOR)PF06650833'], ['IRF4 ASO-1'], ['IRF4 ASO-2'], ['ISOTYPE'], ['IXAZOMIB', 'IXAZUMAB', 'IXAZUMIB', 'IXA'], ['JNK-IN-8', 'JNK-IN-8 (specific)'], ['JNK-IN-8_1'], ['JQ1'], ['KPT', 'KPT330', 'KPT-330', 'KPT330 (Selinexor)'], ['KPT-DEX-BTZ'], ['KPT-DEX-DOX'], ['KPT-DEX-ELOTUZUMAB'], ['KPT1', 'KPT(1)'], ['KPT2'], ['LCL161'], ['LEN', 'LENALIDOMIDE', 'Lenalidomide', 'REVLIMID'], ['LINIFANIB', 'Linifanib', 'linifanib'], ['LJI308'], ['LJI308_1'], ['LOSMAPIMOD'], ['LOSMAPIMOD_1'], ['LY2584702'], ['LY2584702_1'], ['LY2603618'], ['MA7-038'], ['MARK-INHIBITOR', 'MARK_INHIBITOR', 'MARK INHIBITOR'], ['MARK3'], ['MARK3_1'], ['ME-POMALIDOMIDE'], ['MEL', 'Mel', 'MELPHALAN', 'Melphalan'], ['MEL1', 'MELPHALAN1'], ['MEL2', 'MELPHALAN2'], ['MEL3', 'MELPHALAN3'], ['MEL4', 'MELPHALAN4'], ['MEL5', 'MELPHALAN5'], ['MEL6', 'MELPHALAN6'], ['MEL7', 'MELPHALAN7'], ['MEL8', 'MELPHALAN8'], ['MK2206', 'MK-2206'], ['MK2206_1'], ['MOMELOTINIB', 'Momelotinib', 'momelotinib'], ['MOTESANIB', 'Mote', 'mote', 'motesanib', 'Motesanib'], ['MTI101', 'MT101', 'MTI-101'], ['MTI1012', 'MTI-1012'], ['MTX'], ['MYC'], ['NICLOSAMIDE'], ['NU-7441'], ['NU-7441_1'], ['NVP2'], ['ONX'], ['OPR', 'OPROZ', 'Oprozomib'], ['OPR2', 'OPROZ2'], ['OTSSP167'], ['PALBOCICLIB', 'PABLOCICLIB', 'Palbo', 'palbo', 'palbociclib', 'Palbociclib'], ['PANOBINOSTAT', 'PAN', 'PANO', 'Panobino', 'Panobinostat'], ['PANOBINOSTAT1'], ['PANOBINOSTAT2', 'PAN2'], ['PANOBINOSTAT3'], ['PDI'], ['POM', 'POMALIDOMIDE', 'Pomalidomide'], ['POM2'], ['PONATINIB', 'Ponatinib', 'ponatinib'], ['PREXASERTIB', 'PRX'], ['PYRVINIUM'], ['QST', 'Qui'], ['QST2'], ['QUISINOSTAT', 'Quisinostat'], ['R406'], ['R406_1'], ['RABUSERTIB'], ['RABUSERTIB_1'], ['RALIMETINIB', 'Ralimetinib', 'ralimetinib'], ['RICOLINOSTAT', 'ACY-1215', 'ROCILINOSTAT'], ['RUXOLITINIB', 'Ruxolitinib', 'ruxolitinib'], ['S63845'], ['SARACATINIB'], ['SARACATINIB_1'], ['SCH772984'], ['SCH772984_1'], ['SELUMETINIB', 'elumetinib', 'selumetinib', 'Selumetinib'], ['SILMITASERTIB'], ['SILMITASERTIB_1'], ['SNS-032'], ['SORAFENIB'], ['SORAFENIB_1'], ['SR3029', 'SR-3029'], ['SR30292'], ['SR3029_2'], ['SR4835'], ['SR5037'], ['TAI-1'], ['TAI-1_1'], ['TGR-1202'], ['THZ1'], ['TOZASERTIB', 'Toza', 'toza', 'tozasertib', 'Tozasertib'], ['TRAMETINIB', 'Trametinib', 'trametinib'], ['UMI-77'], ['VANDETANIB', 'Vandetanib'], ['VE-822'], ['VE-822_1'], ['VEMURAFENIB', 'emurafenib', 'vemurafenib', 'Vemurafenib'], ['VINCRISTINE'], ['VOLASERTIB', 'Volasertib', 'VOL'], ['VOLASERTIB_1'], ['VOLASERTIB_2'], ['VOLASERTIB_3'], ['VOLASERTIB_4'], ['VOLASERTIB_5'], ['VOLASERTIB_6'], ['VOLASERTIB_7'], ['VOLASERTIB_8'], ['VS4718', 'VS-4718'], ['VX745']]


# pt_s_dir = [[home_dir + "disk_m2/", ["Pt280_SOCCO", "Pt281_SOCCO", "Pt282_SOCCO", "Pt283_SOCCO", "Pt285_SOCCO", "Pt290_SOCCO", "Pt291_SOCCO", "Pt292_SOCCO", "Pt293_SOCCO", "Pt294_SOCCO", "Pt297_SOCCO", "Pt298_SOCCO", "Pt299_SOCCO", "Pt300_SOCCO", "Pt301_SOCCO", "Pt303_SOCCO", "Pt304_SOCCO", "Pt306_SOCCO_SPORE", "Pt307_SOCCO_SPORE", "Pt315_SOCCO_SPORE", "Pt421_SOCCO", "Pt422_SOCCO", "Pt423_SOCCO", "Pt426_SOCCO", "Pt428_SOCCO"]],
# [home_dir + "disk_m1/", ["Pt315_SOCCO_SPORE", "Pt323_SOCCO", "Pt325_SOCCO", "Pt348_SOCCO", "Pt373_SOCCO", "Pt375_SOCCO", "Pt380_SOCCO", "Pt382_SOCCO", "Pt386_SOCCO", "Pt387_SOCCO", "Pt388_SOCCO", "Pt389_SOCCO", "Pt390_SOCCO", "Pt392_SOCCO", "Pt393_SOCCO", "Pt394_SOCCO", "Pt395_SOCCO", "Pt397_SOCCO", "Pt398_SOCCO", "Pt400_SOCCO", "Pt401_SOCCO", "Pt403_SOCCO", "Pt409_SOCCO", "Pt415_SOCCO", "Pt419_SOCCO"]],
# [home_dir + "disk_16t/qibing/", ["Pt170", "Pt171", "Pt174", "Pt176", "Pt177", "Pt178", "Pt180", "Pt181", "Pt182", "Pt184", "Pt186", "Pt187", "Pt193", "Pt196", "Pt198", "Pt199", "Pt200", "Pt201", "Pt203", "Pt204", "Pt205", "Pt206", "Pt207", "Pt210", "Pt211", "Pt212", "Pt213", "Pt219", "Pt220", "Pt222", "Pt223", "Pt224", "Pt226", "Pt227", "Pt229", "Pt230", "Pt235", "Pt236", "Pt238", "Pt242"]]]
# pt_sel = ['Pt174', 'Pt180', 'Pt181', 'Pt196', 'Pt199', 'Pt210', 'Pt220', 'Pt224', 'Pt230', 'Pt298_SOCCO', 'Pt323_SOCCO', 'Pt400_SOCCO', 'Pt415_SOCCO']

pt_sel = ['Pt174', 'Pt180', 'Pt181', 'Pt196', 'Pt210', 'Pt224', 'Pt230', 'Pt298_SOCCO', 'Pt400_SOCCO', 'Pt415_SOCCO']
drug_sel = ['BORTEZOMIB','IXAZOMIB','PANOBINOSTAT','CARFILZOMIB','DEXAMETHASONE','POMALIDOMIDE','LENALIDOMIDE','DARATUMUMAB']

pt_sel = ['Pt174', 'Pt180']
drug_sel = ['BORTEZOMIB', 'DEXAMETHASONE']


pt_s_dir = [[home_dir + "disk_16t/qibing/", pt_sel]]


sample_index = 3
def main():
    pt_dict = {pt:disk[0] for disk in pt_s_dir for pt in disk[1]}
    output_path = home_dir + "disk_16t/qibing/cct_v2_test/"
    os.makedirs(output_path, exist_ok=True)
    log_f = open(output_path + "mylog_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")
    # files = os.listdir(path)
    # pt_s = [x for x in files if ("Pt" in x) and len(x) == 11]  # and len(x) == 5

    configure_path = "./configure.txt"

    processes = []
    # for pt in ["Pt180", "Pt204", "Pt210", "Pt211"]:#Pt170, "Pt238 has abnormal result of image j"
    # for pt in ["Pt171", "Pt181", "Pt242"]:
    for pt in pt_sel:
        path = pt_dict[pt]
    # for drug in ["CONTROL", "BORTEZOMIB", "CARFILZOMIB", "DEXAMETHASONE", "DMSO", "IXAZOMIB", "LENALIDOMIDE", "PANOBINOSTAT", "POMALIDOMIDE", "SR3029"]:
    #     for drug in ["CONTROL", "BORTEZOMIB", "CARFILZOMIB", "DEXAMETHASONE", "IXAZOMIB", "LENALIDOMIDE", "PANOBINOSTAT", "POMALIDOMIDE", "ABT-199", "DARATUMUMAB", "INK128", "ADR", "KPT", "MEL", "VEN"]: #DMSO SR3029
    #     for drug in ["CONTROL", "BORTEZOMIB", "CARFILZOMIB"]:
        for drug in ["CONTROL"] + drug_sel:
            print(pt, drug, file=log_f)
            match_list = []
            # actually this is duplicate in the long match list, but this is useful when add new drug,which is not in the drug_name list.
            match_list.append(drug)
            match_list.append(drug + "_")
            match_list.append(drug + "_2")

            for d1 in range(len(drug_names)):
                if drug in drug_names[d1]:
                    for d2 in range(len(drug_names[d1])):
                        match_list.append(drug_names[d1][d2])
                        match_list.append(drug_names[d1][d2] + "_")
                        match_list.append(drug_names[d1][d2] + "_2")
                    break

            # print(match_list)

            with open(path + pt + "/GraphPadFiles/PtSample/ExperimentalDesign.txt", "r") as f:
                lines = f.readlines()

            g2 = []
            for l in lines:
                m = re.search(';(.+?);', l)
                m_1 = m.group(1)

                # if (drug + "_" == m_1 or drug + "_2" == m_1 or drug == m_1):# the drug == m_1 is for control
                # abbr = drug_abbr_dict.get(drug)
                # if (m_1 in [drug + "_", drug + "_2", drug] or (abbr and m_1 in [abbr, abbr + "_", abbr + "_2"])):  # the drug == m_1 is for control

                if (m_1 in match_list):
                    bea_str = l[0:m.start()]
                    bea_str = re.sub("Beacon-", "", bea_str)
                    bea_1 = bea_str.split(",")
                    g2.append(bea_1)

            if(len(g2) == 0):
                print(pt, " no drug ", drug, file=log_f)
                continue

            bea_s = g2[0] + g2[1]
            # if(drug == "CONTROL"):
            #     bea_s = g2[0] + g2[1]
            # else:
            #     bea_s = [g2[0][0]] + [g2[1][0]]

            bea_arr = np.array(bea_s)
            bea_arr = bea_arr.astype(np.int64)
            bea_arr = np.unique(bea_arr)
            print(bea_arr,  file=log_f)
            loop_cnt = 0

            for beacon in bea_arr:
                input_path = path + pt + "/RawData/Beacon-" + str(beacon) + "/"
                # cell_process.main(configure_path, input_path)
                # exit()

                try:
                    p = multiprocessing.Process(target=cell_process.main, args=(configure_path, input_path, output_path + pt))
                    p.start()
                    processes.append(p)
                    print(time.strftime("%d_%H_%M ", time.localtime()), p, input_path, file = log_f)
                    log_f.flush()

                except Exception as e:  # work on python 3.x
                    print('Exception: ' + str(e))

                loop_cnt = loop_cnt + 1

                while len(processes) == 60:
                    # print(time.strftime("%d_%H_%M ", time.localtime()), len(processes), " processes are running.", file=log_f)
                    for p in processes:
                        if(p.is_alive() == False):
                            print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
                            log_f.flush()
                            p.terminate()
                            processes.remove(p)
                            break
                        else:
                            # print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
                            pass
                    time.sleep(60)

    while len(processes) > 0:
        # print(time.strftime("%d_%H_%M ", time.localtime()), len(processes), " processes are running.", file=log_f)
        for p in processes:
            if(p.is_alive() == False):
                print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
                log_f.flush()
                p.terminate()
                processes.remove(p)
                break
            else:
                # print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
                pass
        time.sleep(60)

    log_f.close()
    print("All processes ended.")

def main_label():

    path = "/home/qibing/disk_t/"
    configure_path = "./configure_make_ground_truth.txt"

    log_f = open(path + "mylog.txt", "w")

    processes = []
    for pt in ["Pt170", "Pt171", "Pt180", "Pt181", "Pt204", "Pt210", "Pt211", "Pt242"]:
        for drug in ["BORTEZOMIB"]:

            print(pt, drug, file=log_f)

            with open("/home/qibing/disk_t/" + pt + "/GraphPadFiles/PtSample/ExperimentalDesign.txt", "r") as f:
                lines = f.readlines()

            g2 = []
            for l in lines:
                m = re.search(';(.+?);', l)
                m_1 = m.group(1)

                if (drug + "_" == m_1 or drug + "_2" == m_1 or drug == m_1):# the drug == m_1 is for control
                    bea_str = l[0:m.start()]
                    bea_str = re.sub("Beacon-", "", bea_str)
                    bea_1 = bea_str.split(",")
                    g2.append([bea_1[0]])

            bea_s = g2[0] + g2[1]
            bea_arr = np.array(bea_s)
            bea_arr = bea_arr.astype(np.int64)
            print(bea_arr)
            loop_cnt = 0

            for beacon in bea_arr:
                input_path = path + pt + "/RawData/Beacon-" + str(beacon) + "/"
                # cell_process.main(configure_path, input_path)
                # exit()

                try:
                    p = multiprocessing.Process(target=cell_process.main, args=(configure_path, input_path, path + "jiao/" + pt + "/"))
                    p.start()
                    processes.append(p)
                    print(time.strftime("%d_%H_%M ", time.localtime()), p, input_path, file = log_f)
                    log_f.flush()

                except Exception as e:  # work on python 3.x
                    print('Exception: ' + str(e))

                loop_cnt = loop_cnt + 1

                while len(processes) == 15:
                    # print(time.strftime("%d_%H_%M ", time.localtime()), len(processes), " processes are running.", file=log_f)
                    for p in processes:
                        if(p.is_alive() == False):
                            print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
                            log_f.flush()
                            p.terminate()
                            processes.remove(p)
                            break
                        else:
                            # print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
                            pass
                    time.sleep(60)

    log_f.close()
    print("All processes ended.")

# def main_2():
#
#     # beacons = chain(range(1, 6, 1), range(11, 16, 1), range(73, 78, 1), range(83, 88, 1), range(54, 59, 1), range(64, 69, 1))
#
#     beacons = chain(range(1, 6, 1), range(11, 16, 1),
#                     range(73, 78, 1), range(83, 88, 1), # BORTEZOMIB
#                     range(54, 59, 1), range(64, 69, 1), #
#                     range(150, 155, 1), range(160, 165, 1), # LENALIDOMIDE
#                     range(30, 35, 1), range(40, 45, 1),
#                     range(78, 83, 1), range(88, 93, 1), # PANOBINOSTAT
#                     range(126, 131, 1), range(136, 141, 1),
#                     range(121, 126, 1), range(131, 136, 1), # CARFILZOMIB
#                     range(169, 174, 1), range(179, 184, 1),
#                     range(25, 30, 1), range(35, 40, 1), # MELPHALAN
#                     range(102, 107, 1), range(112, 117, 1)) # POMALIDOMIDE
#
#
#     beacons = chain(range(1, 6, 1), range(21, 22, 1), range(11, 16, 1), range(23, 24, 1),
#                     range(73, 78, 1), range(93, 94, 1), range(83, 88, 1), range(95, 96, 1),
#                     range(54, 59, 1), range(70, 71, 1), range(64, 69, 1), range(72, 73, 1),
#                     )
#
#     beacons = chain(range(1, 6, 1), range(11, 16, 1),
#                     range(73, 78, 1), range(83, 88, 1), # BORTEZOMIB
#                     range(150, 155, 1), range(160, 165, 1), # LENALIDOMIDE
#                     range(78, 83, 1), range(88, 93, 1), # PANOBINOSTAT
#                     range(121, 126, 1), range(131, 136, 1), # CARFILZOMIB
#                     range(25, 30, 1), range(35, 40, 1), # MELPHALAN
#                     range(102, 107, 1), range(112, 117, 1)) # POMALIDOMIDE
#
#     beacons = list(beacons)
#
#     path = "/home/qibing/disk_t/"
#     configure_path = "./configure.txt"
#
#     for pt in ["Pt204", "Pt210"]:
#         processes = []
#         loop_cnt = 0
#         for beacon in beacons:
#             input_path = path + pt + "/RawData/Beacon-" + str(beacon) + "/"
#             # cell_process.main(configure_path, input_path)
#             # exit()
#
#             try:
#                 p = multiprocessing.Process(target=cell_process.main, args=(configure_path, input_path))
#                 p.start()
#                 processes.append(p)
#
#             except Exception as e:  # work on python 3.x
#                 print('Exception: ' + str(e))
#
#             loop_cnt = loop_cnt + 1
#
#             # if(loop_cnt == 20 or beacon == beacons[-1]):
#             #     print(len(processes), " processes are running.")
#             #     for p in processes:
#             #         print("join: ", p)
#             #         p.join()
#             #
#             #     loop_cnt = 0
#             #     processes = []
#
#             while len(processes) == 15:
#                 print(len(processes), " processes are running.")
#                 for p in processes:
#                     if(p.is_alive() == False):
#                         p.close()
#                         processes.remove(p)
#                         break
#                     else:
#                         pass
#                 time.sleep(60)
#
#     print("All processes ended.")



# def main_1():
#
#     # beacons = chain(range(1, 6, 1), range(11, 16, 1), range(73, 78, 1), range(83, 88, 1), range(54, 59, 1), range(64, 69, 1))
#
#     beacons = chain(range(1, 6, 1), range(11, 16, 1),
#                     range(73, 78, 1), range(83, 88, 1), # BORTEZOMIB
#                     range(54, 59, 1), range(64, 69, 1), #
#                     range(150, 155, 1), range(160, 165, 1), # LENALIDOMIDE
#                     range(30, 35, 1), range(40, 45, 1),
#                     range(78, 83, 1), range(88, 93, 1), # PANOBINOSTAT
#                     range(126, 131, 1), range(136, 141, 1),
#                     range(121, 126, 1), range(131, 136, 1), # CARFILZOMIB
#                     range(169, 174, 1), range(179, 184, 1),
#                     range(25, 30, 1), range(35, 40, 1), # MELPHALAN
#                     range(102, 107, 1), range(112, 117, 1)) # POMALIDOMIDE
#
#
#     beacons = chain(range(1, 6, 1), range(21, 22, 1), range(11, 16, 1), range(23, 24, 1),
#                     range(73, 78, 1), range(93, 94, 1), range(83, 88, 1), range(95, 96, 1),
#                     range(54, 59, 1), range(70, 71, 1), range(64, 69, 1), range(72, 73, 1),
#                     )
#
#     beacons = chain(range(1, 6, 1), range(11, 16, 1),
#                     range(73, 78, 1), range(83, 88, 1), # BORTEZOMIB
#                     range(150, 155, 1), range(160, 165, 1), # LENALIDOMIDE
#                     range(78, 83, 1), range(88, 93, 1), # PANOBINOSTAT
#                     range(121, 126, 1), range(131, 136, 1), # CARFILZOMIB
#                     range(25, 30, 1), range(35, 40, 1), # MELPHALAN
#                     range(102, 107, 1), range(112, 117, 1)) # POMALIDOMIDE
#
#
#     beacons = list(beacons)
#
#     # beacons = np.array([
#     #     1, 21, 11, 23,
#     #     73, 93, 83, 95,
#     #     54, 70, 64, 72,
#     #     150, 166, 160, 168,
#     #     30, 46, 40, 48,
#     #     78, 94, 88, 96,
#     # 126, 142, 136, 144,
#     # 121, 141, 131, 143,
#     # 169, 189, 179, 191,
#     # 25, 45, 35, 47,
#     # 102, 118, 112, 120])
#     #
#
#
#     beacons = parameters[sample_index][1]
#     #
#     path = "/home/qibing/disk_t/"
#
#     for pt in ["Pt204", "Pt210"]:#"Pt211", "Pt210"
#         processes = []
#         loop_cnt = 0
#         for beacon in beacons:
#             # input_path = path + pt + "/RawData/Beacon-" + str(beacon) + "/"
#             # process_one_video_main(path + pt + "/", beacon, 1)
#             try:
#                 p = multiprocessing.Process(target=process_one_video_main, args=(path + pt + "/", beacon, 1, pt))
#                 p.start()
#                 processes.append(p)
#
#             except Exception as e:  # work on python 3.x
#                 print('Exception: ' + str(e))
#
#             loop_cnt = loop_cnt + 1
#
#             if(loop_cnt == 20 or beacon == beacons[-1]):
#                 print(len(processes), " processes are running.")
#                 for p in processes:
#                     print("join: ", p)
#                     p.join()
#
#                 loop_cnt = 0
#                 processes = []
#
#         print("All processes ended.")

# def process_one_video_main(path, Beacon, data_type, pt):
#
#     t0 = time.time()
#     t0_str = time.ctime(t0)
#
#     image_path = path + "RawData/Beacon-" + str(Beacon) + "/"
#
#     # if(os.path.exists(path + "Thumbs.db")):
#     #     os.remove(path + "Thumbs.db")
#     #     print(path + "Thumbs.db has been removed.")
#     # image_num = len(os.listdir(path))
#
#     detector = CellDetector()
#     classifier = CellClassifier(8, 20, 5, 0)
#     # ph_detector = PhagocytosisDetector(10, 30, 5, 0)
#
#     develop = 1
#     if(develop == 0):
#         detector.prepro_frames(image_path, pt, Beacon, parameters[sample_index])
#     else:
#         detector.background_pixel = 100
#         detector.background_pixel_mean = 99.57
#         detector.background_pixel_std = 18.02
#         detector.bg_gau_mean = 100.24
#         detector.bg_gau_std = 4.67
#         detector.cell_core_r = 2.88
#         detector.cell_core_r_std = 0.46
#         detector.noise_radius_thresh = 1.0
#
#     frame_count = 0
#
#     det_out = None
#     tra_out = None
#     # make_video = True
#     make_video = False
#
#     file = open(image_path + "detect_result_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")
#
#     frame_prev = None
#     while True:
#         print(str(Beacon) + "_" + str(frame_count), end = " ", flush=True)
#         ret, frame_org = read_frame(image_path, Beacon, frame_count, data_type, scale)
#
#         if(ret == False):
#             print("done")
#             break
#
#         frame_det = frame_org.copy()
#
#         # centers = detector.detect_by_white_core(frame_det, scale, frame_count)
#         # centers = detector.detect_hybrid(frame, scale)
#         # centers = detector.detect_by_contour_ex(frame_det, scale)
#         temp_t = time.time()
#         # frame_det, centers = detector.detect_edge_test(frame_det, frame_count, scale)
#         frame_det, centers = detector.detect_by_white_core_and_level(image_path, frame_det, frame_count, scale, pt, Beacon)
#         # frame_det, centers = detector.detect_by_edge_core_and_level(image_path, frame_det, frame_count, scale, pt, Beacon)
#
#         print("det t:", time.time() - temp_t)
#
#         cell_count = 0
#         if len(centers) > 0:
#             # centers_cat = np.vstack(centers)
#             frame_tra = frame_org.copy()
#             # frame_tra = classifier.match_track(centers_cat, frame_prev, frame_tra, frame_count, scale)
#             temp_t = time.time()
#             frame_tra = classifier.match_track_3_times(centers, frame_prev, frame_tra, frame_count, scale)
#             print("match t:", time.time() - temp_t)
#
#             # ph_detector.match_track(centers, frame, frame_count)
#
#             for i in range(len(centers)):
#                 # print(len(arr), arr[:, 3].sum(), end=" ")
#                 cell_count = cell_count + len(centers[i])
#                 # file.write(str(len(centers[i])) + " " + str(centers[i][:, 3].sum()) + " ")
#
#                 # if(centers_cat == None):
#                 #     centers_cat = centers[i]
#                 # else:
#                 #     centers_cat = np.concatenate((centers_cat, centers[i]), axis=0)
#
#                 # for point in centers[i]:
#                     # cv2.circle(frame_org, (int(point[0]), int(point[1])), 6, colors[i], 1)
#
#             # print(centers_cat)
#
#             print("\n")
#
#             pass
#         else:
#             print("not detected")
#
#         # cv2.putText(frame, str(frame_count) + " " + str(cell_count), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))
#         # cv2.putText(frame_org, str(frame_count) + " " + str(cell_count), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), int(0.3))
#         # cv2.putText(frame, str(len(centers)), (30*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))
#
#         if make_video == True:
#             if det_out is None:
#                 det_out = cv2.VideoWriter(image_path + "cell_detect_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_det.shape[1], frame_det.shape[0]), isColor=True)
#             if tra_out is None:
#                 tra_out = cv2.VideoWriter(image_path + "cell_track_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)
#
#         if (det_out != None):
#             det_out.write(frame_det)
#         if (tra_out != None):
#             tra_out.write(frame_tra)
#
#         # cv2.namedWindow('det', cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('det', 900, 900)
#         # cv2.imshow('det', frame_det)
#         # cv2.imwrite(image_path + "det_" + str(frame_count) + ".png", frame_det[154:625, 748:1340, :])
#         cv2.imwrite(image_path + "det_" + str(frame_count) + ".png", frame_det)
#         # cv2.waitKey()
#
#         # cv2.namedWindow('tra', cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('tra', 900, 900)
#         # cv2.imshow('tra', frame_tra)
#         # cv2.imwrite(image_path + "tra_" + str(frame_count) + ".png", frame_tra[154:625, 748:1340, :])
#         cv2.imwrite(image_path + "tra_" + str(frame_count) + ".png", frame_tra)
#         # cv2.waitKey()
#
#         frame_prev = frame_org.copy()
#         frame_count = frame_count + 1
#     print()
#
#
#     print("Done!")
#
#     if (det_out != None):
#         det_out.release()
#
#     if (tra_out != None):
#         tra_out.release()
#
#     classifier.background_pixel = detector.background_pixel
#     classifier.cell_core_r = detector.cell_core_r
#     file.write(str(detector.background_pixel))
#     file.write(" ")
#     file.write(str(detector.cell_core_r))
#     file.close()
#
#     # classifier.analyse_classification_3(image_path, frame_count)
#     gt_video_path = path + "TimeLapseVideos/Beacon-" + str(Beacon) + "processed.avi"
#     classifier.analyse_classification_5(image_path, frame_count, gt_video_path, scale)
#     mark_ground_truth(classifier, path, Beacon, data_type, 8, frame_count)
#
#     # mark(classifier, path, Beacon, data_type, scale)
#
#     cv2.destroyAllWindows()
#
#     t1 = time.time()
#     t1_str = time.ctime(t1)
#
#     if (not os.path.exists(image_path)):
#         os.makedirs(image_path)
#     log = t0_str + "\n" + t1_str + "\n" + str((t1 - t0)) + " seconds.\n"
#     log_file = open(image_path + "/" + t0_str + "_log.txt", 'w')
#     log_file.write(log)
#     print(log)
#     log_file.close()
#     # np.savetxt("/home/qibing/disk_t/" + pt + "/log", log)
#
#
# def mark(worker, path, Beacon, data_type, scale):
#
#     out2 = None
#     frame_count = 0
#
#     image_path = path + "/RawData/Beacon-" + str(Beacon) + "/"
#
#     while True:
#         ret, frame = read_frame(image_path, Beacon, frame_count, data_type, scale)
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
#
# def mark_ground_truth(worker, path, Beacon, data_type, scale, frame_sum):
#
#     out2 = None
#     frame_count = 0
#
#     vid = cv2.VideoCapture(path + "TimeLapseVideos/Beacon-" + str(Beacon) + "processed.avi")
#     # vid = None
#
#     image_path = path + "RawData/Beacon-" + str(Beacon) + "/"
#
#     # motion_vector = np.loadtxt(image_path + "input_images/" + "motion_vectors.txt")
#     # motion_vector = motion_vector.astype(int)
#
#     # skip = 3
#     if(vid):
#         skip = frame_sum - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#         print("skip: ", skip)
#
#     while True:
#         ret, frame = read_frame(image_path, Beacon, frame_count, data_type, scale)
#         if(ret == False):
#             print("done")
#             return
#
#         print("mark cells frame_count:" + str(frame_count))
#
#         if(len(frame.shape) == 2):
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#
#         gt_frame = np.zeros(0)
#
#         if(vid and frame_count >= skip):
#             ret, gt_frame = vid.read()
#
#         # frame = worker.mark(frame, frame_count, scale)
#         # ret_1, frame_1 = read_frame(image_path, Beacon, frame_count, data_type, 1)
#
#         frame = worker.mark_gt(frame, frame_count, scale, gt_frame, crop_height, crop_width, image_path)
#         # ph_detector.mark_cells(frame, frame_count)
#
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
#
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
#         image_path = path + "input_images/" + "t" + "{0:0=3d}".format(frame_count) + ".tif"
#         if ((not os.path.exists(image_path))): #  or frame_count > 30
#             print("file not exist: ", image_path)
#             return False, None
#         else:
#             # print(frame_count, image_path)
#             pass
#
#         # frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         # x = 619
#         # y = 321
#         # frame = frame[y:y + crop_height, x:x + crop_width]
#         frame = frame[0:crop_height, 0:crop_width]
#         if (scale > 1):
#             frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
#
#         # frame = frame[154: 625, 748: 1340] # cells
#         # frame = frame[154: 384, 748: 1040] # one ell
#         # frame = frame[264: 494, 434: 1043] # one cell
#         return True, frame
#
#     # frame = frame[0:crop_height * scale, 0:crop_width * scale]
#     # return True, frame
#
# # def preprocess(frame, save_image_path):
# #
# #     # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
# #     # cv2.resizeWindow('preprocess_0', 900, 900)
# #     # cv2.imshow('preprocess_0', frame)
# #
# #     if (len(frame.shape) > 2):
# #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     else:
# #         frame = frame
# #
# #     width = frame.shape[1]
# #     height = frame.shape[0]
# #     box_side = 64
# #     vertical = int(height / box_side + 1)
# #     horizontal = int(width / box_side + 1)
# #
# #     for i in range(vertical):
# #         for j in range(horizontal):
# #             start_x = box_side * j
# #             start_y = box_side * i
# #
# #             end_x = 0
# #             end_y = 0
# #
# #             if(i < vertical and j < horizontal):
# #                 end_x = start_x + box_side
# #                 end_y = start_y + box_side
# #             elif(j ==  horizontal):
# #                 end_x = 1328
# #                 end_y = start_y + box_side
# #             elif(i == vertical):
# #                 end_x = start_x + box_side
# #                 end_y = 1048
# #
# #
# #             window = frame[start_y:end_y, start_x:end_x]
# #
# #             histSize = 256
# #             histRange = (0, 256)  # the upper boundary is exclusive
# #             accumulate = False
# #             window_b_hist = cv2.calcHist([window], [0], None, [histSize], histRange, accumulate=accumulate)
# #
# #             window_hist_max_index = np.argmax(window_b_hist)
# #
# #             hist_max_index = window_hist_max_index
# #             hist_trans = np.zeros(256, dtype=np.uint8)
# #             target_peak = 100
# #
# #             for k in range(hist_max_index):
# #                 hist_trans[k] = target_peak * k / hist_max_index + 0.5
# #
# #             for m in range(hist_max_index, 256):
# #                 hist_trans[m] = ((255 - target_peak) * m - 255 * hist_max_index + target_peak * 255) / (255 - hist_max_index) + 0.5
# #
# #             window = hist_trans[window]
# #             frame[start_y:end_y, start_x:end_x] = window
# #
# #     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))#clipLimit=2.0,
# #     clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))#clipLimit=2.0,
# #     # gray = clahe.apply(gray)
# #     frame = clahe2.apply(frame)
# #
# #     cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
# #     cv2.resizeWindow('Tracking', 900, 900)
# #     cv2.imshow('Tracking', frame)
# #     cv2.waitKey()
# #
# #     np.save(save_image_path, frame)
# #     new_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
# #     return new_frame
#
# def preprocess_2(frame, save_image_path, scale = 1):
#     # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('preprocess_0', 900, 900)
#     # cv2.imshow('preprocess_0', frame)
#
#     if (len(frame.shape) > 2):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     frame_org = frame.copy()
#
#     t0 = time.time()
#     frame = cv2.medianBlur(frame_org, 81)#There is an unexpected effect when ksize as 81, applied to 8 times scaled image.
#
#     frame = frame / 100
#     frame = frame_org / frame
#     np.clip(frame, 0, 255, out = frame)
#     frame = frame.astype(np.uint8)
#
#     if(scale > 1):
#         frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation = cv2.INTER_CUBIC)
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     return frame
#

if __name__ == "__main__":
    # execute main
    main()
