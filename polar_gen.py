# Simulator ###########################################################################
#
# Copyright (c) 2021, Mohammad Rowshan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that:
# the source code retains the above copyright notice, and te redistribtuion condition.
# 
# Freely distributed for educational and research purposes
#######################################################################################

from time import time
import numpy as np
import polar_coding_functions as pcf
from polar_code import PolarCode
from channel import channel
from rate_profile import rateprofile
from crclib import crc
# import csv
# import h5py
# import scipy.io as sio
import pickle
from multiprocessing import Pool

def distribute_samples_according_to_ratio(total_samples, no_snrs, ratios):
    # 比例因子
    total_ratio = sum(ratios)

    # 计算每个类别应分配的样本数
    samples_per_category = [total_samples * ratio // total_ratio for ratio in ratios]

    # 分配后可能还有剩余的样本，因为用整数除法会丢失小数部分
    # 将剩余的样本按比例分配，从比例最高的类别开始
    residual_samples = total_samples - sum(samples_per_category)
    for i in sorted(range(no_snrs), key=lambda x: -ratios[x]):
        if residual_samples <= 0:
            break
        samples_per_category[i] += 1
        residual_samples -= 1

    return samples_per_category


coding = "Polar"        # Polar or PAC
reco_mode = 'len'
trainortest = 'train'
if trainortest == 'train':
    train_mode = 'ubzero'
else:
    train_mode = 'origin'
if trainortest == 'train':
    no_samples_total = 4800000
    snr_range = np.arange(5,13,1) # in dB, (start,endpoint+step,step)
    # snr_range = np.arange(12,13,1) # in dB, (start,endpoint+step,step)
    if train_mode == 'ubzero' or train_mode == 'bzero':
        no_samples_total = 240000 * 20
        # if reco_mode == 'len':
        #     no_samples_total = 240000 * 10
        snr_range = np.arange(0,13,1) # in dB, (start,endpoint+step,step)
else:
    no_samples_total = 240000
    snr_range = np.arange(0,13,1) # in dB, (start,endpoint+step,step)
    # snr_range = np.arange(12,13,1) # in dB, (start,endpoint+step,step)
# N = 2**6
if reco_mode in ['rate', 'prate']:
    Ns = 2**np.arange(8, 9, 1)
else:
    Ns = 2**np.arange(8, 13, 1)
no_Ns = len(Ns)
if reco_mode == 'len':
    # 计算每种码长应分配的样本数，使得每种码长的所有样本的总长度相等
    # 总长度是固定的，由于总长度 = 码长 * 样本数，我们希望所有码长的总长度相等
    # 因此，每种码长分配的样本数与码长成反比

    # 码长的倒数
    inverse_Ns = 1 / Ns

    # 确定比例因子，使得总样本长度等于no_samples_total
    # 因为总样本长度 = sum(样本数 * 码长) = 固定比例 * sum(码长的倒数 * 码长) = 固定比例 * len(Ns)
    # 所以固定比例 = no_samples_total / len(Ns)
    factor = no_samples_total / np.sum(inverse_Ns)

    # 计算每种码长分配的样本数
    no_samples_Ns = (inverse_Ns * factor).astype(int)
else:
    no_samples_per_N = no_samples_total//no_Ns
    no_samples_Ns = [no_samples_per_N + (no_samples_total%no_Ns>i) for i in range(no_Ns)]
# R = 0.5
if reco_mode == 'len':
    Rs = np.array([1/8])
elif reco_mode == 'prate':
    Rs = np.arange(1, 5, 1)/8
else:
    Rs = np.arange(1, 8, 1)/8
no_Rs = len(Rs)
crc_len = 0             # Use 8,12,16 along with the corresponding crc_poly below  # Use 0 for no CRC. # It does not support 6, 9, 11, 24, etc.
crc_poly = 0xA5         # 0x1021 for crc_len=16  # 0xC06  for crc_len=12 # 0xA6  for crc_len=8
list_size = 2**0        # Keep this as it is. Change list_size_max for your desired L.
list_size_max = 2**5    # For adaptive two-stage list decoding to accelerate the decoding process: first with L=list_size. If the deocing fails, then with L=list_size_max
designSNR = 0           # For instance for P(128,64):4, P(512,256):2
# designSNRs = *
# profile_name = "dega"   # Use "rm-polar" for Read-Muller polar code construction, #"pw" for polarization weight construction, "bh"  for Bhattachariya parameter/bound construction.
profile_names = ["pw", "dega"] 
# profile_names = ["pw"] 
no_profile_names = len(profile_names)

# For polar coding, set conv_gen = [1] which makes v=u meaninng no precoding.
if coding == "Polar":  
    conv_gen = [1]          # Use [1] for polar codes
elif coding == "PAC":
    conv_gen = [1,0,1,1,0,1,1] 

snrb_snr = 'SNRb'       # 'SNRb':Eb/N0 or 'SNR':Es/N0
modu = 'BPSK'           # It does not work for higher modulations

no_snrs = len(snr_range)
err_cnt = 50            # The number of error counts for each SNR point, for convergence, choose larger counts for low SNR regimes and for short codes.

systematic = True

# Error Coefficient-reduced: For code modifcation by X number of rows in G_N
# The maximum number of row modifications, you can choose 2 as well. If you increase it, the error coefficient might get betetr but the BLER may not. 
# See https://arxiv.org/abs/2111.08843
max_row_swaps  = 0      # 0 for no construction midifcation # 2 and 3 is recommended. 

class BERFER():
    """structure that keeps results of BER and FER tests"""
    def __init__(self):
        self.fname = str()
        self.label = str()
        self.snr_range = list()
        self.ber = list()
        self.fer = list()

class messageFactory:
    def __init__(self, isCRCinc, K, crc1, coding, pcode, ch, systematic, conv_gen, mem):
        self.isCRCinc = isCRCinc
        self.K = K
        self.crc1 = crc1
        self.coding = coding
        self.pcode = pcode
        self.ch = ch
        self.systematic = systematic
        self.conv_gen = conv_gen
        self.mem = mem


    def __call__(self, i):
        # Generating a K-bit binary pseudo-radom message
        #np.random.seed(t)
        message = np.random.randint(0, 2, size=self.K, dtype=int)
        if self.isCRCinc:
            message = np.append(message, self.crc1.crcCalc(message))

        if self.coding == "Polar":
            x = self.pcode.encode(message, self.systematic)
        elif self.coding == "PAC":
            x = self.pcode.pac_encode(message, self.conv_gen, self.mem, self.systematic)
        
        modulated_x = self.ch.modulate(x)
        # y = ch.add_noise(modulated_x)
        # samples_snr[t] = ch.add_noise(modulated_x)
        return self.ch.add_noise(modulated_x)

#crc_len = len(bin(crc_poly)[2:].zfill(len(bin(crc_poly)[2:])//4*4+(len(bin(crc_poly)[2:])%4>0)*4))
no_diff_sam = no_Ns*no_Rs*no_profile_names*no_snrs
# 创建大小为no_diff_sam的list
dataset = []
label_r = np.zeros(no_diff_sam)
label_n = np.zeros(no_diff_sam)
label_g = np.zeros(no_diff_sam, dtype=np.str)
label_s = np.zeros(no_diff_sam)
for i in range(no_Ns): 
    # 第i个N下每种R的样本数
    no_samples_per_R = no_samples_Ns[i] // no_Rs
    no_samples_Rs = [no_samples_per_R + (no_samples_Ns[i]%no_Rs>j) for j in range(no_Rs)]
    for j in range(no_Rs):
        # 第j个R下每种profile的样本数
        no_samples_per_profile = no_samples_Rs[j] // no_profile_names
        no_samples_profiles = [no_samples_per_profile + (no_samples_Rs[j]%no_profile_names>k ) for k in range(no_profile_names)]
        for k in range(no_profile_names):
            K = int(Ns[i]*Rs[j])
            nonfrozen_bits = K + crc_len
            mem = len(conv_gen)-1

            rprofile = rateprofile(Ns[i],nonfrozen_bits,designSNR,max_row_swaps)

            crc1 = crc(int(crc_len), crc_poly)
            pcode = PolarCode(Ns[i], nonfrozen_bits, profile_names[k], L=list_size, rprofile=rprofile)
            pcode.iterations = 10**7    # Maximum number of iterations if the errors found is less than err_cnt
            pcode.list_size_max = list_size_max

            print("{0}({1},{2}) constructed by {3}({4}dB)".format(coding, Ns[i], nonfrozen_bits,crc_len,profile_names[k],designSNR))
            print("L={} & c={}".format(list_size,conv_gen))
            print("dataset generation is started")

            st = time()
            isCRCinc = True if crc_len>0 else False


            result = BERFER()

            pcode.m = mem
            pcode.gen = conv_gen
            pcode.cur_state = [0 for i in range(mem)]
            log_M = 1   #M:modulation order
            if train_mode == 'ubzero':
                # no_samples_snrs = distribute_samples_according_to_ratio(no_samples_profiles[k], no_snrs, [6, 7, 8, 9] + [25] * (no_snrs - 4))
                # no_samples_snrs = distribute_samples_according_to_ratio(no_samples_profiles[k], no_snrs, [5, 6, 7, 8] + [25] * (no_snrs - 4))
                no_samples_snrs = distribute_samples_according_to_ratio(no_samples_profiles[k], no_snrs, [5, 6, 7, 8] + [10] * (no_snrs - 4))
                if reco_mode == 'len':
                    no_samples_snrs = distribute_samples_according_to_ratio(no_samples_profiles[k], no_snrs, [6, 7, 8, 9, 10] + [25] * (no_snrs - 5))
            else:
                no_samples_per_snr = no_samples_profiles[k] // no_snrs
                no_samples_snrs = [no_samples_per_snr + (no_samples_profiles[k]%no_snrs>l) for l in range(no_snrs)]
            for l in range(no_snrs):
                print("\nSNR={0} dB".format(snr_range[l]))
                ber = 0
                fer = 0
                ch = channel(modu, snr_range[l], snrb_snr, (K / Ns[i])) 
                samples_snr = np.zeros((no_samples_snrs[l], Ns[i]), dtype=int)
                with Pool(10) as p:
                    polarFactoryres = p.map(messageFactory(isCRCinc, K, crc1, coding, pcode, ch, systematic, conv_gen, mem), range(no_samples_snrs[l]))
                for t in range(no_samples_snrs[l]):
                    samples_snr[t] = polarFactoryres[t]
                del polarFactoryres
                # for t in range(no_samples_snrs[l]):
                #     # Generating a K-bit binary pseudo-radom message
                #     #np.random.seed(t)
                #     message = np.random.randint(0, 2, size=K, dtype=int)
                #     if isCRCinc:
                #         message = np.append(message, crc1.crcCalc(message))

                #     if coding == "Polar":
                #         x = pcode.encode(message, systematic)
                #     elif coding == "PAC":
                #         x = pcode.pac_encode(message, conv_gen, mem, systematic)
                    
                #     modulated_x = ch.modulate(x)
                #     # y = ch.add_noise(modulated_x)
                #     samples_snr[t] = ch.add_noise(modulated_x)
                dataset.append(samples_snr)
                label_r[i*no_Rs*no_profile_names*no_snrs+j*no_profile_names*no_snrs+k*no_snrs+l] = Rs[j]
                label_n[i*no_Rs*no_profile_names*no_snrs+j*no_profile_names*no_snrs+k*no_snrs+l] = Ns[i]
                label_g[i*no_Rs*no_profile_names*no_snrs+j*no_profile_names*no_snrs+k*no_snrs+l] = profile_names[k]
                label_s[i*no_Rs*no_profile_names*no_snrs+j*no_profile_names*no_snrs+k*no_snrs+l] = snr_range[l]

# pickle保存
dataset_polar = {'dataset':dataset, 'label_r':label_r, 'label_n':label_n, 'label_g':label_g, 'label_s':label_s}
# if trainortest == 'train':
if reco_mode == 'prate':
    with open('dataset_polar_%s_s%d_%d_%s_sys_n256_ubzero_pwga.pkl' % (trainortest, snr_range[0], snr_range[-1], reco_mode), 'wb') as f:
        pickle.dump(dataset_polar, f)

if reco_mode == 'len':
    with open('dataset_polar_%s_s%d_%d_%s_sys_r0.125_ubzero_pwga.pkl' % (trainortest, snr_range[0], snr_range[-1], reco_mode), 'wb') as f:
        pickle.dump(dataset_polar, f)
if reco_mode == 'type':
    with open('dataset_polar_%s_s%d_%d_%s_sys_bzero_pwga.pkl' % (trainortest, snr_range[0], snr_range[-1], reco_mode), 'wb') as f:
        pickle.dump(dataset_polar, f)
# else:
#     with open('dataset_polar_test_s%d_%d_%s_sys.pkl' % (snr_range[0], snr_range[-1], reco_mode), 'wb') as f:
#         pickle.dump(dataset_polar, f)
# # sio保存
# sio.savemat('dataset_polar.mat', {'dataset':dataset, 'label_r':label_r, 'label_n':label_n, 'label_g':label_g, 'label_s':label_s})
# h5py保存
# with h5py.File('dataset.mat', 'w') as f:
#     f.create_dataset('dataset', data=dataset)
#     f.create_dataset('label_r', data=label_r)
#     f.create_dataset('label_n', data=label_n)
#     f.create_dataset('label_g', data=label_g)
        #         llr_ch = ch.calc_llr3(y)

        #         decoded = pcode.pac_list_crc_decoder(llr_ch, 
        #                                             systematic,
        #                                             isCRCinc,
        #                                             crc1, 
        #                                             list_size)
                
        #         if pcf.fails(message, decoded)>0:
        #                 pcode.edgeOrder = [0 for k in range(pcode.list_size_max)] #np.zeros(L, dtype=int)
        #                 pcode.dLLRs = [0 for k in range(pcode.list_size_max)]
        #                 pcode.PMs = [0 for k in range(pcode.list_size_max)]
        #                 pcode.pathOrder = [0 for k in range(pcode.list_size_max)]
        #                 decoded = pcode.pac_list_crc_decoder(llr_ch, systematic, isCRCinc, crc1, pcode.list_size_max)
                
        #         ber += pcf.fails(message, decoded)
        #         if not np.array_equal(message, decoded):
        #             fer += 1
        #             print("Error # {0} t={1}, FER={2:0.2e}".format(fer,t, fer/(t+1))) #"\nError #
        #         #fer += not np.array_equal(message, decoded)
        #         if fer > err_cnt:    #//:Floor Division
        #             print("@ {0} dB FER is {1:0.2e}".format(snr, fer/(t+1)))
        #             break
        #         if t%2000==0:
        #             print("t={0} FER={1} ".format(t, fer/(t+1)))
        #         if t==pcode.iterations:
        #             break
        #     #print("{0} ".format(ber))
        #     result.snr_range.append(snr)
        #     result.ber.append(ber / ((t + 1) * nonfrozen_bits))
        #     result.fer.append(fer / (t + 1))

        #     print("\n\n")
        #     print(result.label)
        #     print("SNR\t{0}".format(result.snr_range))
        #     print("FER\t{0}".format(result.fer))
        #     print("time on test = ", str(int((time() - st)/60)), ' min\n------------\n')


        # #Filename for saving the results
        # result.fname += "{0}({1},{2}),L{3},m{4}".format(coding, N, pcode.nonfrozen_bits,list_size,mem)
        # if isCRCinc:
        #     result.fname += ",CRC{0}".format(crc_len)
            
        # # Writing the results in file
        # with open(result.fname + ".csv", 'a') as f:
        #     result.label = "{0}({1}, {2})\nL={3}\nRate-profile={4}\ndesign SNR={5}\n" \
        #                 "Conv Poly={6}\nCRC={7} bits, Systematic={8}\n".format(coding, N, pcode.nonfrozen_bits,
        #                 pcode.list_size, profile_name, designSNR, conv_gen, crc_len, systematic)
        #     f.write(result.label)

        #     f.write("\nSNR: ")
        #     for snr in result.snr_range:
        #         f.write("{0}; ".format(snr))
        #     f.write("\nBER: ")
        #     for ber in result.ber:
        #         f.write("{0}; ".format(ber))
        #     f.write("\nFER: ")
        #     for fer in result.fer:
        #         f.write("{0}; ".format(fer))
        #     f.write("\n")

        # print("\n\n")
        # print(result.label)
        # print("SNR\t{0}".format(result.snr_range))
        # print("BER\t{0}".format(result.ber))
        # #print("FER\t{0:1.2e}".format(result.fer))
        # print("FER\t{0}".format(result.fer))

        # print("time on test = ", str(time() - st), ' s\n------------\n')


"""
with open("bit_err_cnt.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ',', lineterminator = '\n') #Default:'\r\n' used in Unix # creating a csv writer object
        #csvwriter.writerow(row)
        csvwriter.writerows(map(lambda x: [x], pcode.bit_err_cnt[pcode.bitrev_indices]))"""
