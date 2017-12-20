# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:44:27 2017

@author: Tim Lin
"""

import re
import gc
import numpy as np
import time
from numba import jit

# global variable
topic = 100
len_voc = 0
len_coll = 0


def init():
    coll_list = []
    len_voc_f = 0
    len_coll_f = 0
    with open('BGLM.txt') as BG:
        for len_voc_f, val_voc in enumerate(BG):
            pass

    with open('Collection.txt') as coll:
        for len_coll_f, val_coll in enumerate(coll):
            coll_line = re.split(' ', val_coll)
            coll_line.remove('\n')
            coll_line = list(filter(''.__ne__, coll_line))  # filter:傳回第一個值為true的第二個值  __ne__ : !=
            coll_list.append(coll_line)
            del coll_line

    global len_voc
    len_voc = len_voc_f + 1
    global len_coll
    len_coll = len_coll_f + 1

    vc = []

    for coll_num, coll_doc in enumerate(coll_list):
        vc_d = {}
        for doc_voc in coll_doc:
            if doc_voc not in vc_d:
                vc_d[doc_voc] = 1
            else:
                vc_d[doc_voc] += 1
        vc.append(vc_d)

    tv = np.random.uniform(0, 10, size=(topic, len_voc))
    tc = np.random.uniform(0, 10, size=(topic, len_coll))
    tv, tc = normalize(tv, tc)

    print("init down!")
    del coll_list
    return vc, tv, tc


def normalize(tv, tc):
    for t in range(0, topic):
        tv_row_amount = np.sum(tv[t, :])
        for j, top_voc in enumerate(tv[t, :]):
            tv[t, j] = tv[t, j] / tv_row_amount
    print(np.sum(tv[1, :]))
    tc_col_amount = np.sum(tc, axis=0)
    for i, amount in enumerate(tc_col_amount):
        for j, coll_top in enumerate(tc[:, i]):
            tc[j, i] = coll_top / amount
    print(np.sum(tc[:, 20]))

    return tv, tc


def em_step(vc, tv, tc):
    global len_voc
    global len_coll
    print(len_voc)
    print(len_coll)

    for iteration in range(1, 100):
        print('iteration:' + str(iteration))
        print(time.strftime("%D,%H:%M:%S"))
        new_tv = np.zeros([topic, len_voc])
        new_tc = np.zeros([topic, len_coll])
        for top in range(0, topic):
            twd = np.zeros([len_voc, len_coll])  # g8 mem error

            #print('topic:' + str(top))
            #print(time.strftime("%D,%H:%M:%S"))

            # E_step:
            for doc_id, doc in enumerate(tc[top, :]):
                for word in vc[doc_id]:
                    twd[int(word), doc_id] = tv[top, int(word)] * doc / np.dot(tv[:, int(word)], tc[:, doc_id])


            # M_step:
            p_wt_dividen = [0] * len_voc

            for doc_id, doc in enumerate(vc[:]):
                for word in doc:
                    p_wt_dividen[int(word)] += doc[word] * twd[int(word), doc_id]

            divisor = sum(p_wt_dividen)
            #print('divisor:' + str(divisor))
            new_tv[top, :] = [p_wt_dividen_num / divisor for p_wt_dividen_num in p_wt_dividen]


            del p_wt_dividen
            gc.collect()
            time.sleep(1)

            for doc_id, doc in enumerate(vc[:]):
                divisor = sum(doc.values())

                dividend = sum(doc[dict_word] * twd[int(dict_word), doc_id] for dict_word in doc)

                new_tc[top, doc_id] = dividend / divisor

        tv = new_tv
        tc = new_tc
        print('iteration down')
        print(time.strftime("%D,%H:%M:%S"))

        np.savetxt('plsa_tv_K100.txt', new_tv, delimiter=',')
        np.savetxt('plsa_tc_K100.txt', new_tc, delimiter=',')


def main():
    (vc, tv, tc) = init()  # vc = voc*coll, tv = voc*topic, tc = topic*coll
    #tc = np.loadtxt('plsa_tc_K10_5', delimiter=',')
    #tv = np.loadtxt('plsa_tv_K10_5', delimiter=',')
    gc.collect()
    time.sleep(2)
    print('gc collect~')
    em_step(vc, tv, tc)


gc.enable()

main()
#
