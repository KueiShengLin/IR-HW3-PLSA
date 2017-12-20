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
topic = 3
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

    tv = np.random.uniform(0, 5, size=(topic, len_voc))
    tc = np.random.uniform(0, 5, size=(topic, len_coll))

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

    for iteration in range(16, 21):
        print('iteration:' + str(iteration))
        print(time.strftime("%D,%H:%M:%S"))

        new_tv = np.zeros([topic, len_voc])
        new_tc = np.zeros([topic, len_coll])
        # p_wt_divisor = 0

        for top in range(0, topic):
            twd = np.zeros([len_voc, len_coll])  # g8 mem error
            p_wt_dividen = []
            print('topic:' + str(top))
            print(time.strftime("%D,%H:%M:%S"))
            # E_step:
            for word_id, word in enumerate(tv[top, :]):
                if word == 0:
                    p_wt_dividen.append(0)
                    continue
                p_wt = word

                twd[word_id, :] = [word * doc / np.dot(tv[:, word_id], tc[:, doc_id]) for doc_id, doc in enumerate(tc[top, :])]
                dividend = sum([vc[doc_id][str(word_id)] * twd[word_id, doc_id] for doc_id, doc in enumerate(tc[top, :]) if str(word_id) in vc[doc_id]])
                if(word_id == 596):
                    print(twd[596, 0])
                # for doc_id, doc in enumerate(tc[top, :]):
                #     total = np.dot(tv[:, word_id], tc[:, doc_id])  # / (p_d * p_t)
                #     twd[word_id, doc_id] = p_wt * doc / total  # p(w|t)*p(T|d)/total
                #     if str(word_id) in vc[doc_id]:
                #         dividend += vc[doc_id][str(word_id)] * twd[word_id, doc_id]

                p_wt_dividen.append(dividend)

                if word_id % 5000 == 0:
                    gc.collect()
                    time.sleep(1)
                    print('E STEP:' + str(word_id))
                    print(time.strftime("%D,%H:%M:%S"))

            print('E_step finish!')

            print(time.strftime("%D,%H:%M:%S"))

            # M_step:
            divisor = sum(p_wt_dividen)
            print('divisor:' + str(divisor))

            new_tv[top, :] = [p_wt_dividen_num / divisor for p_wt_dividen_num in p_wt_dividen]

            # for word_id, word in enumerate(tv[top, :]):
            #     if word == 0:
            #         continue
            #     new_tv[top, word_id] = p_wt_dividen[word_id] / divisor
            #
            #     if word_id % 5000 == 0:
            #         print('M step(p(w|t):' + str(word_id))
            print('new_tv sum:' + str(np.sum(new_tv[top, :])))
            print(time.strftime("%D,%H:%M:%S"))

            del p_wt_dividen
            gc.collect()
            time.sleep(1)

            for doc_id, doc in enumerate(vc[:]):
                divisor = sum(doc.values())
                dividend = sum([doc[dict_word] * twd[int(dict_word), doc_id] for dict_word in doc])

                new_tc[top, doc_id] = dividend / divisor

                if doc_id % 5000 == 0:
                    gc.collect()
                    time.sleep(1)
                    print('M step(p(t|d):' + str(doc_id))
                    print(time.strftime("%D,%H:%M:%S"))

            print("M_step finish!!!!!")
            print(time.strftime("%D,%H:%M:%S"))

        print('new_tc0 sum:' + str(np.sum(new_tc[:, 0])))
        print('new_tc100 sum:' + str(np.sum(new_tc[:, 100])))
        print('new_tc1000 sum:' + str(np.sum(new_tc[:, 1000])))

        tv = new_tv
        tc = new_tc
        print('iteration down')
        print(time.strftime("%D,%H:%M:%S"))

        np.savetxt('plsa_tv_K5_tarr' + str(iteration), new_tv, delimiter=',')
        np.savetxt('plsa_tc_K5_tarr' + str(iteration), new_tc, delimiter=',')


def main():
    (vc, tv, tc) = init()   # vc = voc*coll, tv = voc*topic, tc = topic*coll
    tc = np.loadtxt('model\\K3_iteration\\plsa_tc_K3_1', delimiter=',')
    tv = np.loadtxt('model\\K3_iteration\\plsa_tv_K3_1', delimiter=',')
    gc.collect()
    time.sleep(2)
    print('gc collect~')
    em_step(vc, tv, tc)


gc.enable()

main()
#
