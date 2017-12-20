# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:44:27 2017

@author: Tim Lin
"""

import re
import gc
import numpy as np
import time
from math import log10


# global variable
topic = 5


def init():
    coll_list = []
    len_voc = 0
    len_coll = 0

    with open('BGLM.txt') as BG:
        for len_voc, val_voc in enumerate(BG):
            pass

    with open('Collection.txt') as coll:
        for len_coll, val_coll in enumerate(coll):
            coll_line = re.split(' ', val_coll)
            coll_line.remove('\n')
            coll_line = list(filter(''.__ne__, coll_line))  # filter:傳回第一個值為true的第二個值  __ne__ : !=
            coll_list.append(coll_line)

    len_voc += 1
    len_coll += 1

    vc = np.zeros([len_voc, len_coll])
    print(vc.shape)

    for coll_num, coll_doc in enumerate(coll_list):
        for doc_voc in coll_doc:
            vc[int(doc_voc), coll_num] += 1

    tv = np.random.uniform(0, 5, size=(topic, len_voc))
    tc = np.random.uniform(0, 5, size=(topic, len_coll))

    tv, tc = normalize(tv, tc)

    print("init down!")
    del coll_list
    return vc, tv, tc


def normalize(tv, tc):
    tv_col_amount = np.sum(tv, axis=0)
    for i, amount in enumerate(tv_col_amount):
        for j, top_voc in enumerate(tv[:, i]):
            tv[j, i] = top_voc / amount

    tc_col_amount = np.sum(tc, axis=0)
    for i, amount in enumerate(tc_col_amount):
        for j, coll_top in enumerate(tc[:, i]):
            tc[j, i] = coll_top / amount

    return tv, tc


def em_step(vc, tv, tc):
    t_start = time.time()
    (len_voc, len_coll) = vc.shape
    twd = np.zeros([len_voc, len_coll])     # g8 mem error
    tv_row_amount = np.zeros(topic)
    tc_col_amount = np.sum(tc, axis=0)
    for t in range(0, topic):
        for tv_row in tv[t, :]:
            tv_row_amount[t] += tv_row
    tv_row_amount_total = np.sum(tv_row_amount)
    tc_col_amount_total = np.sum(tc_col_amount)
    new_tv = np.copy(tv)
    new_tc = np.copy(tc)

    for top in range(0, topic):
        print('topic:' + str(top))
        # E_step:
        p_d = (tc_col_amount[top] / tc_col_amount_total)
        for word_id, word in enumerate(tv[top, :]):
            t_w = time.time()
            p_wt = (word/(tv_row_amount[top] / tv_row_amount_total))
            for doc_id, doc in enumerate(tc[top, :]):
                total = 0
                for k in range(0, topic):
                    total += (tv[k, word_id]/(tv_row_amount[k] / tv_row_amount_total))*(tc[k, doc_id]/(tc_col_amount[k] / tc_col_amount_total))
                #   total = log10(total)
                twd[word_id, doc_id] = p_wt * (doc/p_d) / total     # p(w|t)*p(d|d)/total
            if word_id % 10 == 0:
                print(word_id)
                print('twd:' + str(twd[word_id, 0]))
                print(time.time() - t_w)

        t_e = time.time()
        print("E_step finish time:" + str(t_e - t_start))

        # M_step:(update: tv[:. top] td[top, :])
        divisor = 0
        for word_id_p in range(0, len_voc):
            for doc_id_p in range(0, len_coll):
                if vc[word_id_p, doc_id_p] != 0:
                    divisor += vc[word_id_p, doc_id_p] * twd[word_id_p, doc_id_p]
            if word_id_p % 5000 == 0:
                print(word_id_p)
                print(divisor)

        t_div = time.time()
        print('divisor down time:' + str(t_div - t_e))

        for word_id, word in enumerate(tv[top, :]):
            dividend = 0    # 被
            for doc_id, doc in enumerate(vc[word_id, :]):
                if doc != 0:
                    dividend += doc*twd[word_id, doc_id]
            new_tv[top, word_id] = dividend / divisor
            if word_id % 5000 == 0:
                print(word_id)
                print(new_tv[top, word_id])

        t_tv = time.time()
        print('tv down time:' + str(t_tv - t_div))

        for doc_id, doc in enumerate(tc[top, :]):
            dividend = 0    # 被
            divisor = 0
            for word_amount_id, word_amount in enumerate(vc[:, doc_id]):
                dividend += word_amount * twd[word_amount_id, doc_id]
                divisor += word_amount
            new_tc[top, doc_id] = dividend / divisor
            if doc_id % 5000 == 0:
                print(doc_id)
                print(new_tc[top, doc_id])

        t_tc = time.time()
        print('tc down time' + str(t_tc - t_tv))

        print("S_step finish!!!!! time:" + str(time.time() - t_e))

    print('iteration down')

    # Normalize
    new_tv, new_tc = normalize(new_tv, new_tc)

    np.savetxt('plsa_tv', new_tv, delimiter=',', fmt='%f')
    np.savetxt('plsa_tc', new_tc, delimiter=',', fmt='%f')

    return new_tv, new_tc


def main():
    (vc, tv, tc) = init()   # vc = voc*coll, tv = voc*topic, tc = topic*coll
    gc.collect()
    time.sleep(10)
    print('123')
    tv, tc = em_step(vc, tv, tc)


gc.enable()

main()

