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
    global len_voc
    global len_coll
    print(len_voc)
    print(len_coll)
    twd = np.zeros([len_voc, len_coll])     # g8 mem error
    tv_row_amount = np.zeros(topic)
    tc_col_amount = np.sum(tc, axis=0)
    for t in range(0, topic):
        for tv_row in tv[t, :]:
            tv_row_amount[t] += tv_row
    tv_row_amount_total = np.sum(tv_row_amount)
    tc_col_amount_total = np.sum(tc_col_amount)
    new_tv = np.zeros([topic, len_voc])
    new_tc = np.zeros([topic, len_coll])

    for iteration in range(1, 10):
        print('iteration:' + str(iteration))
        print(time.strftime("%D,%H:%M:%S"))

        for top in range(0, topic):
            print('topic:' + str(top))
            print(time.strftime("%D,%H:%M:%S"))
            # E_step:
            # p_d = (tc_col_amount[top] / tc_col_amount_total)
            # p_t = (tv_row_amount[top] / tv_row_amount_total)
            for word_id, word in enumerate(tv[top, :]):
                if word == 0:
                    continue
                # p_wt = (word/p_t)
                p_wt = word
                for doc_id, doc in enumerate(tc[top, :]):
                    total = 0
                    for k in range(0, topic):
                        # total += (tv[k, word_id]/p_t)*(tc[k, doc_id]/p_d)
                        total += (tv[k, word_id] ) * (tc[k, doc_id] )
                    #   total = log10(total)
                    # twd[word_id, doc_id] = p_wt * (doc/p_d) / total     # p(w|t)*p(d|d)/total
                    twd[word_id, doc_id] = p_wt * doc / total  # p(w|t)*p(d|d)/total

                if word_id % 5000 == 0:
                    gc.collect()
                    time.sleep(1)
                    print(word_id)
                    print('twd:' + str(twd[word_id, 0]))
                    print(time.strftime("%D,%H:%M:%S"))

            print('E_step finish!')
            print(time.strftime("%D,%H:%M:%S"))

            # M_step:(update: tv[:. top] td[top, :])

            divisor = 0
            for doc_dict, doc_dict_num in enumerate(vc[:]):
                for dict_word in doc_dict_num:
                    divisor += doc_dict_num[str(dict_word)] * twd[int(dict_word), doc_dict]
            print('divisor' + str(divisor))
            print(time.strftime("%D,%H:%M:%S"))

            for word_id, word in enumerate(tv[top, :]):
                if word == 0:
                    continue
                dividend = 0    # 被
                for doc_id, doc in enumerate(vc[:]):
                    if str(word_id) in doc:
                        dividend += doc[str(word_id)] * twd[word_id, doc_id]
                new_tv[top, word_id] = dividend / divisor

                if word_id % 5000 == 0:
                    gc.collect()
                    time.sleep(1)
                    print(word_id)
                    print(new_tv[top, word_id])

            print(time.strftime("%D,%H:%M:%S"))

            for doc_id, doc in enumerate(vc[:]):
                dividend = 0   # 被
                divisor = sum(doc.values())
                for dict_word in doc:
                    dividend += doc[dict_word] * twd[int(dict_word), doc_id]
                new_tc[top, doc_id] = dividend / divisor

                if doc_id % 5000 == 0:
                    gc.collect()
                    time.sleep(1)
                    print(doc_id)
                    print(new_tc[top, doc_id])

            print("M_step finish!!!!!")
            print(time.strftime("%D,%H:%M:%S"))

        tv = new_tv
        tc = new_tc
        print('iteration down')
        print(time.strftime("%D,%H:%M:%S"))

        np.savetxt('plsa_tv_K5_' + str(iteration), new_tv, delimiter=',')
        np.savetxt('plsa_tc_K5_' + str(iteration), new_tc, delimiter=',')

        # Normalize
        # new_tv, new_tc = normalize(new_tv, new_tc)
        #
        # np.savetxt('plsa_tv_K3_nor', new_tv, delimiter=',', fmt='%10.10f')
        # np.savetxt('plsa_tc_K3_nor', new_tc, delimiter=',', fmt='%10.10f')

        # return new_tv, new_tc


def main():
    (vc, tv, tc) = init()   # vc = voc*coll, tv = voc*topic, tc = topic*coll
    tc = np.loadtxt('model\\K3_iteration\\plsa_tc_K3', delimiter=',')
    tv = np.loadtxt('model\\K3_iteration\\plsa_tv_K3', delimiter=',')
    gc.collect()
    time.sleep(2)
    print('gc collect~')
    em_step(vc, tv, tc)


gc.enable()

main()
#
