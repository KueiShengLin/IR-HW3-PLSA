# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:44:27 2017

@author: Tim Lin
"""

import re
import gc
import numpy as np
import time
import threading
from numba import jit



# global variable
topic = 3
len_voc = 0
len_coll = 0
threadlock = threading.Lock()


class EmThread(threading.Thread):
    def __init__(self, threadid, name):
        threading.Thread.__init__(self)  # parent = super = threading.Thread
        self.threadid = threadid
        self.name = name

    def run(self):
        print('topic:' + self.name)
        em_step(self.threadid)


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


@jit
def em_step(top):
    global len_voc, len_coll
    global vc, tv, tc
    global new_tc, new_tv

    twd = np.zeros([len_voc, len_coll])     # g8 mem error

    print(time.strftime("%D,%H:%M:%S"))

    p_wt_dividen = []

    # print('topic:' + str(top))
    print(time.strftime("%D,%H:%M:%S"))
    # E_step:
    for word_id, word in enumerate(tv[top, :]):
        if word == 0:
            p_wt_dividen.append(0)
            continue
        p_wt = word
        dividend = 0
        for doc_id, doc in enumerate(tc[top, :]):
            total = np.dot(tv[:, word_id], tc[:, doc_id])
            twd[word_id, doc_id] = p_wt * doc / total  # p(w|t)*p(T|d)/total

            if str(word_id) in vc[doc_id]:
                dividend += vc[doc_id][str(word_id)] * twd[word_id, doc_id]

        p_wt_dividen.append(dividend)

        if word_id % 5000 == 0:
            # gc.collect()
            # time.sleep(1)
            print('k:' + str(top))
            print('E_step:' + str(word_id))
            print(time.strftime("%D,%H:%M:%S"))

    print('E_step finish!')

    print(time.strftime("%D,%H:%M:%S"))

    # M_step:(update: tv[:. top] td[top, :])
    divisor = sum(p_wt_dividen)
    print('divisor:' + str(divisor))

    for word_id, word in enumerate(tv[top, :]):
        if word == 0:
            continue
        threadlock.acquire()
        new_tv[top, word_id] = p_wt_dividen[word_id] / divisor
        threadlock.release()

        if word_id % 5000 == 0:
            gc.collect()
            time.sleep(1)
            print('k:' + str(top))
            print('M:p(w|t):' + str(word_id))

    print(time.strftime("%D,%H:%M:%S"))

    for doc_id, doc in enumerate(vc[:]):
        dividend = 0   # 被
        divisor = sum(doc.values())
        for dict_word in doc:
            dividend += doc[dict_word] * twd[int(dict_word), doc_id]

        threadlock.acquire()
        new_tc[top, doc_id] = dividend / divisor
        threadlock.release()

        if doc_id % 5000 == 0:
            print('k:' + str(top))
            print('M:p(t|d):' + str(doc_id))

    print("M_step finish!!!!!")
    print(time.strftime("%D,%H:%M:%S"))


    # print('iteration down')
    # print(time.strftime("%D,%H:%M:%S"))

gc.enable()
(vc, tv, tc) = init()   # vc = voc*coll, tv = voc*topic, tc = topic*coll
new_tv = np.zeros([topic, len_voc])
new_tc = np.zeros([topic, len_coll])

gc.collect()
time.sleep(2)
print('gc collect~')

tc = np.loadtxt('model\\K3_iteration\\plsa_tc_K3_14', delimiter=',')
tv = np.loadtxt('model\\K3_iteration\\plsa_tv_K3_14', delimiter=',')

thread1 = EmThread(0, "k1")
thread2 = EmThread(1, "k2")
thread3 = EmThread(2, "k3")
thread1.start()
thread2.start()
thread3.start()
# thread1.join()
# thread2.join()

while thread1.isAlive() or thread2.isAlive() or thread3.isAlive():
    time.sleep(1)

np.savetxt('plsa_tv_K2_' + str(0), new_tv, delimiter=',')
np.savetxt('plsa_tc_K2_' + str(0), new_tc, delimiter=',')
#
