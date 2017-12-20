import numpy as np
import os
import re
import time
import math
# tc = np.loadtxt('model\\K5_iteration\\plsa_tc_K5_2', delimiter=',')
tv = np.loadtxt('plsa_tv_K30_100.txt', delimiter=',')

# tv = np.loadtxt('plsa_tv_K8_2', delimiter=',')
# tv = np.transpose(tv)

query = []
document = []
BG = []
doc_name = os.listdir("Document")  # Document file name
query_name = os.listdir("Query")  # Query file name

alpha = 0.3
beta = 0.3
topic = 30


def readfile():
    global query
    global document
    global BG
    global doc_name
    global query_name

    # read document , create dictionary
    for doc_id in doc_name:
        doc_dict = {}
        with open("Document\\" + doc_id) as doc_file:
            doc_file_content = doc_file.read()
            doc_voc = re.split(' |\n', doc_file_content)
            doc_voc = list(filter('-1'.__ne__, doc_voc))
            for dv_id, dv_voc in enumerate(doc_voc):
                if dv_id < 5:
                    continue
                if dv_voc in doc_dict:
                    doc_dict[dv_voc] += 1
                else:
                    doc_dict[dv_voc] = 1
            if '' in doc_dict:  # ? error
                doc_dict.pop('')
        document.append(doc_dict)

    for query_id in query_name:
        query_dict = {}
        with open("Query\\" + query_id) as query_file:
            query_file_content = query_file.read()
            query_voc = re.split(' |\n', query_file_content)
            query_voc = list(filter('-1'.__ne__, query_voc))
            for qv_id, qv_voc in enumerate(query_voc):
                if qv_voc in query_dict:
                    query_dict[qv_voc] += 1
                else:
                    query_dict[qv_voc] = 1
            if '' in query_dict:  # ? error
                query_dict.pop('')
        query.append(query_dict)

    # Load BG
    with open('BGLM.txt') as BG_file:
        for len_voc_f, val_voc in enumerate(BG_file):
            bg_split = re.split('   |\n', val_voc)
            BG.append(float(bg_split[1]))

    print('read file down')


def fold_in():
    global tv
    global document

    foldin_td = np.random.uniform(0, 5, size=(topic, len(document)))
    for doc_id in range(0, len(document)):
        norm = np.sum(foldin_td[:, doc_id])
        for d_id in range(0, len(foldin_td[:, doc_id])):
            foldin_td[d_id, doc_id] = foldin_td[d_id, doc_id] / norm

    # foldin_td = np.full((topic, len(document)), 1/topic)
    new_td = np.zeros(foldin_td.shape)
    #foldin_td = np.loadtxt("foldin_td_6.txt",delimiter=',')
    for iteration in range(1, 11):

        print('iteration:' + str(iteration))
        print(time.strftime("%D,%H:%M:%S"))

        for top in range(0, topic):
            # print('topic:' + str(top))

            twd = np.zeros([tv.shape[1], foldin_td.shape[1]])
            # E_step
            for doc_id, doc in enumerate(foldin_td[top, :]):
                for word in document[doc_id]:
                    if tv[top, int(word)] == 0:
                        continue
                    twd[int(word), doc_id] = tv[top, int(word)] * doc / np.dot(tv[:, int(word)], foldin_td[:, doc_id])

            # for word_id, word in enumerate(tv[top, :]):
            #     if word == 0:
            #         continue
            #     twd[word_id, :] = [word * doc / np.dot(tv[:, word_id], foldin_td[:, doc_id]) for doc_id, doc in enumerate(foldin_td[top, :])]
            #     if 0 in twd[word_id, :]:
            #         print(word_id)

            # print('E_step finish!')
            # print(time.strftime("%D,%H:%M:%S"))

            # M_Step
            for doc_id, doc in enumerate(document):
                # dividend = 0  # 被
                divisor = sum(doc.values())
                dividend = sum(doc[dict_word] * twd[int(dict_word), doc_id]for dict_word in doc)
                # for dict_word in doc:
                #     dividend += doc[dict_word] * twd[int(dict_word), doc_id]
                new_td[top, doc_id] = dividend / divisor

            # print("M_step finish!!!!!")
            # print(time.strftime("%D,%H:%M:%S"))

        print('new_tc0 sum:' + str(np.sum(new_td[:, 0])))
        print('new_tc100 sum:' + str(np.sum(new_td[:, 100])))
        print('new_tc1000 sum:' + str(np.sum(new_td[:, 1000])))

        np.savetxt('foldin_td_30.txt', new_td, delimiter=',')
        foldin_td = new_td

        new_td = np.zeros(foldin_td.shape)

    return new_td


def likelihood(td):
    global query, document
    global BG, tv
    global alpha, beta, topic

    BG_sum = sum(BG)
    retrieval_ans = []

    for query_dict in query:
        p_qd = []

        for document_id, document_dict in enumerate(document):
            p_qd_likelihood = 1
            document_sum = np.sum(td[:, document_id])
            for query_voc in query_dict:
                p_plsa = 0
                p_wd = 0
                if query_voc in document_dict:
                    p_wd = document_dict[query_voc] / sum(document_dict.values())

                for k in range(0, topic):
                    # print(np.sum(tv[k, :]))
                    p_plsa += (tv[k, int(query_voc)] ) * (td[k, document_id] / document_sum)

                p_qwd = (alpha * p_wd) + (beta * p_plsa) + ((1 - alpha - beta) * math.exp(BG[int(query_voc)]))
                p_qd_likelihood *= p_qwd

            p_qd.append(p_qd_likelihood)

        retrieval_ans.append(p_qd)

    return retrieval_ans


def writefile(retrieval):
    global query_name
    global doc_name
    with open('Relevant_30_03_0310.txt', 'w') as retrieval_file:
        retrieval_file.write("Query,RetrievedDocuments\n")

        for retrieval_id, retrieval_list in enumerate(retrieval):
                retrieval_file.write(query_name[retrieval_id] + ',')
                sort = sorted(retrieval_list, reverse=True)
                for sort_list in sort:
                    retrieval_file.write(doc_name[retrieval_list.index(sort_list)] + ' ')
                if retrieval_id != len(query_name) - 1:
                    retrieval_file.write('\n')


print('process start')
print(time.strftime("%D,%H:%M:%S"))

readfile()
print('init down')
print(time.strftime("%D,%H:%M:%S"))

td = fold_in()
print('foldin finish')
print(time.strftime("%D,%H:%M:%S"))

# td = np.loadtxt('foldin_td_30.txt', delimiter=',')

retrieval = likelihood(td)
print('likelihood finish')


writefile(retrieval)
print('process finish')
print(time.strftime("%D,%H:%M:%S"))

#
