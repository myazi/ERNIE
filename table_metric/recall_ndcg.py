import sys
import os
import math

def merge_text_and_score(text_file, score_file):
    # for Aurora relevance testset
    score_list = []
    for line in open(score_file):
        score_list.append(line.strip())

    qp_answer = []
    t_not_q_answer = []
    for i, line in enumerate(open(text_file)):
        line = line.strip().split('\t')
        if len(line) == 4:
            q, t, p, l = line
            isq = "0"
        else:
            q, t, p, l, isq = line
        isq = "0"
        qp_answer.append('\t'.join([q, t, p, l, isq, score_list[i]]))

        # title is question
        if float(isq) > 0.12:
            l = "0"
        t_not_q_answer.append('\t'.join([q, t, p, l, isq, score_list[i]]))
    return qp_answer, t_not_q_answer


def select_url_best_para(text_file, score_file):
    # for RAS testset
    score_list = []
    for line in open(score_file):
        score_list.append(line.strip())

    last_q = ''
    last_q_isq = 0
    last_u = ''

    url_max_score = 0
    url_best_para = ''

    qu_answer = []
    notq_qu_answer = []

    for i, line in enumerate(open(text_file)):
        v = line.strip().split('\t')
        q, u, t, p, l, isq = v

        if (q != last_q and last_q != '') or (u != last_u and last_u != ''):
            #print url_best_para + '\t' + str(url_max_score)
            # last query is not question
            if float(last_q_isq) < 0.12:
                #print >>sys.stderr, url_best_para + '\t' + str(url_max_score)
                notq_qu_answer.append(url_best_para + '\t' + str(url_max_score))
            qu_answer.append(url_best_para + '\t' + str(url_max_score))

            url_max_score = 0

        score = score_list[i]
        if float(score) > url_max_score:
            url_max_score = float(score)
            url_best_para = line.strip()

        last_q = q
        last_q_isq = isq
        last_u = u

    if float(last_q_isq) < 0.12:
        #print >>sys.stderr, url_best_para + '\t' + str(url_max_score)
        notq_qu_answer.append(url_best_para + '\t' + str(url_max_score))
    qu_answer.append(url_best_para + '\t' + str(url_max_score))

    return qu_answer, notq_qu_answer


def read_qu_answer(qu_answer):
    q_infos = {}
    for i, line in enumerate(qu_answer):
        v = line.strip().split('\t')
        q = v[0]
        l = v[-3]
        s = v[-1]
        if q not in q_infos:
            q_infos[q] = []
        q_infos[q].append([float(s), int(l), line.strip()])
    return q_infos


def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2**label_list[i] - 1)/math.log(i+2, 2)
        # dcg = (label_list[i])/math.log(i+2, 2)
        # print dcg
        dcgsum += dcg
    return dcgsum


def NDCG(label_list, topK):
    dcg = DCG(label_list[0:topK])
    # print 'dcg:', dcg
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG(ideal_list[0:topK])
    # print 'ideal_dcg', ideal_dcg
    if ideal_dcg == 0:
        return 0
    return dcg / ideal_dcg


def MRR(label_list, topK, min_true_label):
    mrr = 0
    for idx, label in enumerate(label_list[:topK]):
        if label >= min_true_label:
            mrr = 1.0 / (idx + 1)
            break
    return mrr


def cal_recall_q(q_infos, min_true_label):
    recall_1 = 0
    recall_4 = 0
    recall_10 = 0
    recall_30 = 0
    has_ans_q = 0.0001
    has_ans_q_ndcg = 0
    ndcg_10 = 0
    ndcg_4 = 0
    mrr_4 = 0
    mrr_10 = 0

    for q, infos in q_infos.items():
        infos = sorted(infos, key=lambda x:x[0], reverse=True)
        score, label, data = zip(*infos)

        new_label = []
        for ll in label:
            if int(ll) >= min_true_label:
                new_label.append(1)
            else:
                new_label.append(0)

        if sum(new_label) > 0:
            has_ans_q += 1
        if sum(label) > 0:
            has_ans_q_ndcg += 1
        if sum(new_label[:30]) > 0:
            recall_30 += 1
        if sum(new_label[:10]) > 0:
            recall_10 += 1
        if sum(new_label[:4]) > 0:
            recall_4 += 1
        if new_label[0] > 0:
            recall_1 += 1

        q_dcg4 = NDCG(label, 4)
        ndcg_4 += q_dcg4
        ndcg_10 += NDCG(label, 10)
        mrr_4 += MRR(label, 4, min_true_label)
        mrr_10 += MRR(label, 10, min_true_label)
        #print q + '\t' + str(q_dcg4)
        #for i in range(min(4, len(data))):
        #    print data[i]

    has_ans_rate = has_ans_q * 100.0 / len(q_infos)
    recall_1 = recall_1 * 100.0 / has_ans_q
    recall_4 = recall_4 * 100.0 / has_ans_q
    recall_10 = recall_10 * 100.0 / has_ans_q
    recall_30 = recall_30 * 100.0 / has_ans_q
    ndcg_4 = ndcg_4 * 100.0 / has_ans_q_ndcg #len(q_infos)
    ndcg_10 = ndcg_10 * 100.0 / has_ans_q_ndcg #len(q_infos)
    mrr_4 = mrr_4 * 100.0 / has_ans_q
    mrr_10 = mrr_10 * 100.0 / has_ans_q
#    print("q has answer rate: %s/%s=%.2f" % (has_ans_q, len(q_infos), has_ans_rate))
#    print("NDCG@4/10:      {:.2f}/{:.2f}".format(ndcg_4, ndcg_10))
#    print("MRR@4/10:       {:.2f}/{:.2f}".format(mrr_4, mrr_10))
#    print("q-Recall@4/10:  {:.2f}/{:.2f}".format(recall_4, recall_10))
    q_metrics = [ndcg_4, ndcg_10, mrr_4, mrr_10, recall_4, recall_10]
    return q_metrics


def cal_recall_qp(q_infos, min_true_label):
    recall_4 = 0
    recall_10 = 0
    pos_qp_amount_4 = 0
    pos_qp_amount_10 = 0

    has_ans_q = set("none")
    for q, infos in q_infos.items():
        infos = sorted(infos, key=lambda x:x[0], reverse=True)
        score, label, data = zip(*infos)

        q_pos_cnt = 0
        q_recall_4 = 0
        q_recall_10 = 0
        for idx, l in enumerate(label):
            if l >= min_true_label:
                if idx < 4:
                    q_recall_4 += 1
                if idx < 10:
                    q_recall_10 += 1
                q_pos_cnt += 1
                has_ans_q.add(q)

        pos_qp_amount_4 = min(q_pos_cnt, 4)
        pos_qp_amount_10 = min(q_pos_cnt, 10)
        if q_pos_cnt > 0:
            recall_4 += (q_recall_4 * 1.0 / pos_qp_amount_4)
            recall_10 += (q_recall_10 * 1.0 / pos_qp_amount_10)
            #print q_recall_4, pos_qp_amount_4, q_recall_4 * 1.0 / pos_qp_amount_4

    has_ans_rate = len(has_ans_q) * 1.0 / len(q_infos)
    recall_4 = recall_4 * 100.0 / len(has_ans_q)
    recall_10 = recall_10 * 100.0 / len(has_ans_q)
#    print("qp-Recall@4/10: %.2f/%.2f" % (recall_4, recall_10))
    qp_metrics = [recall_4, recall_10]
    return qp_metrics

def print_res(qu_answer, notq_qu_answer, pos_label):
    """
        print all metric results
    """
    qu_q_infos = read_qu_answer(qu_answer)
    q_metrics = cal_recall_q(qu_q_infos, pos_label)
    qp_metrics = cal_recall_qp(qu_q_infos, pos_label)

    notq_q_infos = read_qu_answer(notq_qu_answer)
    notq_q_metrics = cal_recall_q(notq_q_infos, pos_label)
    notq_qp_metriccs = cal_recall_qp(notq_q_infos, pos_label)

    print("NDCG@4/10:      {:.2f}/{:.2f}\t{:.2f}/{:.2f}".format(
        q_metrics[0], q_metrics[1], notq_q_metrics[0], notq_q_metrics[1]))
    print("MRR@4/10:       {:.2f}/{:.2f}\t{:.2f}/{:.2f}".format(
        q_metrics[2], q_metrics[3], notq_q_metrics[2], notq_q_metrics[3]))
    print("q-Recall@4/10:  {:.2f}/{:.2f}\t{:.2f}/{:.2f}".format(
        q_metrics[4], q_metrics[5], notq_q_metrics[4], notq_q_metrics[5]))
    print("qp-Recall@4/10: {:.2f}/{:.2f}\t{:.2f}/{:.2f}".format(
        qp_metrics[0], qp_metrics[1], notq_qp_metriccs[0], notq_qp_metriccs[1]))

def main(path_to_test, path_to_score, type):

    # aurora eval set
    if type[0] == 'a':
        qu_answer, notq_qu_answer = merge_text_and_score(path_to_test, path_to_score)

        print "--- Aurora  --  whole_set  --  t_not_q_subset -- [pos_2]"
        print_res(qu_answer, notq_qu_answer, 2)
        print "--- Aurora  --  whole_set  --  t_not_q_subset -- [pos_12]"
        print_res(qu_answer, notq_qu_answer, 1)
        print ""

    elif type[0] == 'r':
        qu_answer, notq_qu_answer = select_url_best_para(path_to_test, path_to_score)

        print "---   RAS   --  whole_set  --  t_not_q_subset -- [pos_345]"
        print_res(qu_answer, notq_qu_answer, 3)
        print ""

if __name__ == "__main__":
    if len(sys.argv) == 4:
        path_to_test = sys.argv[1]
        cur_path = os.path.split(os.path.realpath(__file__))[0]
        path_to_test = os.path.join(cur_path, path_to_test)
        path_to_score = sys.argv[2]
        type = sys.argv[3]
        main(path_to_test, path_to_score, type)

