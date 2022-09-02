import sys
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

pred_file = sys.argv[1]
task = sys.argv[2]
pre_thre = float(sys.argv[3])

cur_path = os.path.split(os.path.realpath(__file__))[0]
gold_file = os.path.join(cur_path, task)

def get_labels(gold_file):
    labels = []
    with open(gold_file) as inp1:
        for gold in inp1:
            label = int(gold.strip().split('\t')[3])
            label = 1 if label >= 1 else 0
            labels.append(label)
    return labels

def get_scores(pred_file):
    scores = []
    with open(pred_file) as inp:
        for line in inp:
            line = line.strip().split('\t')
            score = float(line[0])
            scores.append(score)
    return scores

def top1_label_and_score(labels, predict_score):
    """
    top1 score for each query
    """
    qid_scores_labels = {}
    for i, qid in enumerate(qid_list):
        if qid not in qid_scores_labels:
            qid_scores_labels[qid] = []
        qid_scores_labels[qid].append([predict_score[i], labels[i]])
        
    top1_labels = []
    top1_scores = []
    has_answer = []
    for qid, infos in qid_scores_labels.items():
        infos = sorted(infos, key=lambda x:x[0], reverse=True)
        top1_scores.append(infos[0][0])
        top1_labels.append(infos[0][1])
        s, l = zip(*infos)
        if sum(l) > 0:
            has_answer.append(1)
        else:
            has_answer.append(0)
    return top1_labels, top1_scores, has_answer

def print_pr_top1(precisions, recalls, thresholds):
    
    #print "threshold\tprecision\trecall\tF1" 
    max_f1 = 0
    max_i = -1
    for i, pre in enumerate(precisions):
        if pre > pre_thre:
            f1 = 2 * pre * recalls[i] / (pre + recalls[i])  
            if f1 > max_f1:
                max_f1 = f1
                max_i = i
    if max_i == -1:
        print("pre_ther is high")
    else:
        #print "%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s" % ("dev_" + str(int(pre_thre*100)), pred_file, round(thresholds[max_i], 6), round(precisions[max_i] * 100, 2),  round(recalls[max_i] * 100, 2), round(max_f1 * 100, 2))
        print("dev_" + str(int(pre_thre*100)) + "\t" +  pred_file + "\t" + str(round(thresholds[max_i], 6)) + "\t" + str(round(precisions[max_i] * 100, 2)) + "\t" + str(round(recalls[max_i] * 100, 2)) + "\t" + str(round(max_f1 * 100, 2)))
if __name__ == '__main__':

    labels = get_labels(gold_file)
    scores = get_scores(pred_file)
    if len(labels) != len(scores):
        print("labels not scores")

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    print_pr_top1(precisions, recalls, thresholds)
