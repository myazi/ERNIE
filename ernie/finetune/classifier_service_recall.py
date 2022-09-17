#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import time
import logging
import numpy as np

#import faiss
from scipy.stats import pearsonr, spearmanr
from six.moves import xrange
import paddle.fluid as fluid

from model.ernie import ErnieModel

log = logging.getLogger(__name__)

def create_model(args,
                 pyreader_name,
                 ernie_config,
                 batch_size=16,
                 is_prediction=False,
                 task_name="",
                 is_classify=False,
                 is_regression=False,
                 save_part='all'):
    if is_classify:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
                [batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
                [batch_size, args.q_max_seq_len, 1],
                [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
                [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
                [batch_size, args.p_max_seq_len, 1],
                [batch_size, 1], [batch_size, 1]],
        dtypes=['int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0],
        name=pyreader_name,
        use_double_buffer=True)
    elif is_regression:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1], [-1, 1], [-1, 1]],
            dtypes=[
                'int64', 'int64', 'int64', 'int64', 'float32', 'float32',
                'int64'
            ],
            lod_levels=[0, 0, 0, 0, 0, 0, 0],
            name=task_name + "_" + pyreader_name,
            use_double_buffer=True)

    (src_ids_q, sent_ids_q, pos_ids_q, task_ids_q, input_mask_q,
     src_ids_p, sent_ids_p, pos_ids_p, task_ids_p, input_mask_p,
     labels, qids) = fluid.layers.read_file(pyreader)
    """
    fluid.layers.Print(src_ids_q, message='src_ids_q')
    fluid.layers.Print(sent_ids_q, message='sent_ids_q')
    fluid.layers.Print(pos_ids_q, message='pos_ids_q')
    fluid.layers.Print(task_ids_q, message='task_ids_q')
    fluid.layers.Print(input_mask_q, message='input_mask_q')
    fluid.layers.Print(src_ids_p, message='src_ids_p')
    fluid.layers.Print(sent_ids_p, message='sent_ids_p')
    fluid.layers.Print(pos_ids_p, message='pos_ids_p')
    fluid.layers.Print(task_ids_p, message='task_ids_p')
    fluid.layers.Print(input_mask_p, message='input_mask_p')
    """

    ernie_q = ErnieModel(
        model_name='query_',
        src_ids=src_ids_q,
        position_ids=pos_ids_q,
        sentence_ids=sent_ids_q,
        task_ids=task_ids_q,
        input_mask=input_mask_q,
        config=ernie_config,
        use_fp16=args.use_fp16)
    ## pos para
    ernie_p = ErnieModel(
        model_name='titlepara_',
        src_ids=src_ids_p,
        position_ids=pos_ids_p,
        sentence_ids=sent_ids_p,
        task_ids=task_ids_p,
        input_mask=input_mask_p,
        config=ernie_config,
        use_fp16=args.use_fp16)

    q_cls_feats = ernie_q.get_pooled_output_recall("query_") #recall的pool与rank的不同
    p_cls_feats = ernie_p.get_pooled_output_recall("titlepara_")
    #p_cls_feats = fluid.layers.concat([pos_cls_feats, neg_cls_feats], axis=0)
    #src_ids_p = fluid.layers.Print(src_ids_p, message='p: ')
    #p_cls_feats = fluid.layers.Print(p_cls_feats, message='p: ')

#    q_cls_feats = fluid.layers.dropout(
#        x=q_cls_feats,
#        dropout_prob=0.1,
#        dropout_implementation="upscale_in_train")
#    q_rep = fluid.layers.fc(
#        input=q_cls_feats,
#        size=128,
#        param_attr=fluid.ParamAttr(
#            name="q_cls_out_w",
#            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
#        bias_attr=fluid.ParamAttr(
#            name="q_cls_out_b", initializer=fluid.initializer.Constant(0.)))
#    ## for init sample neg model
#    p_cls_feats = fluid.layers.dropout(
#        x=p_cls_feats,
#        dropout_prob=0.1,
#        dropout_implementation="upscale_in_train")
#    p_rep = fluid.layers.fc(
#        input=p_cls_feats,
#        size=128,
#        param_attr=fluid.ParamAttr(
#            name="p_cls_out_w",
#            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
#        bias_attr=fluid.ParamAttr(
#            name="p_cls_out_b", initializer=fluid.initializer.Constant(0.)))

    #multiply
    logits = fluid.layers.elementwise_mul(q_cls_feats, p_cls_feats)
    probs = fluid.layers.reduce_sum(logits, dim=-1)
    if is_prediction:
        if save_part == 'query':
            feed_targets_name = [
                src_ids_q.name, sent_ids_q.name, pos_ids_q.name, task_ids_q.name, input_mask_q.name
            ]
            return pyreader, feed_targets_name, q_cls_feats
        elif save_part == 'para':
            feed_targets_name = [
                src_ids_p.name, sent_ids_p.name, pos_ids_p.name, task_ids_p.name, input_mask_p.name
            ]
            return pyreader, feed_targets_name, p_cls_feats
        elif save_part == 'all':
            feed_targets_name = [
                src_ids_q.name, sent_ids_q.name, pos_ids_q.name, task_ids_q.name, input_mask_q.name,
                src_ids_p.name, sent_ids_p.name, pos_ids_p.name, task_ids_p.name, input_mask_p.name
            ]
            return pyreader, feed_targets_name, [q_cls_feats, p_cls_feats, probs]
    #fluid.layers.Print(probs, message='probs: ')
    #logits2 = fluid.layers.elementwise_mul(x=q_rep, y=p_rep)
    #fluid.layers.Print(logits2, message='logits2: ')
    #probs2 = fluid.layers.reduce_sum(logits, dim=-1)
    #fluid.layers.Print(probs2, message='probs2: ')

    matrix_labels = fluid.layers.eye(batch_size, batch_size, dtype='float32')
    matrix_labels.stop_gradient=True

    #print('DEBUG:\tstart loss')
    ce_loss, _ = fluid.layers.softmax_with_cross_entropy(
           logits=logits, label=matrix_labels, soft_label=True, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)
    #print('DEBUG:\tloss done')

    matrix_labels = fluid.layers.argmax(matrix_labels, axis=-1)
    matrix_labels = fluid.layers.reshape(x=matrix_labels, shape=[batch_size, 1])
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=matrix_labels, total=num_seqs)

    if is_classify:
        #ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        #    logits=logits, label=labels, return_softmax=True)
        #loss = fluid.layers.mean(x=ce_loss)
        #accuracy = fluid.layers.accuracy(
        #    input=probs, label=labels, total=num_seqs)
        graph_vars = {
            "loss": loss,
            "probs": probs,
            "accuracy": accuracy,
            "labels": labels,
            "num_seqs": num_seqs,
            "qids": qids,
            "q_rep": q_cls_feats,
            "p_rep": p_cls_feats
        }
    elif is_regression:
        #cost = fluid.layers.square_error_cost(input=logits, label=labels)
        #loss = fluid.layers.mean(x=cost)
        graph_vars = {
            "loss": loss,
            "probs": logits,
            "labels": labels,
            "num_seqs": num_seqs,
            "qids": qids
        }
    else:
        raise ValueError(
            'unsupported fine tune mode. only supported classify/regression')

    return pyreader, graph_vars, probs, q_cls_feats, p_cls_feats


def evaluate_mrr(preds):
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum


def evaluate_map(preds):
    def singe_map(st, en):
        total_p = 0.0
        correct_num = 0.0
        for index in xrange(st, en):
            if int(preds[index][2]) != 0:
                correct_num += 1
                total_p += correct_num / (index - st + 1)
        if int(correct_num) == 0:
            return 0.0
        return total_p / correct_num

    last_qid = None
    total_map = 0.0
    qnum = 0.0
    st = 0
    for i in xrange(len(preds)):
        qid = preds[i][0]
        if qid != last_qid:
            qnum += 1
            if last_qid != None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum


def evaluate_classify(exe,
                      test_program,
                      test_pyreader,
                      graph_vars,
                      eval_phase,
                      use_multi_gpu_test=False,
                      metric='simple_accuracy',
                      is_classify=False,
                      is_regression=False):
    train_fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["num_seqs"].name
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0]), "accuracy": np.mean(outputs[1])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    test_pyreader.start()
    total_cost, total_acc, total_num_seqs, total_label_pos_num, total_pred_pos_num, total_correct_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    qids, labels, scores, preds = [], [], [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name,
        graph_vars["q_rep"].name, graph_vars["p_rep"].name
    ]
    #emb_file = open('emb_qp', 'w')
    while True:
        try:
            if use_multi_gpu_test:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids, q_rep, p_rep = exe.run(
                    fetch_list=fetch_list)
            else:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids, q_rep, p_rep = exe.run(
                    program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            batch_scores = np.diag(np_probs).reshape(-1).tolist()
            scores.extend(batch_scores)
            #for item in list(zip(q_rep, p_rep, batch_scores)):
            #    _left = ' '.join([str(each) for each in item[0]])
            #    _right = ' '.join([str(each) for each in item[1]])
            #    emb_file.write(_left + '\t' + _right + '\t' + str(item[2]) + '\n')
            #scores.extend(np_probs[:, 1].reshape(-1).tolist())
            #np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            #preds.extend(np_preds)
            #total_label_pos_num += np.sum(np_labels)
            #total_pred_pos_num += np.sum(np_preds)
            #total_correct_num += np.sum(np.dot(np_preds, np_labels))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    #for score in np_preds:
    #    print (score)
    #print ('---------------------')
    #time_end = time.time()
    #cost = total_cost / total_num_seqs
    #elapsed_time = time_end - time_begin
    #emb_file.close()
    return None
    evaluate_info = ""
    if metric == 'acc_and_f1':
        ret = acc_and_f1(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, ave_acc: %f, f1: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret['acc'], ret['f1'], total_num_seqs, elapsed_time)
    elif metric == 'matthews_corrcoef':
        ret = matthews_corrcoef(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, matthews_corrcoef: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret, total_num_seqs, elapsed_time)
    elif metric == 'pearson_and_spearman':
        ret = pearson_and_spearman(scores, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, pearson:%f, spearman:%f, corr:%f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret['pearson'], ret['spearman'], ret['corr'], total_num_seqs, elapsed_time)
    elif metric == 'simple_accuracy':
        ret = simple_accuracy(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, acc:%f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret, total_num_seqs, elapsed_time)
    elif metric == "acc_and_f1_and_mrr":
        ret_a = acc_and_f1(preds, labels)
        preds = sorted(
            zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
        ret_b = evaluate_mrr(preds)
        evaluate_info = "[%s evaluation] ave loss: %f, acc: %f, f1: %f, mrr: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret_a['acc'], ret_a['f1'], ret_b, total_num_seqs, elapsed_time)
    else:
        raise ValueError('unsupported metric {}'.format(metric))
    return evaluate_info


def evaluate_regression(exe,
                        test_program,
                        test_pyreader,
                        graph_vars,
                        eval_phase,
                        use_multi_gpu_test=False,
                        metric='pearson_and_spearman'):

    if eval_phase == "train":
        train_fetch_list = [graph_vars["loss"].name]
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[1][0])
        return ret

    test_pyreader.start()
    total_cost, total_num_seqs = 0.0, 0.0
    qids, labels, scores = [], [], []

    fetch_list = [
        graph_vars["loss"].name, graph_vars["probs"].name,
        graph_vars["labels"].name, graph_vars["qids"].name
    ]

    time_begin = time.time()
    while True:
        try:
            if use_multi_gpu_test:
                np_loss, np_probs, np_labels, np_qids = exe.run(
                    fetch_list=fetch_list)
            else:
                np_loss, np_probs, np_labels, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            scores.extend(np_probs.reshape(-1).tolist())
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    elapsed_time = time_end - time_begin

    if metric == 'pearson_and_spearman':
        ret = pearson_and_spearman(scores, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, pearson:%f, spearman:%f, corr:%f, elapsed time: %f s" \
            % (eval_phase, 0.0, ret['pearson'], ret['spearmanr'], ret['corr'], elapsed_time)
    else:
        raise ValueError('unsupported metric {}'.format(metric))

    return evaluate_info


def evaluate(exe,
             test_program,
             test_pyreader,
             graph_vars,
             eval_phase,
             use_multi_gpu_test=False,
             metric='simple_accuracy',
             is_classify=False,
             is_regression=False):

    if is_classify:
        return evaluate_classify(
            exe,
            test_program,
            test_pyreader,
            graph_vars,
            eval_phase,
            use_multi_gpu_test=use_multi_gpu_test,
            metric=metric)
    else:
        return evaluate_regression(
            exe,
            test_program,
            test_pyreader,
            graph_vars,
            eval_phase,
            use_multi_gpu_test=use_multi_gpu_test,
            metric=metric)


def matthews_corrcoef(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc


def f1_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return f1


def pearson_and_spearman(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def acc_and_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()

def build_engine(para_emb_list, dim):
    index = faiss.IndexFlatIP(dim)
    # add paragraph embedding
    p_emb_matrix = np.asarray(para_emb_list)
    index.add(p_emb_matrix.astype('float32'))
    #print ("insert done", file=sys.stderr)
    return index

def predict(args,
            exe,
            test_program,
            test_pyreader,
            graph_vars,
            dev_count=1,
            is_classify=False,
            is_regression=False,
            output_item=0,
            output_file_name='emb'):

    test_pyreader.start()
    preds = []

    fetch_list = [graph_vars["q_rep"].name, graph_vars["p_rep"].name,]

    if output_item == 0:
        emb_file = open(output_file_name, 'w')
    else:
        para_embs = []
        index_file = output_file_name

    while True:
        try:
            q_rep, p_rep = exe.run(program=test_program,
                                            fetch_list=fetch_list)

            if output_item == 0:
                for item in q_rep:
                    _left = ' '.join([str(each) for each in item])
                    emb_file.write(_left + '\n')
            elif output_item == 1:
                for item in p_rep:
                    para_embs.append(np.array(item, dtype='float32'))

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    if output_item == 0:
        emb_file.close()

    elif output_item == 1:
        print("predict para embs cnt: %s" % len(para_embs))
        #para_embs = para_embs[:args.test_data_cnt]
        print("cut para embs cnt: %s" % len(para_embs))
        #engine = build_engine(para_embs, 768)
        #faiss.write_index(engine, index_file)
