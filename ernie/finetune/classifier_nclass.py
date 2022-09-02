#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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

from scipy.stats import pearsonr, spearmanr
from six.moves import xrange
import paddle.fluid as fluid
import paddle.fluid.layers as L

from model.ernie import ErnieModel

log = logging.getLogger(__name__)

def create_model(args,
                 pyreader_name,
                 ernie_config,
                 is_prediction=False,
                 task_name="",
                 is_classify=False,
                 is_regression=False,
                 ernie_version="1.0",
                 preset_batch_size=80,
                 margin=1.0,
                 ):
    if is_classify:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                    [-1, args.max_seq_len, 1], [-1, 1], [-1, 1]],
            dtypes=[
                'int64', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64'
            ],
            lod_levels=[0, 0, 0, 0, 0, 0, 0],
            name=task_name + "_" + pyreader_name,
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

    (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    cls_feats = ernie.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_one_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_one_b",
            initializer=fluid.initializer.Constant(0.)))
    probs = fluid.layers.sigmoid(logits)

    assert is_classify != is_regression, 'is_classify or is_regression must be true and only one of them can be true'
    if is_prediction:
        if is_classify:
            # probs = fluid.layers.softmax(logits)
            probs = logits
        else:
            probs = logits
        feed_targets_name = [
            src_ids.name, sent_ids.name, pos_ids.name, input_mask.name
        ]
        if ernie_version == "2.0":
            feed_targets_name += [task_ids.name]
        return pyreader, probs, feed_targets_name

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    if is_classify:
        
        qid1 = L.expand(qids, [1, preset_batch_size])
        qid2 = L.transpose(qid1, [1, 0])

        # L.Print(qids, message='qids', summarize=-1)
        # L.Print(qid1, message='qid1', summarize=-1)
        # L.Print(qid2, message='qid2', summarize=-1)

        logits1 = L.expand(logits, [1, preset_batch_size])
        logits2 = L.transpose(logits1, [1, 0])

        labels1 = L.expand(labels, [1, preset_batch_size])
        labels2 = L.transpose(labels1, [1, 0])

        pn_labels = (labels1 - labels2) * L.equal(qid1, qid2)
        zeros = L.zeros(shape=[preset_batch_size, preset_batch_size], dtype="int64")
        ones = L.ones(shape=[preset_batch_size, preset_batch_size], dtype="int64")
        

        def cal_hinge_loss(masked_pn_labels, pn_labels_all, margin=1.0):
            hinge_loss = masked_pn_labels * L.relu(L.cast(logits2 - logits1 + margin, "float32"))
            hinge_loss = L.reduce_sum(hinge_loss) / (L.reduce_sum(pn_labels_all) + 1e-8)
            return hinge_loss
        def cal_hinge_loss_martix(masked_pn_labels, pn_labels_all, margin_martix):
            hinge_loss = masked_pn_labels * L.relu(L.cast(logits2 - logits1 + margin_martix, "float32"))
            hinge_loss = L.reduce_sum(hinge_loss) / (L.reduce_sum(pn_labels_all) + 1e-8)
            return hinge_loss
            
        #max_score = 480.0 #设置最大档位分
        #pn_labels_all = L.cast(L.less_than(zeros, pn_labels), "float32")
        #pn_labels_ij = L.cast(L.less_than(zeros, pn_labels), "float32")
        #margin_ij = (pn_labels_ij * pn_labels) / max_score
        #loss_pair = cal_hinge_loss_martix(pn_labels_ij, pn_labels_all, margin_ij)
        
        ### 将pn_label_ij margin_ij 拆出 相关、原唱、热度、歌词四个纬度组pair
        
        margins = {
            'margin_review': 5,
            'margin_rel': 5,
            'margin_orgin': 3,
            'margin_hot': 2,
            'margin_liry': 1
        }
        pn_labels_all = L.cast(L.less_than(zeros, pn_labels), "float32")
        pn_labels_ij_no = L.cast(L.less_than(zeros, pn_labels) * L.less_than(ones * 100000, pn_labels), "float32")
        pn_labels_ij_10000 = L.cast(L.less_than(zeros, pn_labels) * L.less_than(ones * 10000, pn_labels), "float32")
        pn_labels_ij_review = pn_labels_ij_10000 - pn_labels_ij_no

        pn_labels_ij_rel = L.cast(L.less_than(zeros, pn_labels) * L.less_than(ones * 1000, pn_labels), "float32") - pn_labels_ij_10000 
        pn_labels_ij_orgin = L.cast(L.less_than(zeros, pn_labels) * L.less_than(ones * 100, pn_labels), "float32") - pn_labels_ij_rel - pn_labels_ij_10000
        pn_labels_ij_hot = L.cast(L.less_than(zeros, pn_labels) * L.less_than(ones * 10, pn_labels), "float32") - pn_labels_ij_rel - pn_labels_ij_orgin - pn_labels_ij_10000
        pn_labels_ij_liry = L.cast(L.less_than(zeros, pn_labels) * L.less_than(ones * 1, pn_labels), "float32") - pn_labels_ij_rel - pn_labels_ij_orgin - pn_labels_ij_hot - pn_labels_ij_10000

        loss_pair_review = cal_hinge_loss(pn_labels_ij_review, pn_labels_all, margins["margin_review"])

        loss_pair_rel = cal_hinge_loss(pn_labels_ij_rel, pn_labels_all, margins["margin_rel"])
        loss_pair_orgin = cal_hinge_loss(pn_labels_ij_orgin, pn_labels_all, margins["margin_orgin"])
        loss_pair_hot = cal_hinge_loss(pn_labels_ij_hot, pn_labels_all, margins["margin_hot"])
        loss_pair_liry = cal_hinge_loss(pn_labels_ij_liry, pn_labels_all, margins["margin_liry"])
        loss_pair = 10 * loss_pair_review + loss_pair_rel + loss_pair_orgin + loss_pair_hot + loss_pair_liry
        
        #L.Print(pn_labels_all, message='pn_labels_all', summarize=-1)
        #L.Print(pn_labels_ij_rel, message='pn_labels_ij_rel', summarize=-1)
        #L.Print(pn_labels_ij_orgin, message='pn_labels_ij_orgin', summarize=-1)
        #L.Print(pn_labels_ij_hot, message='pn_labels_ij_hot', summarize=-1)
        #L.Print(pn_labels_ij_liry, message='pn_labels_ij_liry', summarize=-1)

        # pointwise loss
        ones = L.ones(shape=[preset_batch_size, 1], dtype="int64")

        pointwise_label_10000 = L.cast(L.less_than(ones * 10000, labels), "float32")
        pointwise_label_all = L.cast(L.less_than(ones * 0, labels), "float32") - pointwise_label_10000
        pointwise_mask_all = pointwise_label_all + L.cast(L.equal(labels, ones * 0), "float32")
        loss_point = pointwise_mask_all * fluid.layers.sigmoid_cross_entropy_with_logits(logits, L.cast(pointwise_label_all, "float32")) ## 0-34pointloss
        loss_point = L.reduce_sum(loss_point) / (L.reduce_sum(pointwise_mask_all) + 1e-10)

        #L.Print(pointwise_label_all, message='pointwise_label_all', summarize=-1)
        #L.Print(pointwise_mask_all, message='pointwise_mask_all', summarize=-1)
        #L.Print(logits, message='logits', summarize=-1)
        #L.Print(loss_point, message='loss_point', summarize=-1)
        #L.Print(loss_pair, message='loss_pair', summarize=-1)
        #L.Print(loss_pair_review, message='loss_pair_review', summarize=-1)
        #L.Print(loss_pair_rel, message='loss_pair_rel', summarize=-1)
        #L.Print(loss_pair_orgin, message='loss_pair_orgin', summarize=-1)
        #L.Print(loss_pair_hot, message='loss_pair_hot', summarize=-1)
        #L.Print(loss_pair_liry, message='loss_pair_liry', summarize=-1)
        if args.only_pointwise:
            loss = loss_point
        elif args.only_pairwise:
            loss = loss_pair * 0.1
        else:
            loss = (loss_point + 1.0 * loss_pair)

        #L.Print(loss_point, message='loss_point', summarize=-1)
        #L.Print(loss_pair, message='loss_pair', summarize=-1)
        #L.Print(loss, message='loss', summarize=-1)
        #L.Print(loss_pair_3_1, message='loss_pair_3_1', summarize=-1)
        #L.Print(loss_pair_3_0, message='loss_paircc_3_0', summarize=-1)
        #L.Print(loss_pair_2_1, message='loss_pair_2_1', summarize=-1)
        #L.Print(loss_pair_2_0, message='loss_pair_2_0', summarize=-1)
        #L.Print(loss_pair_1_0, message='loss_pair_1_0', summarize=-1)

        accuracy = fluid.layers.accuracy(
            input=probs, label=L.cast(pointwise_label_all, "int64"), total=num_seqs)
        graph_vars = {
            "loss": loss,
            "probs": probs,
            "accuracy": accuracy,
            "labels": labels,
            "num_seqs": num_seqs,
            "qids": qids
        }
    elif is_regression:
        cost = fluid.layers.square_error_cost(input=logits, label=labels)
        loss = fluid.layers.mean(x=cost)
        graph_vars = {
            "loss": loss,
            "probs": probs,
            "labels": labels,
            "num_seqs": num_seqs,
            "qids": qids
        }
    else:
        raise ValueError(
            'unsupported fine tune mode. only supported classify/regression')

    return pyreader, graph_vars


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
        graph_vars["num_seqs"].name, graph_vars["qids"].name
    ]
    while True:
        try:
            if use_multi_gpu_test:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                    fetch_list=fetch_list)
            else:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            batch_score = np_probs[:, 1].reshape(-1).tolist()
            scores.extend(batch_score)
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds)
            total_label_pos_num += np.sum(np_labels)
            total_pred_pos_num += np.sum(np_preds)
            total_correct_num += np.sum(np.dot(np_preds, np_labels))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    return
    time_end = time.time()
    cost = total_cost / total_num_seqs
    elapsed_time = time_end - time_begin

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


def predict(exe,
            test_program,
            test_pyreader,
            graph_vars,
            dev_count=1,
            is_classify=False,
            is_regression=False,
            score_f='test.score'
            ):
    test_pyreader.start()
    qids, scores, probs = [], [], []
    preds = []

    fetch_list = [graph_vars["probs"].name, graph_vars["qids"].name]
    output = open(score_f, 'w')

    while True:
        try:
            if dev_count == 1:
                np_probs, np_qids = exe.run(program=test_program,
                                            fetch_list=fetch_list)
            else:
                np_probs, np_qids = exe.run(fetch_list=fetch_list)

            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            if is_classify:
                np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
                preds.extend(np_preds)
                batch_score = np_probs[:].reshape(-1).tolist()
                for score in batch_score:
                    output.write('{}\n'.format(score))
                # print(np_preds)
                # print('probs:', np_probs)
                # print('batch score:', batch_score)

            elif is_regression:
                preds.extend(np_probs.reshape(-1))
                batch_score = np_probs[:].reshape(-1).tolist()
                for score in batch_score:
                    output.write('{}\n'.format(score))
                # print('regression probs:', np_probs)
                # print('batch score:', batch_score)

            probs.append(np_probs)

        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    probs = np.concatenate(probs, axis=0).reshape([len(preds), -1])
    output.close()

    return qids, preds, probs
