#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import math
from collections import defaultdict
from six import moves
from six.moves import range
import sys
import sklearn.utils
_EPS = np.finfo(np.float64).eps
range = moves.range

_LOG2 = math.log(2.0)
input_file ='/mnt/du/wangwenhua/0-alading/2-model/from_authority/result/sigle_tower_v3_margin_rank_loss_0.1_0.1/relevance.test.2.final.qid.score.label'
input_file ='music_annotation_new_url_sort_test_qtp_train.score.894_res2'
# input_file = '/mnt/du/wangwenhua/0-alading/biji_data/ltr/v10/test_qid.score.label.v10'
# input_file = '/mnt/du/wangwenhua/0-alading/biji_data/ltr/baseline/test_qid.score.label.baseline'

def _identity_gain(x):
    return x


def _exp2_gain(x):
    return math.exp(x * _LOG2) - 1.0



def get_sorted_y_positions(y, y_pred, check=True):
    if check:
        y = sklearn.utils.validation.column_or_1d(y)
        y_pred = sklearn.utils.validation.column_or_1d(y_pred)
        sklearn.utils.validation.check_consistent_length(y, y_pred)
    return np.lexsort((y, -y_pred))


def get_sorted_y(y, y_pred, check=True):
    """Returns a copy of `y` sorted by position in `y_pred`.
    Parameters
    ----------
    y : array_like of shape = [n_samples_in_query]
        List of sample scores for a query.
    y_pred : array_like of shape = [n_samples_in_query]
        List of predicted scores for a query.
    Returns
    -------
    y_sorted : array_like of shape = [n_samples_in_query]
        Copy of `y` sorted by descending order of `y_pred`.
        Ties are broken in ascending order of `y`.
    """
    #print y
    #print get_sorted_y_positions(y, y_pred, check=check)
    y = np.array(y)
    return y[get_sorted_y_positions(y, y_pred, check=check)]

def get_gain_fn(name, **args):
    """Returns a gain callable corresponding to the provided gain name.
    Parameters
    ----------
    name : {'identity', 'exp2'}
        Name of the gain to return.
        - identity: ``lambda x : x``
        - exp2: ``lambda x : (2.0 ** x) - 1.0``
    Returns
    -------
    gain_fn : callable
        Callable that returns the gain of target values.
    """
    if name == 'identity':
        return _identity_gain
    elif name == 'exp2':
        return _exp2_gain
    raise ValueError(name + ' is not a valid gain type')


class Metric(object):
    """Base LTR metric class.
    Subclasses must override evaluate() and can optionally override various
    other methods.
    """
    def evaluate(self, qid, targets):
        """Evaluates the metric on a ranked list of targets.
        Parameters
        ----------
        qid : object
            Query id. Guaranteed to be a hashable type s.t.
            ``sorted(targets1) == sorted(targets2)`` iff ``qid1 == qid2``.
        targets : array_like of shape = [n_targets]
            List of targets for the query, in order of predicted score.
        Returns
        -------
        float
            Value of the metric on the provided list of targets.
        """
        raise NotImplementedError()

    def calc_swap_deltas(self, qid, targets):
        """Returns an upper triangular matrix.
        Each (i, j) contains the change in the metric from swapping
        targets[i, j].
        Parameters
        ----------
        qid : object
            See `evaluate`.
        targets : array_like of shape = [n_targets]
            See `evaluate`.
        Returns
        -------
        deltas = array_like of shape = [n_targets, n_targets]
            Upper triangular matrix, where ``deltas[i, j]`` is the change in
            the metric from swapping ``targets[i]`` with ``targets[j]``.
        """
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        original = self.evaluate(qid, targets)
        max_k = self.max_k()
        if max_k is None or n_targets < max_k:
            max_k = n_targets

        for i in range(max_k):
            for j in range(i + 1, n_targets):
                tmp = targets[i]
                targets[i] = targets[j]
                targets[j] = tmp
                deltas[i, j] = self.evaluate(qid, targets) - original
                tmp = targets[i]
                targets[i] = targets[j]
                targets[j] = tmp

        return deltas

    def max_k(self):
        """Returns a cutoff value for the metric.
        Returns
        -------
        k : int or None
            Value for which ``swap_delta()[i, j] == 0 for all i, j >= k``.
            None if no such value.
        """
        return None

    def evaluate_preds(self, qid, targets, preds):
        """Evaluates the metric on a ranked list of targets.
        Parameters
        ----------
        qid : object
            See `evaluate`.
        targets : array_like of shape = [n_targets]
            See `evaluate`.
        preds : array_like of shape = [n_targets]
            List of predicted scores corresponding to the targets. The
            `targets` array will be sorted by these predictions before
            evaluation.
        Returns
        -------
        float
            Value of the metric on the provided list of targets and
            predictions.
        """
        return self.evaluate(qid, get_sorted_y(targets, preds))

    def calc_random_ev(self, qid, targets):
        """Calculates the expectied value of the metric on randomized targets.
        This implementation just averages the metric over 100 shuffles.
        Parameters
        ----------
        qid : object
            See `evaluate`.
        targets : array_like of shape = [n_targets]
            See `evaluate`.
        Returns
        -------
        float
            Expected value of the metric from random ordering of targets.
        """
        targets = np.copy(targets)
        scores = []
        for _ in range(100):
            np.random.shuffle(targets)
            scores.append(self.evaluate(qid, targets))
        return np.mean(scores)

    def calc_mean(self, qids, targets, preds):
        """Calculates the mean of the metric among the provided predictions.
        Parameters
        ----------
        qids : array_like of shape = [n_targets]
            List of query ids. They must be grouped contiguously
            (i.e. ``pyltr.util.group.check_qids`` must pass).
        targets : array_like of shape = [n_targets]
            List of targets.
        preds : array_like of shape = [n_targets]
            List of predicted scores corresponding to the targets.
        Returns
        -------
        float
            Mean of the metric over provided query groups.
        """
        util.check_qids(qids)
        query_groups = util.get_groups(qids)
        return np.mean([self.evaluate_preds(qid, targets[a:b], preds[a:b])
                        for qid, a, b in query_groups])

    def calc_mean_random(self, qids, targets):
        """Calculates the EV of the mean of the metric with random ranking.
        Parameters
        ----------
        qids : array_like of shape = [n_targets]
            See `calc_mean`.
        targets : array_like of shape = [n_targets]
            See `calc_mean`.
        Returns
        -------
        float
            Expected value of the mean of the metric on random orderings of the
            provided query groups.
        """
        util.check_qids(qids)
        query_groups = util.get_groups(qids)
        return np.mean([self.calc_random_ev(qid, targets[a:b])
                        for qid, a, b in query_groups])


class DCG(Metric):
    def __init__(self, k=10, gain_type='exp2'):
        super(DCG, self).__init__()
        self.k = k
        self.gain_type = gain_type
        self._gain_fn = get_gain_fn(gain_type)
        self._discounts = self._make_discounts(256)

    def evaluate(self, qid, targets):
        return sum(self._gain_fn(t) * self._get_discount(i)
                   for i, t in enumerate(targets) if i < self.k)

    def calc_swap_deltas(self, qid, targets, coeff=1.0):
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))

        for i in range(min(n_targets, self.k)):
            for j in range(i + 1, n_targets):
                deltas[i, j] = coeff * \
                    (self._gain_fn(targets[i]) - self._gain_fn(targets[j])) * \
                    (self._get_discount(j) - self._get_discount(i))

        return deltas

    def max_k(self):
        return self.k

    def calc_random_ev(self, qid, targets):
        total_gains = sum(self._gain_fn(t) for t in targets)
        total_discounts = sum(self._get_discount(i)
                              for i in range(min(self.k, len(targets))))
        return total_gains * total_discounts / len(targets)

    @classmethod
    def _make_discounts(self, n):
        return np.array([1.0 / np.log2(i + 2.0) for i in range(n)])

    def _get_discount(self, i):
        if i >= self.k:
            return 0.0
        while i >= len(self._discounts):
            self._grow_discounts()
        return self._discounts[i]

    def _grow_discounts(self):
        self._discounts = self._make_discounts(len(self._discounts) * 2)


class NDCG(Metric):
    def __init__(self, k=10, gain_type='exp2'):
        super(NDCG, self).__init__()
        self.k = k
        self.gain_type = gain_type
        self._dcg = DCG(k=k, gain_type=gain_type)
        self._ideals = {}

    def evaluate(self, qid, targets):
        return (self._dcg.evaluate(qid, targets) /
                max(_EPS, self._get_ideal(qid, targets)))

    def calc_swap_deltas(self, qid, targets):
        ideal = self._get_ideal(qid, targets)
        if ideal < _EPS:
            return np.zeros((len(targets), len(targets)))
        return self._dcg.calc_swap_deltas(
            qid, targets, coeff=1.0 / ideal)

    def max_k(self):
        return self.k

    def calc_random_ev(self, qid, targets):
        return (self._dcg.calc_random_ev(qid, targets) /
                max(_EPS, self._get_ideal(qid, targets)))

    def _get_ideal(self, qid, targets):
        ideal = self._ideals.get(qid)
        if ideal is not None:
            return ideal
        sorted_targets = np.sort(targets)[::-1]
        ideal = self._dcg.evaluate(qid, sorted_targets)
        self._ideals[qid] = ideal
        return ideal

def cal_ndcg(qids, labels, preds):
    ndcg = NDCG(2)
    last_qid = -1
    cur_lables = []
    cur_preds = []
    sum_ndcg = 0
    q_count = 0
    for qid, label, pred in zip(qids, labels, preds):
        if (qid != last_qid):
            if last_qid != -1:
                cur_ndcg = ndcg.evaluate_preds(qid, cur_lables, cur_preds)
                sum_ndcg += cur_ndcg
                q_count += 1
            last_qid = qid
            cur_lables = [label]
            cur_preds = [pred]
        else:
            cur_lables.append(label)
            cur_preds.append(pred)
    avg_ndcg= (sum_ndcg * 1.0 / q_count) * 100
    return avg_ndcg

def recall_k(qids, labels, preds):
    recall_k_dict = dict()
    query_samples_map = defaultdict(list)
    for qid, label, score in zip(qids, labels, preds):
        query_samples_map[qid].append({'qid': qid, 'label': int(label), 'score': float(score)})
    index_list = []
    label_list = []
    for qid in list(query_samples_map.keys()):
        group = query_samples_map[qid]
        group = sorted(group, key=lambda x: x['score'], reverse=True)
        target_label = -1
        for index in range(len(group)):
            target_label = max(target_label, group[index]['label'])

        target_index = -1
        labels = list()
        for index, sample in enumerate(group):
            if target_label !=0 and sample['label'] == target_label and target_index==-1:
                target_index = index
                break
            labels.append(sample['label'])
        index_list.append(target_index)
        label_list.append(labels)

    has_ans_q = sum([1 for index in index_list if index != -1])
    for topk in [1, 2, 4, 10]:
        acc = sum([1 for index in index_list if index<topk and index != -1])*100.0 / has_ans_q
        recall_k_dict[topk] = acc

    return recall_k_dict

def PNRatio(qid, label, pred):
    def update(qid, label, pred):
        if not (qid.shape[0] == label.shape[0] == pred.shape[0]):
            raise ValueError('dimention not match: qid[%s] label[%s], pred[%s]'
                             % (qid.shape, label.shape, pred.shape))
        qid = qid.reshape([-1]).tolist()
        label = label.reshape([-1]).tolist()
        pred = pred.reshape([-1]).tolist()
        assert len(qid) == len(label) == len(pred)
        for q, l, p in zip(qid, label, pred):
            if q not in saver:
                saver[q] = []
            saver[q].append((l, p))
        return saver
    def eval(saver):
        p = 0
        n = 0
        for qid, outputs in saver.items():
            for i in range(0, len(outputs)):
                l1, p_left = outputs[i]
                for j in range(i + 1, len(outputs)):
                    l2, p_right = outputs[j]
                    # print('......')
                    # print('{}\t{}'.format(l1, p_left))
                    # print('{}\t{}'.format(l2, p_right))
                    # print('......')
                    if l1 > l2:
                        if p_left > p_right:
                            p += 1
                        elif p_left < p_right:
                            n += 1
                    elif l1 < l2:
                        if p_left < p_right:
                            p += 1
                        elif p_left > p_right:
                            n += 1
        pn = p / n if n > 0 else 0.0
        return p, n, np.float32(pn)
    qid =np.array(qid)
    label =np.array(label)
    pred =np.array(pred)
    saver = {}
    saver = update(qid, label, pred)
    p, n, pnratio = eval(saver)
    return p, n, pnratio


qids = []
labels = []
scores = []
for line_index, line in enumerate(open(input_file).readlines()):
    if line_index != 0 and  line_index % 100000 == 0:
        print('dealing with input file - line: ', line_index)
    parts = line.split('\t')
    if len(parts) != 3:
        continue
    qid, score, label = [x.strip() for x in parts]
    qids.append(qid)
    labels.append(int(label))
    scores.append(float(score))
p, n, pnratio = PNRatio(qids, labels, scores)
print('query is {}, p is {}, n is {}, pnratio is {}'.format(len(set(qids)), p, n, pnratio))
print('-' * 50)
recall_k_dict = recall_k(qids, labels, scores)
recall_k_result_info = ''
for topk_, values_ in recall_k_dict.items():
    recall_k_result_info += 'Recall@{}:{:.3f} '.format(topk_, values_)
print(recall_k_result_info)
# ndcg@2
avg_ndcg = cal_ndcg(qids, labels, scores)
print('-' * 50)
print("avg ndcg={}".format(avg_ndcg))
print('-' * 50)

