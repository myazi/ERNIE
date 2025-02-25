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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import os
import json
import random
import logging
import numpy as np
import six
from io import open
from collections import namedtuple

import tokenization
from batching import pad_batch_data


log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def csv_reader(fd, delimiter='\t'):
    def gen():
        for i in fd:
            yield i.rstrip('\n').split(delimiter)
    return gen()


class BaseReader(object):
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 q_max_seq_len=128,
                 p_max_seq_len=512,
                 do_lower_case=True,
                 infer_qp="",
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.infer_qp = infer_qp
        self.q_max_seq_len = q_max_seq_len
        self.p_max_seq_len = p_max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, batch_size=16, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f)
            headers = next(reader)
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, q_max_seq_length, p_max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""
        # query
        query = tokenization.convert_to_unicode(example.query)
        tokens_query = tokenizer.tokenize(query)
        self._truncate_seq_pair([], tokens_query, q_max_seq_length - 2)

        # title
        title = tokenization.convert_to_unicode(example.title)
        tokens_title = tokenizer.tokenize(title)
        # para
        para = tokenization.convert_to_unicode(example.para)
        tokens_para = tokenizer.tokenize(para)

        self._truncate_seq_pair(tokens_title, tokens_para, p_max_seq_length - 3)
 
        ## query
        tokens_q = []
        text_type_ids_q = []
        tokens_q.append("[CLS]")
        text_type_ids_q.append(0)
        for token in tokens_query:
            tokens_q.append(token)
            text_type_ids_q.append(0)
        tokens_q.append("[SEP]")
        text_type_ids_q.append(0)

        token_ids_q = tokenizer.convert_tokens_to_ids(tokens_q)
        position_ids_q = list(range(len(token_ids_q)))
        #f = open('tid', 'a')
        #for tid in range(len(token_ids_q)):
        #    f.write(str(token_ids_q[tid]) + '\t' + tokens_q[tid] + '\n')
            #f.write(str(token_ids_q[tid]) + ' ')
        #f.write('\t')

        ### para
        tokens_p = []
        text_type_ids_p = []
        tokens_p.append("[CLS]")
        text_type_ids_p.append(0)

        for token in tokens_title:
            tokens_p.append(token)
            text_type_ids_p.append(0)
        tokens_p.append("[SEP]")
        text_type_ids_p.append(0)

        for token in tokens_para:
            tokens_p.append(token)
            text_type_ids_p.append(1)
        tokens_p.append("[SEP]")
        text_type_ids_p.append(1)

        token_ids_p = tokenizer.convert_tokens_to_ids(tokens_p)
        position_ids_p = list(range(len(token_ids_p)))
        #for tid in range(len(token_ids_p)):
        #    f.write(str(token_ids_p[tid]) + '\t' + tokens_p[tid] + '\n')
            #f.write(str(token_ids_p[tid]) + ' ')
        #f.write('\n')
        #f.close()

        if self.is_inference:
            Record = namedtuple('Record',
            ['token_ids_q', 'text_type_ids_q', 'position_ids_q', \
             'token_ids_p', 'text_type_ids_p', 'position_ids_p'])
            record = Record(
                token_ids_q=token_ids_q,
                text_type_ids_q=text_type_ids_q,
                position_ids_q=position_ids_q,
                token_ids_p=token_ids_p,
                text_type_ids_p=text_type_ids_p,
                position_ids_p=position_ids_p)
        else:
            if self.label_map:
                label_id = self.label_map[example.label]
            else:
                label_id = example.label

            Record = namedtuple('Record',
                ['token_ids_q', 'text_type_ids_q', 'position_ids_q', \
                 'token_ids_p', 'text_type_ids_p', 'position_ids_p', \
                 'label_id', 'qid'
                ])

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(
                token_ids_q=token_ids_q,
                text_type_ids_q=text_type_ids_q,
                position_ids_q=position_ids_q,
                token_ids_p=token_ids_p,
                text_type_ids_p=text_type_ids_p,
                position_ids_p=position_ids_p,
                label_id=label_id,
                qid=qid)
        return record

    def _convert_example_id_to_record(self, example, q_max_seq_length, p_max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        #query = tokenization.convert_to_unicode(example.query)
        #tokens_query = tokenizer.tokenize(query)
        query_token_id = (example.query).split(' ')
        self._truncate_seq_pair([], query_token_id, q_max_seq_length - 2)

        # title
        #title = tokenization.convert_to_unicode(example.title)
        #tokens_title = tokenizer.tokenize(title)
        # para
        #para = tokenization.convert_to_unicode(example.para)
        #tokens_para = tokenizer.tokenize(para)
        title_token_id = (example.title).split(' ')
        para_token_id = (example.para).split(' ')

        self._truncate_seq_pair(title_token_id, para_token_id, p_max_seq_length - 3)

        #tokens_q = []
        #tokens_q.append("[CLS]")
        text_type_ids_q = []
        text_type_ids_q.append(0)
        token_ids_q = []
        token_ids_q.append(self.cls_id)
        for tid in query_token_id:
            #tokens_q.append(token)
            token_ids_q.append(tid)
            text_type_ids_q.append(0)
        #tokens_q.append("[SEP]")
        token_ids_q.append(self.sep_id)
        text_type_ids_q.append(0)

        #token_ids_q = tokenizer.convert_tokens_to_ids(tokens_q)
        position_ids_q = list(range(len(token_ids_q)))
        #f = open('tid', 'a')
        #for tid in range(len(token_ids_q)):
        #    f.write(str(token_ids_q[tid]) + '\n')

        ### para
        #tokens_p = []
        #tokens_p.append("[CLS]")
        text_type_ids_p = []
        text_type_ids_p.append(0)
        token_ids_p = []
        token_ids_p.append(self.cls_id)

        for tid in title_token_id:
            #tokens_p.append(token)
            token_ids_p.append(tid)
            text_type_ids_p.append(0)
        #tokens_p.append("[SEP]")
        token_ids_p.append(self.sep_id)
        text_type_ids_p.append(0)

        for tid in para_token_id:
            #tokens_p.append(token)
            token_ids_p.append(tid)
            text_type_ids_p.append(1)
        token_ids_p.append(self.sep_id)
        text_type_ids_p.append(1)

        #token_ids_p = tokenizer.convert_tokens_to_ids(tokens_p)
        position_ids_p = list(range(len(token_ids_p)))
        #for tid in range(len(token_ids_p)):
        #    f.write(str(token_ids_p[tid]) + '\n')
        #f.close()

        if self.is_inference:
            Record = namedtuple('Record',
            ['token_ids_q', 'text_type_ids_q', 'position_ids_q', \
             'token_ids_p', 'text_type_ids_p', 'position_ids_p'])
            record = Record(
                token_ids_q=token_ids_q,
                text_type_ids_q=text_type_ids_q,
                position_ids_q=position_ids_q,
                token_ids_p=token_ids_p,
                text_type_ids_p=text_type_ids_p,
                position_ids_p=position_ids_p)
        else:
            if self.label_map:
                label_id = self.label_map[example.label]
            else:
                label_id = example.label

            Record = namedtuple('Record',
                ['token_ids_q', 'text_type_ids_q', 'position_ids_q', \
                 'token_ids_p', 'text_type_ids_p', 'position_ids_p', \
                 'label_id', 'qid'
                ])

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(
                token_ids_q=token_ids_q,
                text_type_ids_q=text_type_ids_q,
                position_ids_q=position_ids_q,
                token_ids_p=token_ids_p,
                text_type_ids_p=text_type_ids_p,
                position_ids_p=position_ids_p,
                label_id=label_id,
                qid=qid)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None, read_id=False):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            if read_id is False:
                record = self._convert_example_to_record(example, self.q_max_seq_len,
                                                         self.p_max_seq_len, self.tokenizer)
            else:
                record = self._convert_example_id_to_record(example, self.q_max_seq_len,
                                                         self.p_max_seq_len, self.tokenizer)
            max_len = max(max_len, len(record.token_ids_p))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                max_len = len(record.token_ids_p)
                batch_records = [record]

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        examples = self._read_tsv(input_file)
        return len(examples)

    def get_pre_examples(self, data):
        examples = []
        headers = ["query", "title", "para", "label"]
        Example = namedtuple('Example', headers)
        for (i, line) in enumerate(data):
            query = (line[0].replace(' ', ''))
            title = (line[1].replace(' ', ''))
            para = (line[2].replace(' ', ''))
            label = '0'
            pre_data =[query, title, para, label]
            #print(pre_data)
            example = Example(*pre_data)
            examples.append(example)
        return examples

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None,
                       read_id=False,
                       data=None):
        if phase == 'stream_predict':
            examples = self.get_pre_examples(data)
        else:
            examples = self._read_tsv(input_file, batch_size)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase, read_id=read_id):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []
        def f():
            try:
                for i in wrapper():
                    yield i
            except Exception as e:
                import traceback
                traceback.print_exc()
        return f


class ClassifyReader(BaseReader):
    def _read_tsv(self, input_file, batch_size=16, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f)
            #headers = next(reader)
            headers = 'query\ttitle\tpara\tlabel'.split('\t')
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        if self.for_cn:
                            line[index] = text.replace(' ', '')
                        else:
                            line[index] = text
                example = Example(*line)
                examples.append(example)
            while len(examples) % batch_size != 0:
                examples.append(example)
            return examples

    def _pad_batch_records(self, batch_records):
        batch_token_ids_q = [record.token_ids_q for record in batch_records]
        batch_text_type_ids_q = [record.text_type_ids_q for record in batch_records]
        batch_position_ids_q = [record.position_ids_q for record in batch_records]

        batch_token_ids_p = [record.token_ids_p for record in batch_records]
        batch_text_type_ids_p = [record.text_type_ids_p for record in batch_records]
        batch_position_ids_p = [record.position_ids_p for record in batch_records]

        if not self.is_inference:
            batch_labels = [record.label_id for record in batch_records]
            if self.is_classify:
                batch_labels = np.array(batch_labels).astype("int64").reshape(
                    [-1, 1])
            elif self.is_regression:
                batch_labels = np.array(batch_labels).astype("float32").reshape(
                    [-1, 1])

            if batch_records[0].qid:
                batch_qids = [record.qid for record in batch_records]
                batch_qids = np.array(batch_qids).astype("int64").reshape(
                    [-1, 1])
            else:
                batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        # padding
        padded_token_ids_q, input_mask_q = pad_batch_data(
            batch_token_ids_q, pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids_q = pad_batch_data(
            batch_text_type_ids_q, pad_idx=self.pad_id)
        padded_position_ids_q = pad_batch_data(
            batch_position_ids_q, pad_idx=self.pad_id)
        padded_task_ids_q = np.ones_like(padded_token_ids_q, dtype="int64") * self.task_id

        padded_token_ids_p, input_mask_p = pad_batch_data(
            batch_token_ids_p, pad_idx=self.pad_id, return_input_mask=True)
        padded_text_type_ids_p = pad_batch_data(
            batch_text_type_ids_p, pad_idx=self.pad_id)
        padded_position_ids_p = pad_batch_data(
            batch_position_ids_p, pad_idx=self.pad_id)
        padded_task_ids_p = np.ones_like(padded_token_ids_p, dtype="int64") * self.task_id

        if "infer_p" in self.infer_qp:
            return_list = [
                padded_token_ids_p, padded_text_type_ids_p, padded_position_ids_p, padded_task_ids_p,
                input_mask_p,
            ]
        if "infer_q" in self.infer_qp:
            return_list = [
                padded_token_ids_q, padded_text_type_ids_q, padded_position_ids_q, padded_task_ids_q,
                input_mask_q,
            ]
            #padded_token_ids_q, padded_text_type_ids_q, padded_position_ids_q, padded_task_ids_q,
            #input_mask_q,
        if not self.is_inference:
            return_list += [batch_labels, batch_qids]

        return return_list

if __name__ == '__main__':
    pass
