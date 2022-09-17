# -*- coding: UTF-8 -*-
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
"""Load classifier's checkpoint to do prediction or save inference model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import multiprocessing
import numpy as np
import paddle.fluid as fluid
import paddle
if hasattr(paddle, 'enable_static'):
    paddle.enable_static()

from model.ernie import ErnieConfig
from finetune.classifier_service import create_model
from reader.task_reader_service import ClassifyReader
#import reader.cls as reader
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params
import logging

class QpPredictor(object):
    def __init__(self, args):
        ernie_config = ErnieConfig(args.ernie_config_path)
        ernie_config.print_config()
        self.qp_processor = ClassifyReader(vocab_path=args.vocab_path,
                                          label_map_config=args.label_map_config,
                                          max_seq_len=args.max_seq_len,
                                          do_lower_case=args.do_lower_case,
                                          in_tokens=False,
                                          is_inference=False)

        if "infer" not in args.init_checkpoint:
            self.predict_prog = fluid.Program()
            predict_startup = fluid.Program()
            with fluid.program_guard(self.predict_prog, predict_startup):
                with fluid.unique_name.guard():
                    self.predict_pyreader, self.probs, self.feed_target_names =  create_model(
                        args,
                        pyreader_name='predict_reader',
                        ernie_config=ernie_config,
                        is_classify=True,
                        is_prediction=True,
                        ernie_version=args.ernie_version)
                    if args.use_ema and 'ema' not in dir():
                        self.ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)
                    else:
                        self.ema = None

            self.predict_prog = self.predict_prog.clone(for_test=True)
            if args.use_cuda:
                place = fluid.CUDAPlace(0)
                dev_count = fluid.core.get_cuda_device_count()
            else:
                place = fluid.CPUPlace()
                dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

            #place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
            self.exe = fluid.Executor(place)
            self.exe.run(predict_startup)
            if args.init_checkpoint:
                logging.info('CHECKPOINT: %s' % (args.init_checkpoint))
                init_pretraining_params(self.exe, args.init_checkpoint, predict_startup)
            else:
                raise ValueError("args 'init_checkpoint' should be set for prediction!")
            self.args = args
            fluid.io.save_inference_model(
                args.save_inference_model_path,
                self.feed_target_names, [self.probs],
                self.exe,
                main_program=self.predict_prog)
        else:
            if args.use_cuda:
                place = fluid.CUDAPlace(0)
                dev_count = fluid.core.get_cuda_device_count()
            else:
                place = fluid.CPUPlace()
                dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
                dev_count = 8
            self.exe = fluid.Executor(place)

            assert args.init_checkpoint is not None
            #_, ckpt_dir = os.path.split(args.init_checkpoint.rstrip('/'))
            #dir_name = ckpt_dir + '_inference_model'
            #print("load inference model from %s" % model_path)
            #model_path = os.path.join(args.save_inference_model_path, dir_name)
            self.infer_program, feed_target_names, self.probs = fluid.io.load_inference_model(
                args.init_checkpoint, self.exe) #model_filename='model', params_filename='params')
            self.src_ids = feed_target_names[0]
            self.sent_ids = feed_target_names[1]
            self.pos_ids =  feed_target_names[2]
            self.input_mask = feed_target_names[3]
            self.task_ids =  feed_target_names[4]
            self.args = args

    def predict(self, data):
        predict_data_generator = self.qp_processor.data_generator(
                  input_file=None,
                  batch_size=self.args.batch_size, epoch=1, shuffle=False,
                  phase='stream_predict', data=data)
        all_results = []
        for sample in predict_data_generator():
            src_ids_data = sample[0]
            sent_ids_data = sample[1]
            pos_ids_data = sample[2]
            task_ids_data = sample[3]
            input_mask_data = sample[4]
            output = self.exe.run(
                self.infer_program,
                feed={self.src_ids: src_ids_data,
                      self.sent_ids: sent_ids_data,
                      self.pos_ids: pos_ids_data,
                      self.input_mask: input_mask_data,
                      self.task_ids: task_ids_data},
                fetch_list=self.probs)
            all_results.extend(output[0])
        return all_results

if __name__ == '__main__':
    print_arguments(args)
    # main(args)
    predictor = QpPredictor(args)
