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
from finetune.classifier_service_recall import create_model
from reader.task_reader_service_recall import ClassifyReader
#import reader.cls as reader
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_checkpoint, init_pretraining_params
import logging

class TwinPredictor(object):

    def __init__(self, args):
        ernie_config = ErnieConfig(args.ernie_config_path)
        ernie_config.print_config()
        self.twin_processor = ClassifyReader(vocab_path=args.vocab_path,
                                          label_map_config=args.label_map_config,
                                          q_max_seq_len=args.q_max_seq_len,
                                          p_max_seq_len=args.p_max_seq_len,
                                          do_lower_case=args.do_lower_case,
                                          infer_qp=args.init_checkpoint,#一定需要指定模型
                                          in_tokens=False,
                                          is_inference=False)
        if "infer" not in args.init_checkpoint:
            self.predict_prog = fluid.Program()
            predict_startup = fluid.Program()
            with fluid.program_guard(self.predict_prog, predict_startup):
                with fluid.unique_name.guard():
                    self.predict_pyreader, self.feed_target_names, \
                        results = create_model(
                        args,
                        pyreader_name='predict_reader',
                        ernie_config=ernie_config,
                        is_classify=True,
                        is_prediction=True,
                        save_part=args.save_part)
                    #if args.use_ema and 'ema' not in dir():
                    #    self.ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)
                    #else:
                    #    self.ema = None

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
                init_checkpoint(self.exe, args.init_checkpoint, predict_startup)
            else:
                raise ValueError("args 'init_checkpoint' should be set for prediction!")
            self.args = args
            for var in self.feed_target_names:
                print (var)
            fluid.io.save_inference_model(
                args.save_inference_model_path,
                self.feed_target_names, results,
                self.exe,
            main_program=self.predict_prog) #,
        else:
            if args.use_cuda:
                place = fluid.CUDAPlace(0)
                dev_count = fluid.core.get_cuda_device_count()
            else:
                place = fluid.CPUPlace()
                dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

            place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
            self.exe = fluid.Executor(place)

            self.infer_program, feed_target_names, self.p_rep  = fluid.io.load_inference_model(
                args.init_checkpoint, self.exe) #, model_filename='model', params_filename='params')
            #self.prob, self.q_rep, self.p_rep = fetch_targets[0], fetch_targets[1], fetch_targets[2]
            self.src_ids_p = feed_target_names[0]
            self.sent_ids_p = feed_target_names[1]
            self.pos_ids_p =  feed_target_names[2]
            self.task_ids_p =  feed_target_names[3]
            self.input_mask_p = feed_target_names[4]
            self.args = args


    def predict(self, data):
        predict_data_generator = self.twin_processor.data_generator(
                  input_file=None,
                  batch_size=self.args.batch_size, epoch=1, shuffle=False,
                  phase='stream_predict', data=data)
        p_reps = []
        for sample in predict_data_generator():
            src_ids_data_p = sample[0]
            sent_ids_data_p = sample[1]
            pos_ids_data_p = sample[2]
            task_ids_data_p = sample[3]
            input_mask_data_p = sample[4]

            results = self.exe.run(
                self.infer_program,
                feed={self.src_ids_p: src_ids_data_p,
                      self.sent_ids_p: sent_ids_data_p,
                      self.pos_ids_p: pos_ids_data_p,
                      self.input_mask_p: input_mask_data_p,
                      self.task_ids_p: task_ids_data_p},
                      fetch_list=self.p_rep)
            p_reps.extend(results[0])
        return p_reps

if __name__ == '__main__':
    print_arguments(args)
    # main(args)
    predictor = QpPredictor(args)
