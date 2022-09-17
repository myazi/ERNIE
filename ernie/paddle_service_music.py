import json
from tornado import web
from tornado import ioloop
import time
import os
import logging
import logging.handlers
import urllib
import argparse
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params
import paddle_predict_music

import chardet
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

total_time = 0

def init_log(log_path, level=logging.INFO, when="D", backup=7,
        format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
        datefmt="%m-%d %H:%M:%S"):
    """
        init_log - initialize log module

        Args:
            log_path      - Log file path prefix.
                            Log data will go to two files: log_path.log and log_path.log.wf
                            Any non-exist parent directories will be created automatically
            level         - msg above the level will be displayed
                            DEBUG < INFO < WARNING < ERROR < CRITICAL
                            the default value is logging.INFO
            when          - how to split the log file by time interval
                            'S' : Seconds
                            'M' : Minutes
                            'H' : Hours
                            'D' : Days
                            'W' : Week day
                            default value: 'D'
            format        - format of the log
                            default format:
                            %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
                            INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
            backup        - how many backup file to keep
                            default value: 7
                                                                                                                                                                
            Raises:
                          OSError: fail to create log directories
                          IOError: fail to open log file
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log",
            when=when,
            backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log.wf",
            when=when,
            backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def set_paras():
    parser = argparse.ArgumentParser(__doc__)
    model_g = ArgumentGroup(parser, "model", "options to init, resume and save model.")
    model_g.add_arg("ernie_config_path", str, None, "Path to the json file for bert model config.")
    model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
    model_g.add_arg("save_inference_model_path", str, "inference_model",
                    "If set, save the inference model to this path.")
    model_g.add_arg("use_fp16", bool, False, "Whether to resume parameters from fp16 checkpoint.")
    model_g.add_arg("num_labels", int, 2, "num labels for classify")
    model_g.add_arg("ernie_version", str, "2.0", "ernie_version")
    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
    data_g.add_arg("predict_set", str, None, "Predict set file")
    data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
    data_g.add_arg("data_dir", str, None, "Directory to test data.")
    data_g.add_arg("label_map_config", str, None, "Label_map_config json file.")
    data_g.add_arg("max_seq_len", int, 128, "Number of words of the longest seqence.")
    data_g.add_arg("batch_size", int, 32,
                   "Total examples' number in batch for training. see also --in_tokens.")
    data_g.add_arg("do_lower_case", bool, True,
                   "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
    run_type_g.add_arg("do_prediction", bool, True, "Whether to do prediction on test set.")
    run_type_g.add_arg("server_port", int, 8031, "server port")
    run_type_g.add_arg("use_ema", bool, False, "Whether to use ema.")
    run_type_g.add_arg("ema_decay", float, 0.9999, "Decay rate for expoential moving average.")
    args = parser.parse_args()
    return args


def create_ernie_server(args):
    predictor = paddle_predict_music.QpPredictor(args)

    class ErnieHandler(web.RequestHandler):
        ernie_predictor = predictor
        begin_time = -10

        def __init__(self, application, request, **kwargs):
            web.RequestHandler.__init__(self, application, request)

        def get(self):
            """
            Get request
            """
            results = self.http()
            self.write(json.dumps(results))

        def post(self):
            try:
                """
                json_line = self.get_post_data()
                if json_line is None:
                    ret = {}
                    ret['status'] = 109
                    self.write(json.dumps(ret))
                    return
                input_data = json.loads(json_line)
                """
                input_data = json.loads(self.request.body)
                #json_line = urllib.unquote(input_data)
                #input_data = json.loads(json_line)
                result_str = self.get_proc_res(input_data)
                self.write(result_str)
            except Exception as e:
                ret = {}
                ret['status'] = 110
                self.write(json.dumps(ret))
                logging.fatal(str(e))

        def get_post_data(self):
            """Get http input data & decode into json object
            Returns:
                json object contains the input (q, p) paris
            """
            #logging.info("[get post data]")
            input_data = self.request.body
            json_line = urllib.unquote(input_data)
            if json_line is None or json_line == "":
                logging.fatal("Empty input - {}\n".format(input_data))
                return None
            return json_line

        def get_proc_res(self, input_data):
            """Data format transformation & call pretrained MRC model
            Args:
                Input json object contains the (q, p) paris
            Returns:
                The answer predicted by the MRC model (in json format)
            """
            #logging.info("[get_proc_res]")
            data_list = []
            query = input_data['query']
            #if args.model_type == 'Ernie':
            #    query = query.replace(' ', '')
            paras = input_data['paras']
            if 'titles' in input_data:
                titles = input_data['titles']
            else:
                titles = ['' for i in range(len(query))]

            is_list = isinstance(query, list)
            if is_list is True:
                if len(set([len(query), len(paras), len(titles)])) != 1:
                   return None
                for q_id in range(len(query)):
                   q = query[q_id]
                   para = paras[q_id]
                   title = titles[q_id]
                   data_list.append([q, title, para])
            else:
                #logging.info("[input_data:\t%s]" % (query))
                for para in paras:
                   data_list.append([query, title, para])

            self.begin_time = time.time()
            ernie_output = self.ernie_predictor.predict(data_list)
            cost_time = time.time() - self.begin_time
            global total_time
            total_time += cost_time
            #logging.info("predict time:\t%f" % (total_time))
            output_json = {}
            output_json['probability'] = []
            for prediction in ernie_output:
                output_json['probability'].append(float(prediction))
                #logging.info('DEBUG' + str(prediction))
            # output_json['begin_p_id'] = input_data['begin_p_id']
            # output_json['end_p_id'] = input_data['end_p_id']
            output_json['status'] = 0
            #logging.info('result len: %d' % (len(output_json['probability'])))

            result_str = json.dumps(output_json, ensure_ascii=False)
            #logging.info('[Output from model]: {}'.format(result_str))
            return result_str

    return ErnieHandler


def create_ernie_app(sub_address, ernie_server):
    """
    Create DQA server application
    """
    #print sub_address
    return web.Application([(sub_address, ernie_server)])


if __name__ == "__main__":
    init_log("./log/ernie_server")
    args = set_paras()
    #bert_handler = create_bert_handler(args)
    #start_server(args.server_port, bert_handler)
    sub_address = r'/qp_' + args.init_checkpoint.replace('_infer', '')
    sub_address = r'/music'
    #print sub_address
    ernie_server = create_ernie_server(args)
    if "infer" in args.init_checkpoint:
        app = create_ernie_app(sub_address, ernie_server)
        app.listen(args.server_port)
        ioloop.IOLoop.current().start()
