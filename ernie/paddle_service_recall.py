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
import paddle_predict_recall

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
    data_g.add_arg("q_max_seq_len", int, 32, "Number of words of the longest seqence.")
    data_g.add_arg("p_max_seq_len", int, 384, "Number of words of the longest seqence.")
    data_g.add_arg("batch_size", int, 128,
                   "Total examples' number in batch for training. see also --in_tokens.")
    data_g.add_arg("do_lower_case", bool, True,
                   "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    data_g.add_arg("test_save", str, None, "Directory to test result.")
    data_g.add_arg("output_item", str, None, "Return fields.")
    data_g.add_arg("output_file_name", str, None, "Return fields.")
    data_g.add_arg("read_id", bool, False, ".")
    data_g.add_arg("save_part", str, "all", "save model type: all, query, para")
    data_g.add_arg("for_cn", bool, True, ".")

    run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
    run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
    run_type_g.add_arg("do_prediction", bool, True, "Whether to do prediction on test set.")
    run_type_g.add_arg("server_port", int, 8031, "server port")
    run_type_g.add_arg("use_ema", bool, False, "Whether to use ema.")
    run_type_g.add_arg("ema_decay", float, 0.9999, "Decay rate for expoential moving average.")
    args = parser.parse_args()
    return args


def create_ernie_server(args):
    predictor = paddle_predict_recall.TwinPredictor(args)

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
                json_line = self.get_post_data()
                if json_line is None:
                    ret = {}
                    ret['status'] = 109
                    self.write(json.dumps(ret))
                    return
                input_data = json.loads(json_line)
                # ensure chinese character output
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
            #json_line = urllib.unquote(input_data)
            json_line = urllib.parse.unquote(bytes.decode(input_data))
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
            query_list = input_data['query']
            title_list = input_data['title']
            para_list = input_data['para']
            if len(query_list) != len(para_list) or len(para_list) != len(title_list):
                logging.fatal('input format error %d %d %d' % (len(query_list), len(para_list), len(title_list)))
                return None
            for q_id in range(len(query_list)):
                q = query_list[q_id]
                title = title_list[q_id]
                para = para_list[q_id]
                data_list.append([q, title, para])

            self.begin_time = time.time()
            reps = self.ernie_predictor.predict(data_list)
            #if len(q_reps) != len(p_reps) or len(p_reps) != len(ip_scores):
            #    logging.fatal('ernie result error %d %d %d' % (len(q_reps), len(p_reps), len(ip_scores)))
            #    return None

            cost_time = time.time() - self.begin_time
            global total_time
            total_time += cost_time
            #logging.info("predict time:\t%f" % (total_time))
            output_json = {}
            output_json['p_rep'] = []
            output_json['q_rep'] = []
            for res_id in range(len(reps)):
                if "infer_p" in args.init_checkpoint:
                    output_json['p_rep'].append(reps[res_id].tolist())
                if "infer_q" in args.init_checkpoint:
                    output_json['q_rep'].append(reps[res_id].tolist())
                #logging.info('DEBUG' + str(ip_scores[res_id]))
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
    return web.Application([(sub_address, ernie_server)])


if __name__ == "__main__":
    init_log("./log/ernie_server")
    args = set_paras()
    #bert_handler = create_bert_handler(args)
    #start_server(args.server_port, bert_handler)
    ernie_server = create_ernie_server(args)
    if "infer_p" in args.init_checkpoint:
        sub_address = r"/recall_p"
        app = create_ernie_app(sub_address, ernie_server)
        app.listen(args.server_port)
        ioloop.IOLoop.current().start()
    if "infer_q" in args.init_checkpoint:
        sub_address = r"/recall_q"
        app = create_ernie_app(sub_address, ernie_server)
        app.listen(args.server_port)
        ioloop.IOLoop.current().start()
