## 文件

### 训练
```
run.sh: spo任务训练脚本
run_new.sh: music任务训练脚本
run_recall.sh: recall任务训练脚本

ernie/run_classifier_new.py: 训练入口文件，用于music任务训练
ernie/run_classifier_orgin.py: 训练入口文件，用于spo任务训练
ernie/run_classifier_recall.py: 训练入口文件，用于recall任务训练

ernie/finetune/classifier_nclass.py: finetune模型文件，支持n-pairwise loss，用于music任务训练
ernie/finetune/classifier_orgin.py: finetune模型文件，支持pointwise loss，用于spo任务训练
ernie/finetune/classifier_recall.py: finetune模型文件，recall模型

ernie/reader/task_reader_orgin.py: read文件，支持pointwise数据读取，用于spo任务训练
ernie/reader/task_reader_nclass.py: read文件，支持n-pairwise数据读取，用于music任务训练
ernie/reader/task_reader_recall.py: read文件，支持recall数据读取，永远recall任务训练
```

### 起服务
```
run_paddle_service.sh: spo起服务脚本，脚本执行两次，第一次产出infer model，第二次带上infer起服务
run_paddle_service_music.sh: music起服务脚本，脚本执行两次，第一次产出infer model，第二次带上infer起服务
run_paddle_service_recall.sh: recall起服务脚本，脚本第一次产出infer model，kill掉服务，注释掉infer部分再执行脚本起服务

ernie/paddle_service.py: 起服务文件，支持spo
ernie/paddle_service_music.py: 起服务文件，支持music
ernie/paddle_service_recall.py: 起服务文件，支持recall

ernie/paddle_predict.py: 起服务预测文件，支持spo
ernie/paddle_predict_music.py: 起服务预测文件，支持music
ernie/paddle_predict_recall.py: 起服务预测文件，支持recall任务

ernie/finetune/classifier_service.py: finetune模型文件，支持spo任务起服务
ernie/finetune/classifier_service_music.py: finetune模型文件，支持music任务起服务
ernie/finetune/classifier_service_recall.py: finetune模型文件，支持recall任务起服务

ernie/reader/task_reader_service.py: read文件，支持line数据读取，用于spo任务服务
ernie/reader/task_reader_service_music.py: read文件，支持line数据读取，用于music任务服务
ernie/reader/task_reader_service_recall.py: read文件，支持reacll数据读取，用户recall任务服务

ernie/finetune_args.py: 训练参数配置，增加了一些q_max_seq_len、only_pointwise、margin等
ernie/model/ernie.py:  增加默认参数model_name，召回模型中输出参数会带上model_name前缀, pool输出层需要区别rank和recall来使用
ernie/model/transformer_encoder.py: 增加默认参数model_name、end_learning_rate，适配现有代码
ernie/tokenizer_char_word_mix.py: 支持5万词表的模型训练 
```

