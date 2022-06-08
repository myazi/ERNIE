#!/bin/bash

rm -rf *.wav
sh run_sedit_en.sh       # 语音编辑任务(英文) 
sh run_gen_en.sh         # 个性化语音合成任务(英文)
sh run_clone_en_to_zh.sh # 跨语言语音合成任务(英文到中文的语音克隆)