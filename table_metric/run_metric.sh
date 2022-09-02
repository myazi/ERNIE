task_name=$1
thre=$2
path=data
cur=`dirname $0`

for task in ${task_name};
do
    echo ${task}: step - threshold - pre - rec - f1 - auc
    for i in `ls ${path}`;do
        if [[ "${i}" == ${task}*score* ]];then
            /root/anaconda3/envs/pytorch/bin/python ${cur}/train_dev_pr.py ${path}/${i} ${task} $thre
        fi
    done
done
