path=$1
cur=`dirname $0`

for task in music_annotation_new_url_sort_test_qtp1;
do
    for i in `ls ${path}`;do
        if [[ "${i}" == ${task}* ]];then
            echo ${i}
            python_gcc4/bin/python ${cur}/recall_ndcg.py ${task} ${path}/${i} a
        fi
#        python ${cur}/metric_pv.py ${path}/${task}.score.$step.0.0 $step
    done
done
