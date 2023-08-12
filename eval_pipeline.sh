# E.g., run:
#   bash eval_pipeline.sh /isaac-sim/python.sh pickup_object peract clip 0 1

RUNNER=$1
TASK=$2
AGENT=$3
LANG=$4
STATE_HEAD=$5
PARTIAL_GT=$6


if [ ${STATE_HEAD} = "1" ]; then
    until ${RUNNER} ckpt_selection.py task=${TASK} model=${AGENT} lang_encoder=${LANG} \
                                      mode=eval state_head=1; do :; done
else
    until ${RUNNER} ckpt_selection.py task=${TASK} model=${AGENT} lang_encoder=${LANG} \
                                      mode=eval; do :; done
fi

if [ ${STATE_HEAD} = "1" ]; then
    until ${RUNNER} eval.py task=${TASK} model=${AGENT} lang_encoder=${LANG} \
                            mode=eval state_head=1; do :; done
    if [ ${PARTIAL_GT} = "1" ]; then
        until ${RUNNER} eval.py task=${TASK} model=${AGENT} lang_encoder=${LANG} \
                                mode=eval use_gt=[1,0] state_head=1; do :; done
    fi
else
    until ${RUNNER} eval.py task=${TASK} model=${AGENT} lang_encoder=${LANG} \
                            mode=eval; do :; done
    if [ ${PARTIAL_GT} = "1" ]; then
        until ${RUNNER} eval.py task=${TASK} model=${AGENT} lang_encoder=${LANG} \
                                mode=eval use_gt=[1,0]; do :; done
    fi
fi
