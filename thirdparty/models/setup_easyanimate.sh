#!/bin/bash

export THIRD_PARTY_DIR=$(cd $(dirname $0); pwd)
export EASYANIMATE_DIR=${THIRD_PARTY_DIR}/EasyAnimate

# setup easyanimate
echo "> setup easyanimate ..."
git clone https://github.com/aigc-apps/EasyAnimate.git
cd $EASYANIMATE_DIR
git reset b54412ceb0af6a06bf907e049920f18508c862f1 --hard
git apply ${THIRD_PARTY_DIR}/patch/easyanimate.diff
