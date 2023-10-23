#!/bin/bash

# ssh-copy-id pd2023103@lorien.atp-fivt.org

# pd2023103@lorien.atp-fivt.org

SRV='pd2023103@lorien.atp-fivt.org'
PROJ_DIR="$(basename $(dirname $(readlink -e $0)))"

ssh "$SRV" "mkdir -p $PROJ_DIR"
file_list=$(ls -a | grep -v -E '^\.' | grep -v build)
rsync -e ssh -ah $file_list "$SRV:$PROJ_DIR/"
ssh "$SRV" "export PATH=~/cmake/bin:\$PATH && cd $PROJ_DIR && cmake -B build \. && make -C build && $1"
