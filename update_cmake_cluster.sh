#!/bin/bash

SRV='pd2023103@lorien.atp-fivt.org'

ssh "$SRV" '((if [ ! -d cmake ]; then git clone "https://gitlab.kitware.com/cmake/cmake.git"; fi) && cd cmake && ./configure && make -j10 && ls $(pwd)/bin/cmake) && echo >> ~/.bashrc && echo -n export PATH=$(pwd)/bin/cmake:\$PATH >> ~/.bashrc'
