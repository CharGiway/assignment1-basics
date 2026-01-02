#!/bin/bash
nohup ./run_train_tokenizer.sh > train.log 2>&1 & echo $! > train.pid