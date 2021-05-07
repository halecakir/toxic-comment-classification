#!/bin/bash

TRAIN_OR_TEST=$1
WORDVECTORS=$2

if [ $TRAIN_OR_TEST = "train" ]; then
	CUBLAS_WORKSPACE_CONFIG=:16:8 python run.py train --train-data=data/train.csv --verbose --external-embedding=data/vectors/$WORDVECTORS
elif [ $TRAIN_OR_TEST = "test" ]; then
    CUBLAS_WORKSPACE_CONFIG=:16:8 python run.py test --test-data-sents=data/test.csv --test-data-labels=data/test_labels.csv --verbose
else
	echo "Invalid Option Selected"
fi