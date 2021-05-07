# -*- coding: utf-8 -*-
#
# Huseyin Alecakir <huseyinalecakir@gmail.com>

SHELL = /bin/bash


# check whether the correct python version is available
ifeq (, $(shell which python3 ))
	$(error "python3 not found in $(PATH)")
endif

# check whether the cuda compiler driver
ifeq (, $(shell which nvcc))
	$(error "cuda compiler driver not found in $(PATH)")
endif

GOOGLE_WORDVEC := "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
FASTTEXT_WORDVEC := "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
GLOVE_WORDVEC := "http://nlp.stanford.edu/data/glove.840B.300d.zip"

VENV := "venv/bin/activate"


venv/bin/activate: requirements.txt
	@echo -e "\e[0;32mINFO     Creating virtual environment and installing requirements...\e[0m"
	test -d venv || python3 -m venv venv
	. $(VENV) && pip3 install -Ur requirements.txt
	@touch venv/bin/activate

data/vectors:
	@echo -e "\e[0;32mINFO     Creating word vector folder...\e[0m"
	@mkdir data/vectors

data/vectors/.google: data/vectors
	@echo -e "\e[0;32mINFO     Fetching google wordvec corpus...\e[0m"
	@wget -O "data/vectors/google.bin.gz" $(GOOGLE_WORDVEC)
	@cd data/vectors && gunzip google.bin.gz
	@touch data/vectors/.google

data/vectors/.fasttext: data/vectors
	@echo -e "\e[0;32mINFO     Fetching fasttext wordvec corpus...\e[0m"
	@wget -O "data/vectors/fasttext.bin.gz" $(FASTTEXT_WORDVEC)
	@cd data/vectors && gunzip fasttext.bin
	@touch data/vectors/.fasttext

data/vectors/.glove: data/vectors
	@echo -e "\e[0;32mINFO     Fetching glove wordvec corpus...\e[0m"
	@wget -O "data/vectors/glove.840B.300d.zip" $(GLOVE_WORDVEC)
	@cd data/vectors/ && unzip glove.840B.300d.zip && rm glove.840B.300d.zip
	@cd data/vectors/ && mv glove.840B.300d.txt glove.txt
	@touch data/vectors/.glove

.PHONY: fetch_all
fetch_all: data/vectors/.glove data/vectors/.fasttext data/vectors/.google

.PHONY: train
train: venv/bin/activate fetch_all
	@echo -e "\e[0;32mINFO     Training Toxic Comment Classifier with default settings...\e[0m"
	@. $(VENV) && sh run.sh train $(ARGS)

.PHONY: test
test: venv/bin/activate experiments/model experiments/model experiments/model.params
	@echo -e "\e[0;32mINFO     Testing Toxic Comment Classifier...\e[0m"
	@. $(VENV) && sh run.sh test

.PHONY: clean
clean:
	@rm -rf experiments
	@rm -rf logs


