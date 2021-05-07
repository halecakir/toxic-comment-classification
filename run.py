"""Toxic Comment Classification

Huseyin Alecakir huseyinalecakir@gmail.com

Usage:
  run.py train --train-data=<file>  [options]
  run.py test --test-data-sents=<file> --test-data-labels=<file> [options]
  run.py optim --train-data=<file> [options]

  main.py (-h | --help)

Example, try:
  python run.py train --train-data=data/train.csv

Options:
  -h --help                         Show this screen.
  --train-data=<file>               Train set.
  --test-data-sents=<file>          Test set unlabeled sentences.
  --test-data-labels=<file>         Test set labels.
  --external-embedding=<file>       Pre-trained vector embeddings.
  --batch-size=<int>                Batch size [default: 256].
  --embedding-size=<int>            Embedding size [default: 300].
  --hidden-size=<int>               Hidden size [default: 128].
  --lstm-num-layers=<int>           LSTM num layers [default: 2].
  --output-size=<int>               Output size [default: 6].
  --dropout-rate=<float>            Dropout rate [default: 0.2].
  --lr=<float>                      Learning rate [default: 0.001].
  --lr-decay=<float>                Learning rate decay [default: 0.5].
  --epoch=<int>                     Epoch num [default: 50].
  --report-niter=<int>              Report everty nth iteration [default: 50].
  --valid-niter=<int>               Validate everty nth iteration [default: 100].
  --valid-ratio=<float>             Validation ratio [default: 0.1].
  --valid-metric=<str>              Validation metric [default: roc_auc_macro].
  --save-dir=<file>                 Artifact directory [default: experiments]
  --model-file=<file>               Model params save path [default: model]
  --params-file=<file>              Hyperparams & vocab save path [default: model.params]
  --optim-file=<file>               Optimizator state save path [default: model.optim]
  --patience=<int>                  Number of decay learning rate [default: 5].
  --max-trial=<int>                 Maximum number of trial [default: 20].
  --seed=<int>                      Seed number [default: 3].
  --disable-cuda                    Disable cuda [default: False].
  --verbose                         Verbose print [default: False].
"""
import logging
import os
import pickle
import random
import time

from comet_ml import Experiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from docopt import docopt
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from classifier import ToxicCommentClassifier
from utils.general_utils import (Vocab, get_minibatches, load_embeddings_file,
                                 read_test_corpus, read_train_corpus)
from utils.log_utils import ColoredLogger
from utils.nlp_utils import NLPUtils

logger = ColoredLogger("project")

experiment = Experiment(
    api_key="nLqFerDLnwvCiAptbL4u0FZIj",
    project_name="toxic-comment-classification",
    workspace="halecakir",
)


def reset_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train(args, device):
    def prepare_data(train_data, params_save_path, valid_ratio):
        sentences, labels = read_train_corpus(train_data, limit=1000)
        sentences = NLPUtils.preprocess_pipeline(sentences)

        sentences = np.array(sentences, dtype=object)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            sentences, labels, test_size=valid_ratio
        )

        vocab = Vocab(device)
        vocab.words2indices(X_train)

        with open(params_save_path, "wb") as paramsfp:
            pickle.dump((vocab, args), paramsfp)

        logger.debug(f"Train/Dev size : {len(X_train)}/{len(X_test)}")
        logger.debug(f"Train vocab size : {len(vocab.w2i)}")

        return [X_train, y_train], [X_test, y_test], vocab

    def add_to_history(history, epoch, train_iter, train_loss, val_loss, val_metrics):
        history["epoch"].append(epoch)
        history["train_iter"].append(train_iter)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for metric in val_metrics:
            if metric not in history:
                history[metric] = []
            history[metric].append(val_metrics[metric])

    start = time.time()
    train_instances, validation_instances, vocab = prepare_data(
        args["--train-data"], args["--params-file"], float(args["--valid-ratio"])
    )
    logger.debug(f"Data prepration duration {(time.time() - start):.2f}s")

    model = ToxicCommentClassifier(
        int(args["--embedding-size"]),
        int(args["--hidden-size"]),
        int(args["--lstm-num-layers"]),
        int(args["--output-size"]),
        vocab,
        float(args["--dropout-rate"]),
    )

    if args["--external-embedding"] is not None:
        logger.debug("Initializing word embeddings by pre-trained vectors")
        start = time.time()
        ext_embeddings, ext_emb_dim = load_embeddings_file(
            args["--external-embedding"], lower=True
        )
        assert (
            int(args["--embedding-size"]) == ext_emb_dim
        ), f"External embedding dim is {ext_emb_dim}"

        count, vocab_size = model.initalize_pretrained_embeddings(ext_embeddings)
        logger.debug(
            f"Vocab size: {vocab_size}; #words having pretrained vectors: {count}"
        )
        logger.debug(
            f"Pre-trained embedding reading&loading duration {(time.time() - start):.2f}s"
        )

    model.to(device=device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args["--lr"]))
    criterion = nn.BCELoss()

    logger.debug("Training started")
    start_train = time.time()
    cum_loss = 0
    report_loss = 0
    cum_sents = 0
    report_sents = 0

    patience = 0
    trial = 0
    train_iter = 0
    best_valid_metric = None
    history = {"epoch": [], "train_iter": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, int(args["--epoch"]) + 1):
        logger.debug(f"Epoch {epoch} started")
        start = time.time()
        for sents, labels in get_minibatches(
            train_instances, batch_size=int(args["--batch-size"]), shuffle=True
        ):
            train_iter += 1
            optimizer.zero_grad()

            labels = torch.tensor(labels).type(torch.FloatTensor).to(device=device)

            batch_size = len(sents)
            scores = model(sents)

            losses = criterion(scores, labels)

            batch_loss = losses.sum()
            loss = batch_loss / batch_size
            loss.backward()

            optimizer.step()

            batch_losses_val = batch_loss.item()
            cum_loss += batch_losses_val
            report_loss += batch_losses_val

            cum_sents += batch_size
            report_sents += batch_size
            if train_iter % int(args["--report-niter"]) == 0:
                logger.debug(
                    f"[Train report]: train-iter {train_iter}: cumulative instances {report_sents}, cumulative loss={report_loss / report_sents}"
                )
                report_loss, report_sents = 0, 0

            if train_iter % int(args["--valid-niter"]) == 0:
                results, val_loss = evaluate(
                    model, validation_instances, int(args["--batch-size"]), device
                )
                logging.debug(
                    f"[Validation report] Train loss {val_loss} Val loss {cum_loss/cum_sents}"
                )

                add_to_history(
                    history=history,
                    epoch=epoch,
                    train_iter=train_iter,
                    train_loss=cum_loss / cum_sents,
                    val_loss=val_loss,
                    val_metrics=results,
                )

                # log train metrics to comet.ml
                with experiment.train():
                    experiment.log_metric("loss", cum_loss / cum_sents, step=train_iter)

                # log validation metrics to comet.ml
                with experiment.validate():
                    experiment.log_metric("val_loss", val_loss, step=train_iter)
                    for metric in results:
                        experiment.log_metric(metric, results[metric], step=train_iter)

                cum_loss, cum_sents = 0, 0

                if (
                    not best_valid_metric
                    or results[args["--valid-metric"]] > best_valid_metric
                ):
                    best_valid_metric = results[args["--valid-metric"]]
                    logger.debug(
                        f"[Validation report]: Best validation {args['--valid-metric']} score : {best_valid_metric}"
                    )
                    patience = 0
                    torch.save(model.state_dict(), args["--model-file"])
                    torch.save(optimizer.state_dict(), args["--optim-file"])

                elif patience < int(args["--patience"]):
                    patience += 1
                    logger.debug(
                        f"[Validation report]: Hit patience {patience}/{int(args['--patience'])}"
                    )

                    if patience == int(args["--patience"]):
                        trial += 1
                        logger.debug(
                            f"[Validation report]: Hit trial {trial}/{int(args['--max-trial'])}"
                        )
                        if trial == int(args["--max-trial"]):
                            logger.debug("[Validation report]: Early stop")
                            logger.debug(
                                f"Training duration {(time.time() - start_train):.2f}s"
                            )
                            experiment.stop_early(epoch=epoch)

                            if os.path.exists(args["--model-file"]):
                                model.load_state_dict(torch.load(args["--model-file"]))
                                model = model.to(device)
                            return model, history

                        lr = optimizer.param_groups[0]["lr"] * float(args["--lr-decay"])

                        model.load_state_dict(torch.load(args["--model-file"]))
                        model = model.to(device)

                        optimizer.load_state_dict(torch.load(args["--optim-file"]))

                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        patience = 0

            labels.cpu()
        logger.debug(f"Epoch {epoch} duration {(time.time() - start):.2f}")
    logger.debug(f"Training duration {(time.time() - start_train):.2f}s")

    if os.path.exists(args["--model-file"]):
        model.load_state_dict(torch.load(args["--model-file"]))
        model = model.to(device)
    return model, history


def evaluate(model, validation_instances, batch_size, device):
    was_training = model.training
    metrics = {}

    pred_labels = []
    true_labels = []
    cum_loss = 0
    cum_sents = 0
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        for sents, labels in get_minibatches(validation_instances, batch_size):
            labels = torch.tensor(labels).type(torch.FloatTensor).to(device=device)
            scores = model(sents)
            losses = criterion(scores, labels)

            cum_loss += losses.item()
            cum_sents += len(sents)
            pred_labels.extend(scores.flatten().tolist())
            true_labels.extend(labels.flatten().tolist())
            labels.cpu()

    if was_training:
        model.train()

    roc_auc_macro = roc_auc_score(true_labels, pred_labels, average="macro")
    pr_auc_macro = average_precision_score(true_labels, pred_labels, average="macro")

    roc_auc_micro = roc_auc_score(true_labels, pred_labels, average="micro")
    pr_auc_micro = average_precision_score(true_labels, pred_labels, average="micro")

    roc_auc_weighted = roc_auc_score(true_labels, pred_labels, average="weighted")
    pr_auc_weighted = average_precision_score(
        true_labels, pred_labels, average="weighted"
    )

    metrics["roc_auc_macro"] = roc_auc_macro
    metrics["pr_auc_micro"] = pr_auc_micro

    metrics["roc_auc_micro"] = roc_auc_micro
    metrics["pr_auc_macro"] = pr_auc_macro

    metrics["roc_auc_weighted"] = roc_auc_weighted
    metrics["pr_auc_weighted"] = pr_auc_weighted

    return metrics, cum_loss / cum_sents


def test(args, device):
    def prepare_data(test_data_sents, test_data_labels):
        sentences, labels = read_test_corpus(test_data_sents, test_data_labels, limit=100)
        sentences = NLPUtils.preprocess_pipeline(sentences)
        sentences = np.array(sentences, dtype=object)
        labels = np.array(labels)
        instances = [sentences, labels]
        return instances

    def load_model(params_save_path, model_save_path):
        with open(params_save_path, "rb") as paramsfp:
            vocab, stored_args = pickle.load(paramsfp)

        model = ToxicCommentClassifier(
            int(stored_args["--embedding-size"]),
            int(stored_args["--hidden-size"]),
            int(stored_args["--lstm-num-layers"]),
            int(stored_args["--output-size"]),
            vocab,
            float(stored_args["--dropout-rate"]),
        )
        model.load_state_dict(torch.load(model_save_path))
        model.to(device=device)
        return model

    start = time.time()
    instances = prepare_data(args["--test-data-sents"], args["--test-data-labels"])
    logger.debug(f"Data prepration duration {(time.time() - start):.2f}s")

    logger.debug("Loading saved model state...")
    model = load_model(args["--params-file"], args["--model-file"])

    logging.debug("Evaluation started")
    start = time.time()
    with experiment.test():
        results, loss = evaluate(model, instances, int(args["--batch-size"]), device)
        logger.debug(f"Evaluation duration {(time.time() - start):.2f}s")
        logger.debug(results)

        for metric in results:
            experiment.log_metric(metric, results[metric])
    return metric


def plot_history(history, save_dir):
    df = pd.DataFrame.from_dict(history)

    if df.shape[0] == 0:
        return

    experiment.log_table(filename="experiments.csv", tabular_data=df)
    df.to_csv(os.path.join(save_dir, "experiments.csv"))

    plot = df.plot(
        title="Train/Validation loss",
        x="epoch",
        y=["val_loss", "train_loss"],
        xlabel="Epoch",
        ylabel="Loss",
        colormap="jet",
        figsize=(12, 6),
    )
    fig = plot.get_figure()
    fig.savefig(os.path.join(save_dir, "train_val_loss_epoch.png"))
    experiment.log_figure(figure_name="train_val_loss_epoch")

    plot = df.plot(
        title="Train/Validation loss",
        x="train_iter",
        y=["val_loss", "train_loss"],
        xlabel="Train iteration",
        ylabel="Loss",
        colormap="jet",
        figsize=(12, 6),
    )
    fig = plot.get_figure()
    fig.savefig(os.path.join(save_dir, "train_val_loss_train_iter.png"))
    experiment.log_figure(figure_name="train_val_loss_train_iter")

    plot = df.plot(
        title="Validation scores",
        x="train_iter",
        subplots=True,
        y=["roc_auc_macro", "pr_auc_macro"],
        xlabel="Train iteration",
        ylabel="score",
        colormap="jet",
        figsize=(12, 6),
    )

    fig = plot[0].get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "validation_scores_macro.png"))
    experiment.log_figure(figure_name="validation_scores_macro")

    plot = df.plot(
        title="Validation scores",
        x="train_iter",
        subplots=True,
        y=["roc_auc_micro", "pr_auc_micro"],
        xlabel="Train iteration",
        ylabel="score",
        colormap="jet",
        figsize=(12, 6),
    )

    fig = plot[0].get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "validation_scores_micro.png"))
    experiment.log_figure(figure_name="validation_scores_micro")

    plot = df.plot(
        title="Validation scores",
        x="train_iter",
        subplots=True,
        y=["roc_auc_weighted", "pr_auc_weighted"],
        xlabel="Train iteration",
        ylabel="score",
        colormap="jet",
        figsize=(12, 6),
    )

    fig = plot[0].get_figure()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "validation_scores_weighted.png"))
    experiment.log_figure(figure_name="validation_scores_weighted")


def main():
    args = docopt(__doc__)

    save_dir = args["--save-dir"]
    disable_cuda = args["--disable-cuda"]
    seed = int(args["--seed"])
    verbose = args["--verbose"]

    args["--model-file"] = os.path.join(save_dir, args["--model-file"])
    args["--optim-file"] = os.path.join(save_dir, args["--optim-file"])
    args["--params-file"] = os.path.join(save_dir, args["--params-file"])

    if not verbose:
        logger.setLevel(logging.INFO)

    if not disable_cuda and torch.cuda.is_available():
        logger.debug("Cuda device is enabled")
        device = torch.device("cuda:0")
    else:
        logger.debug("Cuda device is disabled")
        device = torch.device("cpu")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log experiment parameters
    experiment.log_parameters(args)

    reset_seeds(seed)
    if args["train"]:
        logger.debug("Training...")
        model, history = train(args, device)
        plot_history(history, save_dir)
        if os.path.exists(args["--model-file"]):
            experiment.log_asset(args["--model-file"], file_name="model")
            experiment.log_asset(args["--params-file"], file_name="model.params")
    elif args["test"]:
        logger.debug("Testing...")
        test(args, device)


if __name__ == "__main__":
    main()
