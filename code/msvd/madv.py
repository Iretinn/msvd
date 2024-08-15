# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing
import models
import utils
from utils import multi_data_loader_st, data_loader
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import auc
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def load_dataset(tokenizer, file_path, args):
    data_dir = args.data_dir
    print(file_path)
    examples = []
    inputs = []
    df = pd.read_csv(os.path.join(data_dir, file_path))
    funcs = df["func"].tolist()
    labels = df["target"].tolist()
    for i in tqdm(range(len(funcs))):
        source_tokens, source_ids = convert_examples_to_features(funcs[i], tokenizer, args)
        examples.append(source_tokens)
        inputs.append(source_ids)
    return np.array(inputs), np.array(labels)


def convert_examples_to_features(func, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_tokens, source_ids


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(model, tokenizer, args):
    """ Train the model """
    # load data (source train, target train, target test)
    source_list = args.source_train_file_list
    source_train_inputs = []
    source_train_labels = []
    input_size = []
    for source_file in source_list:
        inputs, labels = load_dataset(tokenizer, source_file, args)
        source_train_inputs.append(inputs)
        source_train_labels.append(labels)
        input_size.append(len(inputs))
    target_train_inputs, _ = load_dataset(tokenizer, args.target_train_file, args)

    valid_list = args.source_valid_file_list
    valid_inputs = []
    valid_tests = []
    for valid_file in valid_list:
        inputs, labels = load_dataset(tokenizer, valid_file, args)
        valid_inputs.append(inputs)
        valid_tests.append(labels)
    target_test_inputs = np.concatenate(valid_inputs)
    target_test_labels = np.concatenate(valid_tests)

    # target_test_inputs, target_test_labels = load_dataset(tokenizer, args.target_test_file, args)
    input_size.append(len(target_train_inputs))

    # Prepare optimizer and schedule (linear warmup and decay)
    n_batch = int(min(input_size) / args.train_batch_size)
    args.max_steps = args.epochs * n_batch
    args.warmup_steps = args.max_steps // 20
    optimizer_grouped_parameters = model.get_parameters(args, lr=args.learning_rate)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # Train!
    number_domain = args.number_domain
    logger.info("***** Running training *****")
    for i in range(number_domain):
        source_file_name = os.path.basename(source_list[i])
        logger.info("  Num examples of %s source data = %d", source_file_name, len(source_train_inputs[i]))
    logger.info("  Num examples of target data = %d", len(target_train_inputs))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    best_f1 = 0
    model.to(args.device)
    model.zero_grad()
    stop = 0
    mu = args.transfer_loss_weight
    clf_loss_log = []
    transfer_loss_log = []
    for i in range(number_domain):
        clf_loss_log.append([])
        transfer_loss_log.append([])
    total_loss_log = []
    for idx in range(args.epochs):
        model.train()
        train_loss_clf = []
        train_loss_transfer = []
        for i in range(number_domain):
            train_loss_clf.append(utils.AverageMeter())
            train_loss_transfer.append(utils.AverageMeter())
        train_loss_total = utils.AverageMeter()
        # model.epoch_based_processing(n_batch)

        train_loader = multi_data_loader_st(source_train_inputs, source_train_labels, target_train_inputs, n_batch, args.train_batch_size)
        for sinputs, slabels, tinput in tqdm(train_loader, desc=f"batch", total=n_batch):
            for i in range(number_domain):
                sinputs[i] = torch.tensor(sinputs[i]).to(args.device)
                slabels[i] = torch.tensor(slabels[i]).to(args.device)
            tinput = torch.tensor(tinput).to(args.device)

            clf_losses, transfer_losses = model(sinputs, tinput, slabels)
            loss = torch.mean(clf_losses) + mu * torch.mean(transfer_losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            for i in range(number_domain):
                train_loss_clf[i].update(clf_losses[i].item())
                train_loss_transfer[i].update(transfer_losses[i].item())
                clf_loss_log[i].append(clf_losses[i].item())
                transfer_loss_log[i].append(transfer_losses[i].item())
            train_loss_total.update(loss.item())
            total_loss_log.append(loss.item())

        info = 'Epoch: [{:2d}/{}], total_Loss: {:.4f}'.format(idx+1, args.epochs, train_loss_total.avg)
        for i in range(number_domain):
            info += ', domain {}: clf_loss: {:.4f}, transfer_loss: {:.4f}'.format(i+1, train_loss_clf[i].avg, train_loss_transfer[i].avg)

        # Test
        stop += 1
        result = test(args, model, target_test_inputs, target_test_labels)
        info += ', test_loss {:4f}, test_f1: {:.4f}'.format(result["test_loss"], result["test_f1"])

        if best_f1 < result["test_f1"]:
            best_f1 = result["test_f1"]
            logger.info("  " + "*" * 20)
            logger.info("  Best f1:%s", round(best_f1, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-f1-madv'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
            stop = 0
        if 0 < args.early_stop <= stop:
            print(info)
            break
        print(info)
    if args.store_loss:
        logger.info("save loss!")
        with open("./clf_loss_log.txt", 'w') as clf_los:
            for cll in clf_loss_log:
                clf_los.write(str(cll))
                clf_los.write('\n')
        with open("./transfer_loss_log.txt", 'w') as tra_los:
            for tll in transfer_loss_log:
                tra_los.write(str(tll))
                tra_los.write('\n')
        with open("./total_loss_log.txt", 'w') as total_los:
            total_los.write(str(total_loss_log))
    print('Transfer result: {:.4f}'.format(best_f1))


def test(args, model, test_inputs, test_labels, best_threshold=0.5):
    test_data_loader = data_loader(test_inputs, test_labels, args.eval_batch_size, shuffle=False)
    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_inputs))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    y_trues = []
    test_loss = utils.AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    n_batch = math.ceil(len(test_inputs)/args.eval_batch_size)
    for tx, ty in tqdm(test_data_loader, desc=f"test batchs", total=n_batch):
        tx = torch.tensor(tx).to(args.device)
        ty = torch.tensor(ty).to(args.device)
        with torch.no_grad():
            clf = model.predict(tx)
            logit = torch.softmax(clf, dim=-1)
            loss = criterion(clf, ty)
            test_loss.update(loss.item())
            logits.append(logit.cpu().numpy())
            y_trues.append(ty.cpu().numpy())
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    rocauc = roc_auc_score(y_trues, logits[:, 1])
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_auc": float(rocauc),
        "test_threshold": best_threshold,
        "test_loss": test_loss.avg
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return result


def get_parser():
    parser = argparse.ArgumentParser()
    # files name and path
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The dir path of data files.")
    parser.add_argument("--source_train_file_list", default=None, nargs='+', type=str, required=False,
                        help="The input source training data file list (one or more csv files).")
    parser.add_argument("--source_valid_file_list", default=None, nargs='+', type=str, required=False,
                        help="The input source training data file list (one or more csv files).")
    parser.add_argument("--target_train_file", default=None, type=str, required=False,
                        help="The input target training data file (a csv file).")
    parser.add_argument("--target_test_file", default=None, type=str,
                        help="The input target test data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # model
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="run training.")
    parser.add_argument("--do_analysis", action='store_true',
                        help="run analysis (t-sne) on well-trained model.")
    parser.add_argument("--do_test", action='store_true',
                        help="run testing on well-trained model.")
    parser.add_argument("--store_loss", action='store_true',
                        help="save training losses.")

    # train set
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--early_stop', type=int, default=15, help="Early stopping")
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")

    # transfer para
    parser.add_argument('--num_class', type=int, default=2,
                        help="number of classes.")
    parser.add_argument('--bottleneck_width', type=int, default=768)
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    return parser


def get_model(args):
    args.number_domain = len(args.source_train_file_list)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = models.MSVD(args)
    return model, tokenizer


def main():
    parser = get_parser()
    args = parser.parse_args()
    # Setup CUDA, GPU
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    set_seed(args)
    model, tokenizer = get_model(args)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train(model, tokenizer, args)

    if args.do_test:
        target_test_inputs, target_test_labels = load_dataset(tokenizer, args.target_test_file, args)
        checkpoint_prefix = f'checkpoint-best-f1-madv/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test(args, model, target_test_inputs, target_test_labels)

    if args.do_analysis:
        pass


if __name__ == "__main__":
    main()
