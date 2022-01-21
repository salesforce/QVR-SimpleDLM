'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import shutil
import pandas as pd
import pickle
from scipy.special import expit

from utils import dbscan_clustering, group_bbox_by_axis_dbscan, intersection_over_batch_word_area
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from data import QADataset
from modeling import QueryValueRetrieval_SimpleDLM, QueryValueRetrieval_LayoutLM, QueryValueRetrieval_Bert, LayoutlmConfig, SimpleDLMConfig

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, QueryValueRetrieval_Bert, BertTokenizer),
    "layoutlm": (LayoutlmConfig, QueryValueRetrieval_LayoutLM, BertTokenizer),
    "simpledlm": (SimpleDLMConfig, QueryValueRetrieval_SimpleDLM, BertTokenizer),
}

words_grouped_image = {}
words_grouped_improved = {}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(
    args, train_dataset, model, tokenizer, labels, pad_token_label_id
):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=args.log_name + os.path.basename(args.output_dir))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=None,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num downstreaming_tasks = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            input_ids = batch[0].to(args.device)
            label_ids = batch[3].to(args.device)
            tokens = batch[6]

            inputs = {
                "input_ids": input_ids,
                "attention_mask": batch[1].to(args.device),
                "labels": label_ids,
            }
            if args.model_type in ["layoutlm", "simpledlm"]:
                inputs["bbox"] = batch[4].to(args.device)
            inputs["token_type_ids"] = (
                batch[2].to(args.device)
            )

            inputs["query_token_nums"] = tokens[0]

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank in [-1, 0] and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(
                            args,
                            model,
                            tokenizer,
                            pad_token_label_id,
                            mode=args.eval_mode
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def compute_eval_metrics(args, eval_loss, preds_info, mode, prefix):
    def _load_gt_pairs(label_file_path):
        """Load query-value ground-truth"""
        query_labels = dict()
        with open(label_file_path, encoding="utf-8") as fl:
            for lline in fl:
                lsplits = lline.split("\t")
                if lsplits[0] in ['', ' ']:
                    continue

                if len(lsplits) < 2:
                    continue
                fname = lsplits[-1].strip()
                query_label = lsplits[2].strip()

                if not fname in query_labels:
                    query_labels[fname] = dict()
                if not query_label in query_labels[fname]:
                    query_labels[fname][query_label] = []
                query_labels[fname][query_label].append(lsplits[0])

        return query_labels

    def _calc_query_accuracy(pred_pairs, query_to_eval, gt_pairs, case_sensitive=False):
        """Calculate TP, FP, FN based on exact string matching"""
        tp, fp, fn = 0, 0, 0

        if (query_to_eval not in pred_pairs) and (query_to_eval not in gt_pairs):
            return tp, fp, fn

        found = False

        for query in pred_pairs:

            if query == query_to_eval:
                if query_to_eval not in gt_pairs:
                    fp += 1
                    continue

                query_labels = gt_pairs[query_to_eval]
                query_labels = [a.replace('"', '') for a in query_labels]
                query_labels = [a.replace(' ', '') for a in query_labels]
                query_labels = [a.replace('\n', '') for a in query_labels]
                query_labels = [a.replace('\r', '') for a in query_labels]
                query_labels = [a.replace(',', '') for a in query_labels]

                text = pred_pairs[query_to_eval]['value'].replace('"', '').replace(' ', '')
                text = text.replace('\n', '').replace('\r', '').replace(',', '')

                if not case_sensitive:
                    query_labels = [a.lower() for a in query_labels]
                    text = text.lower()

                if text in query_labels:
                    tp += 1
                    found = True
                else:
                    fp += 1

        if not found:
            if query_to_eval in gt_pairs:
                query_labels = gt_pairs[query_to_eval]
                query_labels = [a.replace('"', '') for a in query_labels]
                query_labels = [a.replace(' ', '') for a in query_labels]
                query_labels = [a.replace('\n', '') for a in query_labels]
                query_labels = [a.replace('\r', '') for a in query_labels]
                query_labels = [a.replace(',', '') for a in query_labels]

                if (len(query_labels) == 0) and (query_to_eval not in pred_pairs):
                    tp += 1
                else:
                    fn += 1
        return tp, fp, fn

    def _calc_end2end_F1(query_eval_list, pred_pairs_all, gt_pairs_all):
        """Evaluation using query-level precision, recall F1"""
        tps = 0
        fps = 0
        fns = 0

        for image_path in pred_pairs_all:
            if image_path not in gt_pairs_all:
                continue

            gt_pairs = gt_pairs_all[image_path]
            pred_pairs = pred_pairs_all[image_path]

            for query_to_eval in query_eval_list:
                tp, fp, fn = _calc_query_accuracy(pred_pairs, query_to_eval, gt_pairs)
                tps += tp
                fps += fp
                fns += fn

        results = {'precision': {}, 'recall': {}, 'f1 score': {}}
        precision = tps / max(1e-10, tps + fps)
        recall = tps / max(1e-10, float(tps+fns))
        f1 = 2. * precision * recall / max(1e-10, precision + recall)
        results['precision'] = precision
        results['recall'] = recall
        results['f1 score'] = f1

        return results

    def _eval_query_pred(args, eval_loss, preds_info, mode, prefix):
        # load ground-truth
        label_file_path = os.path.join(args.data_dir, "{}_labels.txt".format(mode))
        gt_pairs_all = _load_gt_pairs(label_file_path)

        query_eval_list = []
        for fn in preds_info:
            query_eval_list.extend([k for k in preds_info[fn]])
        query_eval_list = list(set(query_eval_list))

        end2end_results = _calc_end2end_F1(query_eval_list, preds_info, gt_pairs_all)

        results = {
            "loss": eval_loss,
            "precision": end2end_results['precision'],
            "recall": end2end_results['recall'],
            "f1": end2end_results['f1 score'],
        }

        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    results = _eval_query_pred(args, eval_loss, preds_info, mode, prefix)
    return results

def improve_word_grouping_splitKV(words_group_input, split_indicator):
    """ Improve grouped words by spliting potential key-value """
    words_group = words_group_input.copy()
    recognized_strings = list(words_group['recognized_string'])
    idx_to_drop = []
    for i, ph in enumerate(recognized_strings):
        x1 = words_group.loc[i, 'x1']
        x2 = words_group.loc[i, 'x2']
        y1 = words_group.loc[i, 'y1']
        y2 = words_group.loc[i, 'y2']
        detection_score = words_group.loc[i, 'score']

        for sind in split_indicator:
            index = ph.lower().find(sind.lower())

            if index < 0:
                continue
            index += len(sind)

            if index >= len(ph):
                continue

            new_string1 = ph[0:index].strip()
            new_string2 = ph[index:].strip()

            width = x2 - x1
            x1_new = x1 + float(len(new_string1)) / len(ph) * width
            bbox1 = [x1, y1, x1_new, y2]
            bbox2 = [x1_new, y1, x2, y2]
            words_group = words_group.append({'x1': bbox1[0],
                                                      'y1': bbox1[1],
                                                      'x2': bbox1[2],
                                                      'y2': bbox1[3],
                                                      'recognized_string': new_string1,
                                                      'score': 0.0}, ignore_index=True)

            words_group = words_group.append({'x1': bbox2[0],
                                                            'y1': bbox2[1],
                                                            'x2': bbox2[2],
                                                            'y2': bbox2[3],
                                                            'recognized_string': new_string2,
                                                            'score': detection_score}, ignore_index=True)
            if not i in idx_to_drop:
                idx_to_drop.append(i)

    words_group = words_group.drop(idx_to_drop)
    words_group.index = range(len(words_group.index))

    return words_group

def prepare_group_data(words, actual_bboxes, scores, axis_keys = ['x1', 'y1', 'x2', 'y2'], dbscan_factor=25.0):

    def _combine_row_df(row_df, axis_keys, x_group_thred = 2.0, y_group_thred = 0.5):
        """
            Aggregate strings in the same row into a single string
            Consider horizontal distance as a threshold
        """

        row_bbox = row_df.agg({axis_keys[0]: min,
                               axis_keys[1]: min,
                               axis_keys[2]: max,
                               axis_keys[3]: max})
        if row_bbox.shape[0] == 1:
            row_bbox['recognized_string'] = row_df.apply(lambda df: ' '.join(
                df.sort_values(axis_keys[0])['recognized_string']))
            row_bbox['score'] = row_df['score'].agg(max)
        else:
            cols = ['x1', 'y1', 'x2', 'y2',
                    'recognized_string', 'score']
            data_list = []
            for i, item in row_df:
                tmp_df = row_df.get_group(i).sort_values(axis_keys[0])
                len_tmp_df = tmp_df.shape[0]
                tmp_strs = []
                tmp_pos = []
                tmp_scores = []
                index = -1
                for j, row in tmp_df.iterrows():
                    index += 1

                    if index == 0:
                        tmp_str = row['recognized_string']
                        x1 = row['x1']
                        y1 = row['y1']
                        x2 = row['x2']
                        y2 = row['y2']
                        score = row['score']
                        row_prev = row.copy()
                    else:
                        if np.abs(row['x1'] - row_prev['x2']) < x_group_thred * min(row['x2'] - row['x1'],
                                                                          row_prev['x2'] - row_prev['x1']) \
                                and np.abs(row['y1'] - row_prev['y1']) < y_group_thred * min(row['y2'] - row['y1'],
                                                                                   row_prev['y2'] - row_prev['y1']):
                            tmp_str += ' ' + row['recognized_string']
                            score = max(score, row['score'])
                            x2 = max(x2, row['x2'])
                            y2 = max(y2, row['y2'])
                        else:
                            tmp_strs.append(tmp_str)
                            tmp_pos.append([x1, y1, x2, y2])
                            tmp_scores.append(float(score))
                            tmp_str = row['recognized_string']
                            x1 = row['x1']
                            y1 = row['y1']
                            x2 = row['x2']
                            y2 = row['y2']
                            score = row['score']
                        row_prev = row.copy()
                    if index == len_tmp_df - 1:
                        tmp_strs.append(tmp_str)
                        tmp_pos.append([x1, y1, x2, y2])
                        tmp_scores.append(float(score))

                for k in range(len(tmp_strs)):
                    tmp_data = [tmp_pos[k][0], tmp_pos[k][1], tmp_pos[k][2],
                                tmp_pos[k][3], tmp_strs[k], tmp_scores[k]]
                    data_list.append(tmp_data)
            row_bbox = pd.DataFrame(data_list, columns=cols)
        return row_bbox

    def _process_cluster_keys(cluster_df, axis_keys):
        # Aggregate text boxes that are in the same text row
        cluster_df = group_bbox_by_axis_dbscan(
            cluster_df.copy(), 1)
        row_groups = cluster_df.groupby(['row_id'])
        grouped_row_results = _combine_row_df(row_groups, axis_keys)
        pieces = [x.to_frame().T for i, x in grouped_row_results.iterrows()]
        return pieces

    pd_data = pd.DataFrame({axis_keys[0]: np.array(actual_bboxes)[:,0],
                                axis_keys[1]:np.array(actual_bboxes)[:,1],
                                axis_keys[2]: np.array(actual_bboxes)[:,2],
                                axis_keys[3]:np.array(actual_bboxes)[:,3],
                                'recognized_string':words,
                                'score': scores})
    # group words horizontally based on their locations
    pd_data['cluster_id'] = dbscan_clustering(
        pd_data, max(max(pd_data[axis_keys[1]]), max(pd_data[axis_keys[3]]))
                   / dbscan_factor, axis_keys)
    words_cluster = pd_data.groupby('cluster_id')

    # group the words in the same rows
    words_rows = []
    for name, df in words_cluster:
        words_rows.extend(_process_cluster_keys(df, axis_keys))
    words_grouped = pd.concat(words_rows)
    words_grouped.index = range(len(words_grouped.index))
    return words_grouped

def post_processing_w_distance(preds_info, scores_pos, label_ids, filenames, query_words, tokens, actual_boxes,
                               pad_token_label_id):
    """
    Postprocessing: group OCR words to phrases based on their locations and assign prediction scores for each group
    """
    batch_size = label_ids.shape[0]
    # use global parameters to avoid redundant computation
    global words_grouped_image
    global words_grouped_improved

    for bs in range(0, batch_size):
        fn = filenames[bs]
        query = query_words[bs]
        query_token_num = tokens[0][bs]
        words = []
        actual_boxes_tmp = []
        scores_pos_tmp = []
        for i_ in range(label_ids.shape[1]):
            if i_ - query_token_num + 1 < 0:
                # skip if a token is from the query
                continue
            if label_ids[bs][i_] == pad_token_label_id:
                #skip if a token is not the first token split from a word
                continue
            actual_boxes_tmp.append(actual_boxes[bs][i_])
            scores_pos_tmp.append(scores_pos[bs][i_])
            words.append(tokens[i_-query_token_num+1][bs])
        actual_boxes_tmp = np.array(actual_boxes_tmp)
        scores_pos_tmp = np.array(scores_pos_tmp)

        # group OCR words to value candidates based on locations
        # words_grouped_image is used to avoid repeated computation for the same image
        if not fn in words_grouped_image:
            words_grouped = prepare_group_data(words, actual_boxes_tmp, scores_pos_tmp)
            words_grouped_image[fn] = words_grouped
        else:
            words_grouped = words_grouped_image[fn]
        # improve words grouping by split potential key out
        if (fn, tuple([query])) not in words_grouped_improved:
            words_grouped = improve_word_grouping_splitKV(words_grouped, [query])
            words_grouped_improved[(fn, tuple([query]))] = words_grouped
        else:
            words_grouped = words_grouped_improved[(fn, tuple([query]))]

        grouped_boxes = words_grouped[['x1', 'y1', 'x2', 'y2']]
        grouped_boxes = np.array(grouped_boxes).reshape(-1, 4)
        # assign OCR words to groupped boxes by measuring intersection over union
        ious = intersection_over_batch_word_area(actual_boxes_tmp, grouped_boxes)
        grouped_scores = np.zeros(ious.shape[1])
        for c_ in range(0, ious.shape[1]):
            if len(scores_pos_tmp[ious[:, c_]>0.7]) == 0:
                continue
            grouped_scores[c_] = np.max(scores_pos_tmp[ious[:, c_]>0.7])

        value_idx = np.argmax(grouped_scores)
        pred_value = words_grouped['recognized_string'][value_idx]
        pred_bbox = grouped_boxes[value_idx, :]
        if not fn in preds_info:
            preds_info[fn] = dict()
        if not query in preds_info[fn]:
            preds_info[fn][query] = dict()
        preds_info[fn][query]['value'] = pred_value
        preds_info[fn][query]['value_bbox'] = list(pred_bbox)

    return preds_info, words_grouped_image

def evaluate(args, model, tokenizer, pad_token_label_id, mode, prefix=""):
    """ Evaluate the model """

    eval_dataset = QADataset(args, tokenizer, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=None,
    )

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num downstreaming_tasks = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    preds_info_w_distance = dict()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            # [CLS] query_token 1,...,query_token M, [SEP], OCR token 1,..., OCR token N
            input_ids = batch[0].to(args.device)
            # pairing label of each token, 1 means a part of the target value, 0 otherwise. -100 means ignore
            label_ids = batch[3].to(args.device)
            query_words = batch[7] # query phrases
            filenames = batch[5]
            # tokens[0][:] records the token length of the query phrase including [CLS] and [SEP]
            # tokens[1:513][:] records the OCR tokens from the form
            tokens = batch[6]
            # actual box location in the form corresponding to each token of input_ids
            actual_boxes = batch[8].detach().cpu().numpy()
            inputs = {
                "input_ids": input_ids,
                "attention_mask": batch[1].to(args.device),
                "labels": label_ids,
            }
            if args.model_type in ["layoutlm", "simpledlm"]:
                inputs["bbox"] = batch[4].to(args.device)
            inputs["token_type_ids"] = (
                batch[2].to(args.device)
            )

            inputs["query_token_nums"] = tokens[0]

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = (
                    tmp_eval_loss.mean()
                )  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        scores_pos = expit(logits.detach().cpu().numpy())

        # grouping words to value candidates based on their locations
        preds_info_w_distance, words_grouped_image = post_processing_w_distance(preds_info_w_distance, scores_pos, label_ids, filenames,
                                                       query_words, tokens, actual_boxes, pad_token_label_id)
    # save the final predictions
    with open(os.path.join(args.output_dir, 'final_predictions_{}.pkl'.format(args.test_mode)), 'wb') as f:
        pickle.dump(preds_info_w_distance, f)

    eval_loss = eval_loss / nb_eval_steps

    end2end_results = compute_eval_metrics(args, eval_loss, preds_info_w_distance, mode, prefix)
    print(end2end_results)

    return end2end_results, preds_info_w_distance

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--model_type",
        default='simpledlm',
        type=str,
        required=True,
        choices=['bert', 'layoutlm', 'simpledlm'],
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.9, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=45,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Log every X updates steps."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        default=True,
        type=bool,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        default=True,
        type=bool,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--train_mode", type=str, default="train_qa"
    )
    parser.add_argument(
        "--eval_mode", type=str, default="test_qa"
    )
    parser.add_argument(
        "--test_mode", type=str, default="test_qa"
    )
    parser.add_argument(
        "--log_name", type=str, default="runs"
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
    ):
        if not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )
        else:
            if args.local_rank in [-1, 0]:
                shutil.rmtree(args.output_dir)

    if not os.path.exists(args.output_dir) and args.do_eval:
        raise ValueError(
            "Output directory ({}) must match model path ({}) during inference stage.".format(
                args.output_dir, args.model_name_or_path
            )
        )

    if (
        not os.path.exists(args.output_dir)
        and args.do_train
        and args.local_rank in [-1, 0]
    ):
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log")
        if args.local_rank in [-1, 0]
        else None,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    labels = ['background', 'foreground']
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # print(args)
        train_dataset = QADataset(
            args, tokenizer, pad_token_label_id, mode=args.train_mode
        )
        global_step, tr_loss = train(
            args, train_dataset, model, tokenizer, labels, pad_token_label_id
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # output the model using a fixed epoch
        if (args.local_rank in [-1, 0]):
            results, _ = evaluate(
                args,
                model,
                tokenizer,
                pad_token_label_id,
                mode=args.eval_mode
            )
            output_dir = os.path.join(args.output_dir, "output_model")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            with open(os.path.join(output_dir, 'saved_model_evaluation.txt'), 'w') as f:
                f.write("groupped words f1 is {} in step {}".format(str(results['f1']),
                                                                    str(global_step)))
            logger.info("Saving model checkpoint to %s", output_dir)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(
                args,
                model,
                tokenizer,
                pad_token_label_id,
                mode=args.test_mode,
                prefix=global_step,
            )
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    return results


if __name__ == "__main__":
    main()
