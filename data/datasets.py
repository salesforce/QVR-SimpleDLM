'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import logging
import os
import random
import itertools

import torch
from torch.utils.data import Dataset
from utils.pretrain_utils import convert_hocr_to_input_example

logger = logging.getLogger(__name__)

class QADataset(Dataset):
    """ The dataset class for the query-value-retrieval task. """

    def __init__(self, args, tokenizer, pad_token_label_id, mode):
        if args.local_rank not in [-1, 0] and "train" in mode:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(args, mode)
            features = convert_QAexamples_to_features(
                examples,
                args.max_seq_length,
                tokenizer,
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and "train" in mode:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

        self.features = features
        file_names = [f.file_name for f in features]
        #[CLS], query_token1,...,query_tokenM, [SEP], ocr_token1,...,ocr_tokenN, [SEP]
        self.all_input_ids = []
        self.all_input_mask = []
        self.all_segment_ids = []
        self.all_bboxes = []
        self.all_label_ids = []
        self.words = []
        self.query_words = []
        self.file_names = []
        self.actual_bboxes = []

        for fi, fn in enumerate(file_names):
            words = self.features[fi].words
            input_ids = self.features[fi].input_ids
            max_sequence_len = len(input_ids)
            input_mask = self.features[fi].input_mask
            segment_ids = self.features[fi].segment_ids
            boxes = self.features[fi].boxes
            actual_boxes = self.features[fi].actual_bboxes
            word_token_mask = self.features[fi].word_token_mask
            query_ids_list = self.features[fi].query_ids_list
            query_bbox_list = self.features[fi].query_bbox_list
            query_words_list = self.features[fi].query_words_list
            # group the same query words (if they are nearby) to a query phrase
            queries_tmp = [' '.join(_) for _ in query_words_list]
            grouped_queries = [list(g) for _, g in itertools.groupby(queries_tmp, lambda x: x)]
            query_start_idx_tmp = 0
            query_intervals = [] # [start, end)
            for gk in  grouped_queries :
                query_intervals.append((query_start_idx_tmp, query_start_idx_tmp + len(gk)))
                query_start_idx_tmp += len(gk)

            for interval_idx, interval in enumerate(query_intervals):
                query = grouped_queries[interval_idx][0]
                if query == 'background':
                    continue
                # start and end indexes of the query word in the query word list
                interval_start, interval_end = interval
                query_ids = query_ids_list[interval_start]
                query_token_num = len(query_ids)
                label_id = [0]*len(input_ids)
                pos_label_idxes_list = [] # target value token indexes list for this query
                for qw_idx in range(interval_start, interval_end):
                    pos_label_idxes_list.append(self.features[fi].label_ids[qw_idx])

                for pos_label_idxes in pos_label_idxes_list:
                    for pi, pos_label_idx in enumerate(pos_label_idxes):
                        # adjust OCR word label position after adding the query at the beginning of the input_ids
                        if pos_label_idx + query_token_num >= len(label_id):
                            continue
                        if pi == 0:
                            # assign label for the first token of a word
                            label_id[pos_label_idx + query_token_num] = 1
                        else:
                            # use pad label for the rest tokens of a word
                            label_id[pos_label_idx + query_token_num] = pad_token_label_id

                for wtm_idx, wtm in enumerate(word_token_mask):
                    # double make sure to use pad label for the rest tokens of a word
                    if wtm == 0 and (wtm_idx + query_token_num) < len(label_id):
                        label_id[wtm_idx + query_token_num] = pad_token_label_id

                # add query number at the beginning for record
                words_pad = [query_token_num]+words
                # add query at the beginning
                all_input_ids = query_ids+input_ids
                all_input_mask = [0]*query_token_num+input_mask
                all_segment_ids = [1]*query_token_num+segment_ids
                all_boxes = query_bbox_list[interval_start]+boxes
                all_actual_boxes = query_bbox_list[interval_start]+actual_boxes

                all_input_ids = [all_input_ids[_] for _ in range(max_sequence_len)]
                all_input_mask = [all_input_mask[_] for _ in range(max_sequence_len)]
                all_segment_ids = [all_segment_ids[_] for _ in range(max_sequence_len)]
                all_boxes = [all_boxes[_] for _ in range(max_sequence_len)]
                all_actual_boxes = [all_actual_boxes[_] for _ in range(max_sequence_len)]

                for _ in range(max_sequence_len):
                    if all_input_mask[_] == 0:
                        label_id[_] = pad_token_label_id

                self.all_input_ids.append(all_input_ids)
                self.all_input_mask.append(all_input_mask)
                self.all_segment_ids.append(all_segment_ids)
                self.all_bboxes.append(all_boxes)
                self.actual_bboxes.append(all_actual_boxes)
                self.all_label_ids.append(label_id)
                self.words.append(words_pad)
                query_phrase = ' '.join(query_words_list[interval_start])
                self.query_words.append(query_phrase)
                self.file_names.append(fn)

        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            self.all_input_ids, dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            self.all_input_mask, dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            self.all_segment_ids, dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            self.all_label_ids, dtype=torch.long
        )
        self.all_bboxes = torch.tensor(self.all_bboxes, dtype=torch.long)
        self.actual_bboxes = torch.tensor(self.actual_bboxes, dtype=torch.long)

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
            self.file_names[index],
            self.words[index],
            self.query_words[index],
            self.actual_bboxes[index],
        )

class MLMDataset_cdip(Dataset):
    """The dataset class for pretraining SimpleDLM with IIT-CDIP dataset.

    Attributes:
        args: arguments passed in.
        tokenizer: the tokenizer to tokenize the text.
        pad_token_label_id: the id you want to assign to the padded token.
        hocr_file_list_addr: the address to the file which contains all the hocr files' addresses used in the pretrain.
    """
    def __init__(self, args, tokenizer, pad_token_label_id, hocr_file_list_addr):
        if args.local_rank not in [-1, 00]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        self.args = args
        self.tokenizer = tokenizer
        self.pad_token_label_id = pad_token_label_id

        with open(os.path.join(args.data_dir, self.args.hocr_file_list_addr), 'r') as f:
            lines = f.readlines()

        self.hocr_file_list = lines

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    def __len__(self):
        return len(self.hocr_file_list)

    def __getitem__(self, index):
        # get the hocr file address
        hocr_file = os.path.join(self.args.data_dir, self.hocr_file_list[index].strip())
        # convert hocr to input example format
        input_example = convert_hocr_to_input_example(hocr_file)
        # convert input example format to features
        features = convert_examples_to_features(
            [input_example],
            self.args.max_seq_length,
            self.tokenizer,
            cls_token_at_end=False,
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=self.pad_token_label_id,
        )

        # convert features to the tensor
        # Convert to Tensors and build dataset
        input_ids = torch.tensor(
            features[0].input_ids, dtype=torch.long
        )
        input_mask = torch.tensor(
            features[0].input_mask, dtype=torch.long
        )
        segment_ids = torch.tensor(
            features[0].segment_ids, dtype=torch.long
        )
        label_ids = torch.tensor(
            features[0].label_ids, dtype=torch.long
        )
        bboxes = torch.tensor(features[0].boxes, dtype=torch.long)

        return (
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            bboxes,
        )


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: list. The labels for each word of the sequence.
            boxes: list. The resized boxes of each word of the sequence.
            actual_bboxes: list. The actual boxes of each word of the sequence.
            file_name: string. The file's name.
            page_size: list. The size of the input page.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes,
        actual_bboxes,
        file_name,
        page_size,
        words,
        word_token_mask=None,
        query_ids_list=None,
        query_words_list=None,
        query_bbox_list=None,
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size
        self.words = words
        self.query_ids_list = query_ids_list
        self.query_words_list = query_words_list
        self.query_bbox_list = query_bbox_list
        self.word_token_mask=word_token_mask


def read_examples_from_file(args, mode):
    data_dir = args.data_dir
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f, open(
        box_file_path, encoding="utf-8"
    ) as fb, open(image_file_path, encoding="utf-8") as fi:
        words = [] # OCR word
        boxes = [] # re-scaled box of an OCR word
        actual_bboxes = [] # actual box in a document of an OCR word
        file_name = None
        page_size = None
        labels = [] # field label/query correspond to an OCR word
        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                            boxes=boxes,
                            actual_bboxes=actual_bboxes,
                            file_name=file_name,
                            page_size=page_size,
                        )
                    )
                    guid_index += 1
                    words = []
                    boxes = []
                    actual_bboxes = []
                    file_name = None
                    page_size = None
                    labels = []
            else:
                splits = line.split("\t")
                bsplits = bline.split("\t")
                isplits = iline.split("\t")
                assert len(splits) == 2
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0]
                
                words.append(splits[0])
                if len(splits) > 1:
                    label = splits[-1].replace("\n", "")
                    labels.append(label)
                    box = bsplits[-1].replace("\n", "")
                    box = [int(b) for b in box.split()]
                    boxes.append(box)
                    actual_bbox = [int(b) for b in isplits[1].split()]
                    actual_bboxes.append(actual_bbox)
                    page_size = [int(i) for i in isplits[2].split()]
                    file_name = isplits[3].strip()
                else:
                    # downstreaming_tasks could have no label for mode = "test"
                    labels.append("background")
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )
    return examples


def convert_QAexamples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token=0,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of InputBatch"""

    features = []
    for (ex_index, example) in enumerate(examples):
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        words = [] # word corresponds to each OCR token
        tokens = [] # OCR token
        token_boxes = [] # re-scaled box corresponds to each OCR token
        actual_bboxes = [] # real box in a form corresponds to each OCR token
        label_indexes = [] # target value (OCR token) indexes correspond to a query phrase in query_tokens_list below
        query_words_list = [] # a list of query words for the image, 'background' will be ignored
        query_tokens_list = [] # a list of query tokens for the image, [CLS] and [SEP] are included
        query_ids_list = [] # ids correspond to query tokens above
        query_bbox_list = [] # dummy locations correspond to query tokens above
        word_token_mask = [] # 1 for the first token of every word, 0 otherwise
        token_count = 0
        for word, label, box, actual_bbox in zip(
                example.words, example.labels, example.boxes, example.actual_bboxes
        ):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            words.extend([word] * len(word_tokens))
            word_token_mask.extend([1]+[0]*(len(word_tokens)-1))
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_indexes.append([_ for _ in range(token_count, token_count+len(word_tokens))])

            query_words = label.split(' ')
            query_words_list.append(query_words)
            query_tokens = []
            for qw in query_words:
                query_tokens.extend(tokenizer.tokenize(qw))
            query_bbox_list.append([cls_token_box]+[[0,0,1000,1000]]*len(query_tokens)+[sep_token_box])
            query_tokens = [cls_token]+query_tokens+[sep_token]
            query_tokens_list.append(query_tokens)
            query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
            query_ids_list.append(query_ids)
            token_count += len(word_tokens)

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            words = words[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
            word_token_mask = word_token_mask[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        words += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        word_token_mask += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        word_token_mask += [0] * padding_length
        token_boxes += [pad_token_box] * padding_length
        actual_bboxes += [pad_token_box] * padding_length
        words += ['na'] * padding_length

        keep_query_idx = [] # only keep if value index is within max_seq_length
        for _, label_idx in enumerate(label_indexes):
            if label_idx[-1] < max_seq_length:
                keep_query_idx.append(_)
        label_indexes = [label_indexes[_] for _ in keep_query_idx]
        query_ids_list = [query_ids_list[_] for _ in keep_query_idx]
        query_words_list = [query_words_list[_] for _ in keep_query_idx]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        assert len(words) == max_seq_length
        assert len(word_token_mask) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_indexes,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
                words=words,
                word_token_mask=word_token_mask,
                query_ids_list=query_ids_list,
                query_words_list=query_words_list,
                query_bbox_list=query_bbox_list,
            )
        )
    return features

def convert_examples_to_features(
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-1,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Convert input example to input feature format which has the padded sequence and tokenized words. """

    features = []
    for (ex_index, example) in enumerate(examples):
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        words = [] # word corresponds to each OCR token
        tokens = [] # OCR token
        token_boxes = [] # re-scaled box corresponds to each OCR token
        actual_bboxes = [] # real box in a form corresponds to each OCR token

        for word_idx, (word, label, box, actual_bbox) in enumerate(zip(
            example.words, example.labels, example.boxes, example.actual_bboxes
        )):

            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            words.extend([word] * len(word_tokens))
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            words = words[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]

        # Get random words for masked language modeling pretraining
        tokens, label_ids = random_word(tokens, tokenizer, pad_token_label_id=pad_token_label_id)

        tokens += [sep_token]
        words += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]

        if sep_token_extra:
            tokens += [sep_token]
            words += [sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            words += [cls_token]
            token_boxes += [cls_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            words = [cls_token] + words
            token_boxes = [cls_token_box] + token_boxes
            actual_bboxes = [[0, 0, width, height]] + actual_bboxes
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
            words = (['na'] * padding_length) + words
            tokens = (['na'] * padding_length) + tokens
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            token_boxes += [pad_token_box] * padding_length
            words += ['na'] * padding_length
            tokens += ['na'] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        assert len(words) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
                words=words,
            )
        )
    return features

def random_word(tokens, tokenizer, pad_token_label_id=-1):
    """ Follow the bert paper, randomly masking the tokens and generate correspond labels for masked language modeling. """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # Mask token with a certain probability, followed bert paper, set the ratio to be 15%.
        ratio = 0.15
        if prob < ratio:
            prob /= ratio

            # Randomly mask 80% of the selected tokens by change the tokens to [MASK] tokens.
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # Randomly change 10% tokens to randomly selected tokens from vocabulary.
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # The unchanged tokens will be ignored by loss function during pretrain.
            output_label.append(pad_token_label_id)

    return tokens, output_label
