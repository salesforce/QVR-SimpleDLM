'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import numpy as np
import re
from lxml import html
import random

def get_text(node):
    return re.sub(r'\s+', '\x20', node.text_content()).strip()

def get_bbox(node):
    try:
        data = node.attrib['title']
        bboxre = re.compile(r'\bbbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')

        return [int(x) for x in bboxre.search(data).groups()]

    except:
        return []

def filter_helper(texts, bboxes):
    keep = []
    for i, t in enumerate(texts):
        if t == "":
            continue
        keep.append(i)
    texts_ = [texts[_] for _ in keep]
    bboxes_ = [bboxes[_] for _ in keep]

    return texts_, bboxes_

def get_text_n_box_n_page_size_from_hocr(hocr_path):
    try:
        doc = html.parse(hocr_path.strip())

        total_page = len(doc.xpath("//*[@class='ocr_page']"))
        num_page = random.randint(1, total_page)
        page = doc.getroot().find("./body/div/[@id='page_{}']".format(num_page))
        ocr_lines_one_page = []
        for ocr_ele in page.iter():
            if 'class' in ocr_ele.attrib and ocr_ele.attrib['class'] == 'ocr_line':
                ocr_lines_one_page.append(ocr_ele)
        lines = ocr_lines_one_page
        page_box = get_bbox(doc.xpath("//*[@class='ocr_page']")[num_page - 1])
    except:
        return [], [], (2000, 2000)
    width, height = page_box[2], page_box[3]
    texts = []
    bboxes = []
    for line in lines:
        for word in line:
            text = get_text(word)
            bbox = get_bbox(word)
            texts.append(text)
            bboxes.append(bbox)
    return texts, bboxes, (width, height)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev downstreaming_tasks, but not for test downstreaming_tasks.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


def convert_hocr_to_input_example(hocr_file):
    """Convert hocr fiel to input example format.

    Args:
        hocr_file: the addr of the hocr file, str
    return:
        the input example extracted from the hocr file, InputExample
    """
    try:
        texts, bboxes, (page_width, page_height) = get_text_n_box_n_page_size_from_hocr(hocr_file)
        texts, bboxes = filter_helper(texts, bboxes)
    except:
        texts, bboxes, (page_width, page_height) = [], [], (2000, 2000)

    bboxes_np = np.array(bboxes)

    resize_divider = np.array([page_width, page_height, page_width, page_height])

    if bboxes_np.size != 0:
        bboxes_resized = ((bboxes_np / resize_divider) * 1000).astype(int)

        # clip the bboxes
        bboxes_resized = np.clip(bboxes_resized, 0, 1023)

        bboxes_resized = bboxes_resized.tolist()
    else:
        bboxes_resized = bboxes_np.astype(int).tolist()

    labels = ["background"] * len(texts)
    guid = ["0"] * len(texts)
    page_size = [page_width, page_height]

    input_example = InputExample(guid=guid, words=texts, labels=labels, boxes=bboxes_resized,
                                 actual_bboxes=bboxes, file_name=hocr_file, page_size=page_size)

    assert len(texts) == len(bboxes)

    return input_example