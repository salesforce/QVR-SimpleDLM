# Value Retrieval with Arbitrary Queries for Form-like Documents

## Introduction

Pytorch Implementation of [Value Retrieval with Arbitrary Queries for Form-like Documents](https://arxiv.org/pdf/2112.07820.pdf).

## Environment
```angular2
CUDA="11.0"
CUDNN="8"
UBUNTU="18.04"
```

## Install
~~~bash
bash install.sh
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install .
# under our project root folder
pip install .
~~~

## Data Preparation
Our model is pre-trained on [IIT-CDIP](https://ir.cs.georgetown.edu/downloads/sigir06cdipcoll_v05-with-authors.pdf) dataset, fine-tuned on [FUNSD](https://guillaumejaume.github.io/FUNSD/) train set and evaluated on [FUNSD](https://guillaumejaume.github.io/FUNSD/) test set and [INV-CDIP](https://arxiv.org/pdf/2110.04282.pdf) test set.

* [Download](https://console.cloud.google.com/storage/browser/sfr-qvr-simple-dlm-research/datasets) our processed OCR results of IIT-CDIP with hocr_list_addr.txt and put under PRETRAIN_DATA_FOLDER/.

* [Download](https://console.cloud.google.com/storage/browser/sfr-qvr-simple-dlm-research/datasets) our processed FUNSD and INV-CDIP datasets and put under DATA_DIR/.

## Reproduce Our Results

* Download our model fine-tuned on FUNSD [here](https://console.cloud.google.com/storage/browser/sfr-qvr-simple-dlm-research/models/fine-tuned-model).

* Do inference following

```angular2
# $MODEL_PATH here is where you save the fine-tuned model.
# DATASET_NAME is FUNSD or INV-CDIP.
bash reproduce_results.sh $MODEL_PATH $DATA_DIR/DATASET_NAME
```

* You should get the following results.

|  Datasets        | Precision  | Recall     | F1         |
| ------------- | ---------- | ---------- | ---------- |
| FUNSD   | 60.4     | 60.9     | 60.7     |
| INV-CDIP | 50.5     | 47.6     | 49.0     |

## Pre-training
* You can skip the following steps by downloading our pre-trained SimpleDLM model [here](https://console.cloud.google.com/storage/browser/sfr-qvr-simple-dlm-research/models/pre-trained-model).

* Or download [layoutlm-base-uncased](https://drive.google.com/drive/folders/1Htp3vq8y2VRoTAwpHbwKM0lzZ2ByB8xM).

* Do pre-training following

```angular2
# $NUM_GPUS is the number of gpus you want to do the pretraining on. To reproduce the paper's results we recommend to use 8 gpus.
# $MODEL_PATH here is where you save the LayoutLM model.
# $PRETRAIN_DATA_FOLDER is the folder of IIT-CDIP hocr files.

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS pretraining.py \
--model_name_or_path $MODEL_PATH  --data_dir $PRETRAIN_DATA_FOLDER \
--output_dir $OUTPUT_DIR

```

## Fine-tuning

* Do fine-tuning following

```angular2
# $MODEL_PATH is where you save the pre-trained simpleDLM model.

CUDA_VISIBLE_DEVICES=0 python run_query_value_retrieval.py --model_type simpledlm --model_name_or_path $MODEL_PATH \
--data_dir $DATA_DIR/FUNSD/ --output_dir $OUTPUT_DIR --do_train --evaluate_during_training
```

## Citation
If you find this codebase useful, please cite our paper:

``` latex
@article{gao2021value,
  title={Value Retrieval with Arbitrary Queries for Form-like Documents},
  author={Gao, Mingfei and Xue, Le and Ramaiah, Chetan and Xing, Chen and Xu, Ran and Xiong, Caiming},
  journal={arXiv preprint arXiv:2112.07820},
  year={2021}
}
```

## Contact
Please send an email to mingfei.gao@salesforce.com or lxue@salesforce.com if you have questions.