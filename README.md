# Grad2Task: Improved Few-shot Text Classification Using Gradients for Task Representation

## Prerequisites
This repo is built upon a local copy of `transformers==2.1.1`.
This repo has been tested on `torch==1.4.0` with `python 3.7` and `CUDA 10.1`.

To start, create a new environment and install: 
```bash
conda create -n grad2task python=3.7
conda activate grad2task
cd Grad2Task
pip install -e .
```

We use [wandb](https://wandb.ai/site) for logging. Please set it up following [this doc](https://docs.wandb.ai/quickstart) and specify your project name on `wandb` in `run_meta_training.sh`:
```bash
export WANDB=[YOUR PROJECT NAME]
```

Download the dataset and unzip it under the main folder: https://drive.google.com/file/d/1uAdgZFYv9epk6tQVQ3SwboxFpSlkC_ZW/view?usp=sharing

If need to place it somewhere else, specify its path in `path.sh`.

# Fine-tuning Baseline

# Train & Evaluation
To train/evaluate models:
```bash
bash meta_learn.sh [MODEL_NAME] [MODE] [EXP_ID]
```
where `[MODEL_NAME]` refers to model name, `[MODE]` is experiment model and `[EXP_ID]` is an optional experiment id used for mark different runs using the same model. Options for `[MODEL_NAM]` and `MODE` are listed as follow: 
| `[MODE]` | Description |
| :---- | :---------- |
| train | Training models. |
| test_best | Test the model with the best validation performance. |
| test_latest | Test the latest checkpoint. |
| test | Test model without meta-training. Only applicable to the `fine-tune-baseline` model. |

| `[MODEL_NAME]` | Description |
| :---- | :---------- |
| fine-tune-baseline | Fine-tuning BERT for each task separately. |
| bert-protonet-euc | ProtoNet with BERT as encoder, using Euclidean distance as distance metric. |
| bert-protonet-euc-bn | ProtoNet with BERT+Bottleneck Adapters as encoder, using Euclidean distance as distance metric. |
| bert-protonet | ProtoNet with BERT as encoder, using cosine distance as distance metric. |
| bert-protonet-bn | ProtoNet with BERT+Bottleneck Adapters as encoder, using cosine distance as distance metric. |
| bert-leopard | Leopard with pretrained BERT [1]. |
| bert-leopard-fixlr | Leopard but with fixed learning rates. |
| bert-cnap-bn-euc-context-cls-shift-scale-ar | Our proposed approach using gradients as task representation. |
| bert-cnap-bn-euc-context-cls-shift-scale-ar-X | Our proposed approach using average input encoding as task representation. |
| bert-cnap-bn-euc-context-cls-shift-scale-ar-XGrad | Our proposed approach using both gradients and input encoding as task representation. |
| bert-cnap-bn-euc-context-cls-shift-scale-ar-XY | Our proposed approach using input and textual label encoding as task representation. |
| bert-cnap-bn-euc-context-shift-scale-ar | Same with our proposed approach except adapting all tokens instead of just the [CLS] token as we do. |
| bert-cnap-bn-pretrained-taskemb | Our proposed approach with pretrained task embedding model. |
| bert-cnap-bn-hyper | A hypernetwork based approach. |

To run a model with different hyperparameters, first name this run by `[EXP_ID]` and then specify the new hyperparameters in `run/meta_learn.sh`. For example, if one wants to run `bert-protonet-euc` with a smaller learning rate, they could modify `run/meta_learn.sh` as:
```bash
...
elif [ $1 == "bert-protonet-bn" ]; then # ProtoNet with cosince distance
    export LEARNING_RATE=2e-5
    export CHECKPOINT_FREQ=1000
    if [ ${EXP_ID} == *"lr1e-5" ]; then
        export LEARNING_RATE=1e-5
        export CHECKPOINT_FREQ=2000
        # modify other hyperparameters here
    fi
...
```
and then run:
```bash
bash meta_learn.sh bert-protonet-bn train lr1e-5
```

# Reference
[1] T. Bansal, R. Jha, and A. McCallum. Learning to few-shot learn across diverse natural language classification tasks. In Proceedings of the 28th International Conference on Computational Linguistics, pages 5108–5123, 2020.