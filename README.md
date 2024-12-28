# Transformer

Pytorch implementation of the paper "[Attention is all you need](https://arxiv.org/pdf/1706.03762)".

## Dataset

The dataset for the English-Spanish translation task was obtained from https://opus.nlpl.eu. The specific dataset is https://opus.nlpl.eu/Books/en&es/v1/Books, but the HuggingFace version can be used.

## Set up

Create a virtual environment: `python3 -m venv env`

Activate using `source env/bin/activate`

Install all requirements with `pip install -r requirements.txt`

Tune the hyperparameters and other settings in config.py

Train the model by running `python3 train.py`