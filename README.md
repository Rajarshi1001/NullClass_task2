## NullClass Task 2

This task demands implementation of beam search decoding for enchanching translations using the same pretrained LSTM model from a user passed english sentence to french. The beam-search algorithm is implemented in the notebook.The model params and configurations are saved in the folder `english_to_french_lstm_model`. The two tokenizers are saved in `english_tokenizer.json` and `french_tokenizer.json`.

In order to run the notebook, follow the steps:

1. Create a conda environment

```bash
conda create --name nullclass python=3.9
```
2. Activate the environment

```bash
conda activate nullclass
```
3. Install cudnn plugin
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

4. Install tensorflow
```bash
pip install --upgrade pip
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11" 
```

The same environment `nullclass` can be used for running notebooks for other tasks as well. Now run the notebook named `task2.ipynb`.