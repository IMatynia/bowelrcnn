# Bowel-model-2024
BowelRCNN

# Python environment preparation

Recomended python version is 3.12+. To set up the python venv use the following commands:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Config preparation

You can get configuration files from `configs/` folder or `experiments/auto/configs`. Best model's config can be found under `configs/best.json`. 

# Dataset preparation

Firstly, download bowelsound dataset from [kaggle.com](https://www.kaggle.com/datasets/robertnowak/bowel-sounds). To use a custom dataset you need to modify the config's dataset section. There you need to specify which files from the dataset/raw folder correspond to which set.

After downloading, prepare a dataset folder with the following structure:
```
./dataset/
    raw/ <- put all .wav and .csv files here
```
When the dataset is ready, preprocess the data using the following command:
```sh
python -m src.new.runners.process_data --config ./config.json --data-root ./dataset
```
> Warning, different configs can generate different processed data (audio_properties and dataset sections of the config determine the behaviour). 

# Training

To train each model use the following command:

## Time window classification model
```sh
python3 -m src.new.runners.train_model --config ./config.json --data-root ./dataset --model-to-train classification_model --model-output ./classification_model_weights.h5 --wandb
```

## Time window scaling regression model
```sh
python3 -m src.new.runners.train_model --config ./config.json --data-root ./dataset --model-to-train pattern_model --model-output ./pattern_model_weights.h5 --wandb
```

`--wandb` is responsible for the optional weights and biases integration. Both models are required for the predictions to work.

# Predictions

To run predictions on a WAV file `example.wav` run the following command:

```sh
python3 -m src.new.runners.model_inference --config config.json --wav-file example.wav --output example.csv --pattern-model-weights ./pattern_model_weights.h5 --classification-model-weights ./classification_model_weights.h5 --detection-treshold 0.9 --min-vote-fraction 0.1 --region-overlap 25
```

Optionally you can use a spectrogram dump from the generated dataset by instead of pointing to a WAV file, call `--spectrofgram-dump example.bin`. Spectrogram dump must match config's properties however.

# Testing

Testing and statistic gathering is done on a per-csv-file basis. You need to first generate appropriate predictions into a CSV file to use them here afterwards.

```
python3 -m src.new.runners.test_model --config config.json --mode basic_only --output ./statistics --wav-file example.wav --predictions example.csv --ground-truth example_truth.csv
```

This command will only produce a json with basic statistics and metrics for the given CSV file. To generate predefined point-of-interest graphs one must also include `--wav-file`, `--spectrogram-dump` and `--mode visuals_only` flags.

# Running experiments
> Warning, running all experiments at once takes multiple hours, as multiple neural networks have to be trained and anylized. Consider selecting a subset of models/experiments to run using call arguments.

Skippable experiments:
```py
[
    "data_gen", # Dataset generation
    "training", # Models training
    "pred_param_experiment_preds", # Calculating predictions at different predictions parameters for heatmap
    "preds_base", # Calculating predictions for all models
    "meta_algorithm", # Creating predictions for meta algorithm
    "testing_basic", # Calculate basic merics for all models
    "testing_pred_param_experiment", # Calculate basic metrics for predictions from the prediction parameter experiment
    "testing_basic_additional", # Calculate basic stats for additional models (Reference model, meta algorithm)
    "basic_stats_summary", # Summarize basic stats into a CSV tables
    "testing_visualizations", # Draw graphs of selected points of interest and save them
    "pred_param_experiment_heatmap", # Draw prediction parameter experiment results heatmap
    "model_comparison" # Compare best model against reference model's predictions on a sound-by-sound basis. Save findings to a txt and json file.
]
```
Example call that skips the most time and memory intensive parts of the experiments
```sh
$ python -m src.new.runners.experiments_runner --skip-experiments data_gen training pred_param_experiment_preds preds_base
```