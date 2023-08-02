# Config Files

This folder contains some configuration files to allow users to easily understand the configuration methods of various functions and quickly reproduce the processing flow of different datasets.

## Usage

```shell
# To process your dataset.
python tools/process_data.py --config xxx.yaml
# To analyse your dataset.
python tools/analyze_data.py --config xxx.yaml
```

## Categories

The current configuration files are classified into the subsequent categories.

### Demo

Demo configuration files are used to help users quickly familiarize the basic functions of Data-Juicer. Please refer to the [demo](demo) folder for details.


### Redpajama

We have reproduced the processing flow of some redpajama datasets. Please refer to the [redpajama](redpajama) folder for details.

### Bloom

We have reproduced the processing flow of some bloom datasets. please refer to the [bloom](bloom) folder for details.

### Refine_recipe
We have refined some open source datasets (including SFT datasets) by using Data-Juicer and have provided configuration files for the refine flow. please refer to the [refine_recipe](refine_recipe) folder for details.