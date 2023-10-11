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


### Reproduced Redpajama

We have reproduced the processing flow of some RedPajama datasets. Please refer to the [reproduced_redpajama](reproduced_redpajama) folder for details.

### Reproduced BLOOM

We have reproduced the processing flow of some BLOOM datasets. please refer to the [reproduced_bloom](reproduced_bloom) folder for details.

### Data-Juicer Recipes
We have refined some open source datasets (including CFT datasets) by using Data-Juicer and have provided configuration files for the refined flow. please refer to the [data_juicer_recipes](data_juicer_recipes) folder for details.