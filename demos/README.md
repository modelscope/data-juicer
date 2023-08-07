# Demos

This folder contains some demos, which allow users to easily experience the basic functions and tools of Data-Juicer.

## Usage

Use `app.py` in the subdirectory of `demos` to run the demos.

```shell
cd <subdir_of_demos>
streamlit run app.py
```

## Available Demos

- Data (`data`)
  - This folder contains some sample datasets.

- Data process loop (`data_process_loop`)
  - This demo analyzes and processes a dataset, providing a comparison of statistical information before and after the processing.

- Data visualization diversity (`data_visualization_diversity`)
  - This demo analyzes the verb-noun structure of the SFT dataset and plots its diversity in sunburst format.

- Data visualization op effect (`data_visualization_op_effect`)
  - This demo analyzes the statistics of dataset, and displays the effect of each Filter op by setting different thresholds.

- Data visualization statistics (`data_visualization_statistics`)
  - This demo analyzes the dataset and obtain up to 13 statistics.

- Text quality classifier (`tool_quality_classifier`)
  - This demo provides 3 text quality classifier to score the dataset.

- Dataset splitting by language (`tool_dataset_splitting_by_language`)
  - This demo splits a dataset to different sub-datasets by language.

## Demos Coming Soon
- Overview scan
- Auto evaluation helm
- Data mixture
- SFT data zh
- Process sci data
- Process code data
- Data process hpo
