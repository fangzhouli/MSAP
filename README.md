# Model Selection and Analysis Pipeline

## How to Start

### Dependencies

Python >= 3.10

### Installation

```console
git clone https://github.com/fangzhouli/MSAP.git
cd MSAP
pip install -e .
pip install -r requirements.txt
```

### How to Use

#### Model Selection

```console
python -m msap.run_model_selection \
    {Input file path} \
    {Output file path} \
    {Preprocessed data directory path} \
    {Target variable column} \
    --feature-kfold {K-fold split variable column} \
    --load-data-preprocessed {Load existing preprocessed data} \
    --random-state {Random state}
```

Arguments:
- Input file path: Path to the input file. The input file should be a CSV file containing both features and target variable.
- Output file path: Path to the output pickle file. This output file will contain the results of model selection.
- Preprocessed data directory path: Path to the preprocessed data directory. If the preprocessed data directory does not exist, it will be created. For each preprocessing combination, 2 files will be created:
    - Preprocessed data file: Named {scale_mode}_{impute_mode}_{outlier_mode}.csv
    - Outlier indices file: Named {scale_mode}_{impute_mode}_{outlier_mode}_outlier_indices.txt
- Target variable column: Column name of the target variable in the input data file.

Options:
- K-fold split variable column: Column name of the variable used for K-fold split. Some datasets prefer k-fold splitting using specific index column. If not specified, normal k-fold split will be used, which splits based on rows.
- Load existing preprocessed data: If True, existing preprocessed data will be loaded. If False, preprocessing will be performed.
- Random state: Random state for reproducibility.

#### Model Analysis

```console
python -m msap.run_model_selection \
    {Model selection result path} \
    {Preprocessed data directory path} \
    {Input file path} \
    {Output directory path} \
    {Target variable column} \
    --feature-kfold {K-fold split variable column} \
    --random-state {Random state}
```

Arguments:
- Model selection result path: Path to the model selection result pickle file.
- Preprocessed data directory path: Path to the preprocessed data directory.
- Input file path: Path to the input file. The input file should be a CSV file containing both features and target variable.
- Output directory path: Path to the output directory. This output directory will contain the results of model analysis. If the output directory does not exist, it will be created.
- Target variable column: Column name of the target variable in the input data file.

Options:
- K-fold split variable column: Column name of the variable used for K-fold split. Some datasets prefer k-fold splitting using specific index column. If not specified, normal k-fold split will be used, which splits based on rows.
- Random state: Random state for reproducibility.
