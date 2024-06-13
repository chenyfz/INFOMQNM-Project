# INFOMQNM Project

### Env and Dependencies
we use `pipenv`, for dependencies and python version, see `pipfile`

### Dataset extraction
The dataset will not be uploaded to GitHub because they are private sensitive data.
Therefore, after clone the git repo, you should:
1. Create a folder named `dataset-zip` in the root directory. 
2. Place the `k-emophone.zip` file inside this folder.
3. Create an empty `dataset` folder in the root directory.
4. Run `dataset_unzip.py` to unzip all data to `dataset` folder

The `dataset-zip` and `dataset` folders are ignored by git.

### Procedure
1. Run the `timeframes_extraction.py` to extract social media usage timeframes for all participants. Note: some step of timeframes extraction is done in the `feature_extraction.py`.
2. Run the `feature_extraction.py` to preprocess data and extract feature for both HRV and EDA. Note that the warning for EDA processing is only because the data sample rate is less than 6Hz.
3. The `hrv_analysis.ipynb` contains the results of HRV data analysis
4. The `eda_analysis.ipynb` contains the results of EDA data analysis.
5. If you want to do the multivariate analysis in `multivariate_analysis.ipynb`, you have to execute the aforementioned two jupyter notebooks first to have the result files for HRV and EDA.

⚠️ The execution of `dataset_unzip.py` and `feature_extraction.py` uses multiprocessing (all cores). It is normal that your computer will become slow during processing. 