# Music Popularity Analysis
This project looks at what makes a song popular, using a Spotify songs dataset. Each track has audio features such as danceability, energy, loudness, tempo and valence, plus lyrics and playlist metadata. After exploratory data analysis and preprocessing, track popularity is grouped into two classes, low (0–50) and high (51–100). We then apply data mining techniques to this dataset: clustering with KMeans and DBSCAN, outlier detection with Isolation Forest and Local Outlier Factor, feature selection with Recursive Feature Elimination, and classification with tuned Random Forest and SVM models that predict whether a song will be popular.


## Main Folders and files in root directory
- **data/** - contains raw and processed datasets, inside folders callled raw and preprocessed respectively. The precocessed data is saved here from running the EDA juptyter notebook.
- **EDA/** - contains a exploratory data analysis notebook, which has visualizations and insights about the dataset, along with code to generate the preprocessed data.
- **final_project/** - main project folder containing all scripts for classification, clustering, feature selection, and data preprocessing. The main.ipynb file contains the documents the end-to-end 
workflow (code cells, figures, metrics, and brief interpretations for each 
task). The other folders here contain scripts for different tasks, and these folders are described below.:
    - **classification/**- contains scripts for classification tasks including model evaluation and visualization, preparing train/test data and hyperparameter tuning.
    - **cluster_analysis/** - contains scripts for DBSCAN and KMeans clustering, along with visualization, and silhouette score computation.
    - **feature_selection/** - contains script for feature selection using Recursive Feature Elimination.
    - **get_data/** - contains script for different data loading. For eg, enables loading dataset with or without target variable.
    - **outlier_detection/** - contains scripts for outlier detection using Isolation Forest and visualization of outliers.
- **requirements.txt** - lists all required packages for the project.
- **.gitignore** - specifies files and directories to be ignored by git

---

## Setup and running the project instructions


All required packages are listed in `requirements.txt`, and can be installed via:

```bash
pip install -r requirements.txt
```

Then all the final project work can be found and run using the `final_project/main.ipynb` notebook. 

