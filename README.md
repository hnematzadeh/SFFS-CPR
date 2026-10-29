




# SFFS-CPR: Sample-wise Frequency Feature Selection with Congestion-based Re-ranking

Code for the experiments in the paper **"[Paper title]"** ([authors], [venue/year]).

SFFS-CPR is a hybrid of two filters feature selection method for small-sample, high-dimensional data (e.g., gene expression):

1. **DFFS** (sample-wise selection): each training sample is removed and compared with every class of the remaining samples. A feature is kept for that sample if its nearest class (smallest normalized variance) is the sample's true class.
2. **SFFS** (frequency ranking): the top-L features of each sample (L = ceil(sqrt(m)/5), m = number of features) are counted, and features are ranked by how often they appear.
3. **CPR** (re-ranking): the SFFS candidates are re-sorted by the aggregated separability score Sep = (A·S·D)^(1/3), computed from the run lengths of the class labels after sorting the samples by the feature value.

## Experimental protocol (no information leakage)

- Each dataset is split **once** into a stratified train (90%) and test (10%) set (`random_state=123`; plain split if a class is too small to stratify).
- Every feature selection method uses the **training set only**.
- Features are applied to train and test, a `DecisionTreeClassifier(random_state=42)` is fitted on the training set, and the accuracy is measured on the **same test set** for all methods.
- Reported values: accuracy for the top-k features (k = 1, ..., 30), their mean, and the all-features baseline.
- The DFFS radar/bar plots (Top 1-5, precision/recall/F1) use leave-one-out on the training set. Because DFFS picks each sample's features with that sample's own label, these values describe separability on the training data, not generalization accuracy.
- The stability (bootstrap, Jaccard) and feature-correlation analyses use the full datasets because they characterize the selection procedure, not predictive performance.

## Datasets

Colon, CNS, GLI, SMK, Covid-19 (3 classes), Leukemia (3 classes), MLL (3 classes), SRBCT (4 classes). Datasets are **not included**; [add sources/links].
Expected format: `.csv` or `.xlsx`, **no header**, one sample per row, class label in the **last column**. Set `DATA_DIR` at the top of each script.

## Methods compared

CPR, SFFS, SFFS-CPR, MPR, CEI, EMC, MC and DMC (the last two are binary only), Fisher score, Mutual Information, Pearson correlation, Chi-square, ReliefF, mRMR, and tree-based importances (Decision Tree, Random Forest, XGBoost).

## Requirements

Python 3.9, numpy, pandas, scikit-learn, scipy, matplotlib, openpyxl (for `.xlsx`), joblib, scikit-rebate (ReliefF), xgboost.

```bash
pip install numpy pandas scikit-learn scipy matplotlib openpyxl joblib scikit-rebate xgboost
```

## Usage

```bash
# 1. Edit DATA_DIR in the script, then run it
python mean_accuracy_filters.py        # MPR, CEI, EMC, MC, DMC, CPR, SFFS-CPR (mean accuracy over top 1-30)
python sffs_vs_sffs_cpr.py             # SFFS vs SFFS-CPR
python baselines_top30.py              # SFFS-CPR vs Fisher, MI, Corr, Chi2, ReliefF
python dffs_loo_top1_5.py              # DFFS Top 1-5 vs all features (radar plot numbers)
python dffs_metrics.py                 # accuracy, precision, recall, F1: DFFS vs all features
python plot_curves.py                  # accuracy curves from the saved CSV
```

Results (CSV tables) are written to the `*_results/` folders created in the current working directory.

## Reproducibility

Fixed seeds: data split `random_state=123`, decision tree `random_state=42`. With few samples, one test sample changes the accuracy by several percentage points, so use `SEEDS = list(range(10))` in the scripts for a mean ± std over splits.

## Citation

```bibtex
@article{...,
  title   = {...},
  author  = {...},
  journal = {...},
  year    = {...}
}
```

## License

[e.g., MIT]

## Contact

[Name, affiliation, email]


















# SFFS-CPR

SFFS-CPR is a hybrid method that integrates two filters: Static Feature-wise Feature Selection (SFFS), which generalizes the Dynamic Feature-wise Feature Selection (DFFS), and frequency-based Constructive Pattern Recognition (CPR).

--> The file **significant_features.pkl** contains the set of features dynamically selected by the Dynamic Feature-wise Feature Selection (DFFS) method for each dataset. To save you time, we have prepared this file in advance. Therefore, you can skip lines 112–206. It is in the "DFFS selected features folder". Simply update the file path in line 214 to match your local system, and load the corresponding significant_features.pkl file using lines 219–220. However, if you wish to apply DFFS to your own dataset, you must first load your dataset and then execute lines 219–220. The datasets used in the paper have already been preloaded in advance, spanning lines 18–105.

--> SFFS utilizes the significant_features—a set of features selected by DFFS—and generalizes them by selecting a subset that is applicable across the entire dataset, as implemented in lines 226–253.

--> CPR begins at line 262, where the response variable is reordered based on the sorted values of each feature, as implemented in lines 262–273. CPR performs feature sorting in lines 280–355.

**Supplementary codes and file:**

--> Lines 486–850 compute the accuracy of the top 50 features selected by existing filters, including Fisher Score, Mutual Information, ReliefF, and others.

--> Lines 854–867 compute the prediction accuracy using all available features, evaluated with the Leave-One-Out strategy.

--> The correlation analysis presented in the paper is implemented in lines 2318–2383.

--> Stability analysis.py is the corresponding file for calculation the stability of the feature selection methods.

**Plots and visualizations of the SFFS-CPR:** 

    - The accuracy of the top 50 features is computed in lines 364–412 and plotted in lines  871-1139, corresponding to Figure 4 in the paper.
    
    - The accuracy, precision, recall, and F1 score of DFFS are evaluated through per-sample feature selection, as implemented in lines 425–458, and plotted in lines 2080-2310 corresponding to Figure 6 in the paper.
    
    - The accuracy of DFFS using the top 5 features is computed in lines 464–482, and plotted in lines 1144 1517, corresponding to Figure 5 in the paper.

    - The box plot analysis is plotted in lines 1525-1622, corresponding to Figure 7 in the paper.

    - Lines 1672–1992 generate plots corresponding to Figure 2 and Figure 3 in the paper.
    

    
