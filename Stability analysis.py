import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import tree
import pickle
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")




# Load the dataset  COLON
df = pd.read_excel('D:\\DataS Projects\\DataS\\dataset2\\Colon.xlsx', header=None)
le = LabelEncoder()
y_encoded = le.fit_transform(df.iloc[:, -1])  # returns numpy array of ints

# Assign as pandas Series with integer dtype
y = pd.Series(y_encoded, dtype=int)
# Separate features and labels
X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
##############################   
###################### CNS; 
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\CNS.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
###################GLI##########  
df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\GLI.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
#######################SMK########################

df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\SMK.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]

################# Covid-3c
df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\Covid.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
df.iloc[:, -1] = df.iloc[:, -1].astype(int)   # force dtype change# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.astype(int)
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
####################Leukemia-3c
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\Leukemia_3c.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
#######################MLL-3c
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\MLL.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]
######################SRBCT-4c
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\SRBCT.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
# X_subsets1 = {label: X[y == label].reset_index(drop=True) for label in labels}
# Initialize hc1 and hc2: each feature m maps to a list of values
num_features = X.shape[1]

def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(selected_feature_sets):
    """
    Compute stability across multiple feature subsets using the Jaccard index.
    Works whether input is list-of-lists or list-of-sets.
    """
    from itertools import combinations
    scores = []
    for A, B in combinations(selected_feature_sets, 2):
        A, B = set(A), set(B)  # 🔹 ensure sets
        scores.append(jaccard_index(A, B))
    return np.mean(scores) if scores else 0.0




from collections import defaultdict

def run_lengths_for_label(y, label):
    """Return list of run lengths for `label` in sequence y (y can be list or 1D array)."""
    lengths = []
    current = 0
    for v in y:
        if v == label:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths

def sep_b_cube_root(y, label):
    """
    Compute Sep_b = (A * S * D)^(1/3) with:
      A = (k_max - k_b) / (k_max - 1),  k_max = count_b
      S = 1 / (1 + std(run_lengths)/mean(run_lengths))
      D = max(run_lengths) / count_b
    Special-case: if label absent -> 0, if single run -> 1
    """
    # ensure list
    y_list = list(y)
    run_lengths = run_lengths_for_label(y_list, label)
    if len(run_lengths) == 0:
        return 0.0
    k_b = len(run_lengths)
    count_b = sum(run_lengths)
    if k_b == 1 or count_b <= 1:
        return 1.0

    # A term
    k_max = count_b
    if k_max <= 1:
        A = 1.0
    else:
        A = (k_max - k_b) / (k_max - 1)

    # S term (smooth CV)
    mean_len = np.mean(run_lengths)
    std_len = np.std(run_lengths, ddof=0)
    cv = std_len / mean_len if mean_len > 0 else 0.0
    S = 1.0 / (1.0 + cv)

    # D term (dominance)
    D = max(run_lengths) / count_b

    # geometric cube-root
    prod = A * S * D
    # numerical safety: clamp to [0,1]
    if prod <= 0.0:
        return 0.0
    sep = prod ** (1.0 / 3.0)
    return float(sep)


# Top=int(0.5*X.shape[1])
limit=np.ceil(((X.shape[1])**0.5)/5)
Top= int(limit * X.shape[0])
top_features = []
n_bootstraps = 10   # number of bootstrap resamples

for i in range(n_bootstraps):
    # Bootstrap resample (sample with replacement)
    X_resampled, y_resampled = resample(X, y, replace=True, random_state=42+i)
    X_reduced = X_resampled.copy()

    sorted_X = pd.DataFrame()
    reordered_y = {}

    for col in X_reduced.columns:
        sorted_idx = X_reduced[col].sort_values().index
        sorted_X[col] = X_reduced[col].sort_values().values
        reordered_y[col] = y_resampled[sorted_idx].values

    all_labels = sorted({lbl for arr in reordered_y.values() for lbl in np.unique(arr)})

    # Compute per-feature per-label Sep using cube-root Sep
    results = {}
    for feat_key, y_arr in reordered_y.items():
        per_label = {}
        for lbl in all_labels:
            per_label[lbl] = sep_b_cube_root(y_arr, lbl)
        per_label["aggregated"] = np.mean(list(per_label.values()))
        results[feat_key] = per_label

    dff = pd.DataFrame.from_dict(results, orient="index")

    # Aggregated sorting
    aggregated_features = dff["aggregated"].sort_values(ascending=False).index.tolist()

    # Take top-3 features
    top_k_features = aggregated_features[:Top]

    # 🔹 Convert to set so Jaccard can be computed
    top_features.append(set(top_k_features))

# Compute stability with Jaccard index
stability_score = stability_jaccard(top_features)

print("Selected features per bootstrap run:", top_features)
print("Jaccard stability of CPR (bootstrapping):", round(stability_score,2))







########################################################################

import numpy as np
import pandas as pd
from sklearn.utils import resample
from itertools import combinations

##############################
# 🔹 Fisher score
##############################
def fisher_score(X, y):
    scores = []
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)

    for col in X.columns:
        numerator = 0
        denominator = 0
        for c in classes:
            x_c = X[y == c][col]
            mean_c = x_c.mean()
            var_c = x_c.var()
            n_c = len(x_c)
            numerator += n_c * (mean_c - overall_mean[col]) ** 2
            denominator += n_c * var_c
        score = numerator / (denominator + 1e-6)
        scores.append(score)

    return np.array(scores)

##############################
# 🔹 Jaccard stability utils
##############################
def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(selected_feature_sets):
    """Compute stability across multiple feature subsets using Jaccard index."""
    scores = []
    for A, B in combinations(selected_feature_sets, 2):
        A, B = set(A), set(B)
        scores.append(jaccard_index(A, B))
    return np.mean(scores) if scores else 0.0

##############################
# 🔹 Bootstrap Fisher stability
##############################
def fisher_feature_selection(X, y, top_ratio=0.5, n_bootstraps=10):
    """
    Apply Fisher score with bootstrapping and return selected top features per run.
    """
    top_features = []
    # Top = int(top_ratio * X.shape[1])
    Top= int(limit * X.shape[0])
    for i in range(n_bootstraps):
        # Bootstrap resample
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42+i)

        # Compute Fisher scores
        scores = fisher_score(X_resampled, y_resampled)
        Fisher = np.argsort(scores)[-Top:][::-1]
        ranked_features = X_resampled.columns[Fisher]

        # Store top features as a set
        top_features.append(set(ranked_features))

    return top_features


##############################
# 🔹 Run Fisher stability
##############################
limit=np.ceil(((X.shape[1])**0.5)/5)
top_features_fisher = fisher_feature_selection(X, y, top_ratio=limit, n_bootstraps=10)
stability_score_fisher = stability_jaccard(top_features_fisher)

print("Selected features per bootstrap run:", top_features_fisher)
print("Jaccard stability of Fisher (bootstrapping):", round(stability_score_fisher,2))





############ Mutual Info

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations

##############################
# 🔹 Mutual Information scores
##############################
def mutual_information_scores(X, y):
    """
    Calculate Mutual Information scores for each feature in X with respect to target y.
    """
    scores = mutual_info_classif(X, y, random_state=42)
    return scores

##############################
# 🔹 Jaccard stability utils
##############################
def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(selected_feature_sets):
    """Compute stability across multiple feature subsets using Jaccard index."""
    scores = []
    for A, B in combinations(selected_feature_sets, 2):
        A, B = set(A), set(B)
        scores.append(jaccard_index(A, B))
    return np.mean(scores) if scores else 0.0

##############################
# 🔹 Bootstrap MI stability
##############################
def mi_feature_selection(X, y, top_ratio=0.5, n_bootstraps=10):
    """
    Apply Mutual Information with bootstrapping and return selected top features per run.
    """
    top_features = []
    # Top = int(top_ratio * X.shape[1])
    Top= int(limit * X.shape[0])


    for i in range(n_bootstraps):
        # Bootstrap resample
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42+i)

        # Compute MI scores
        scores = mutual_information_scores(X_resampled, y_resampled)

        # Rank features
        MI_indices = np.argsort(scores)[-Top:][::-1]
        ranked_features = X_resampled.columns[MI_indices]

        # Store top features as a set
        top_features.append(set(ranked_features))

    return top_features


##############################
# 🔹 Run MI stability
##############################



limit=np.ceil(((X.shape[1])**0.5)/5)
top_features_mi = mi_feature_selection(X, y, top_ratio=limit, n_bootstraps=10)
stability_score_mi = stability_jaccard(top_features_mi)

print("Selected features per bootstrap run:", top_features_mi)
print("Jaccard stability of Mutual Information (bootstrapping):", round(stability_score_mi,2))












import numpy as np
import pandas as pd
from sklearn.utils import resample
from itertools import combinations

##############################
# 🔹 Correlation coefficient scores
##############################
def correlation_scores(X, y):
    """
    Calculate absolute correlation coefficient between each feature and target y.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: Absolute correlation coefficients for each feature.
    """
    scores = []
    y_series = pd.Series(y)

    for col in X.columns:
        corr = np.corrcoef(X[col], y_series)[0, 1]
        scores.append(abs(corr))   # take absolute correlation
    return np.array(scores)

##############################
# 🔹 Jaccard stability utils
##############################
def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(selected_feature_sets):
    """Compute stability across multiple feature subsets using Jaccard index."""
    scores = []
    for A, B in combinations(selected_feature_sets, 2):
        A, B = set(A), set(B)
        scores.append(jaccard_index(A, B))
    return np.mean(scores) if scores else 0.0

##############################
# 🔹 Bootstrap Correlation stability
##############################
def corr_feature_selection(X, y, top_ratio=0.5, n_bootstraps=10):
    """
    Apply correlation coefficient with bootstrapping and return selected top features per run.
    """
    top_features = []
    Top= int(limit * X.shape[0])

    # Top = int(top_ratio * X.shape[1])

    for i in range(n_bootstraps):
        # Bootstrap resample
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42+i)

        # Compute correlation scores
        scores = correlation_scores(X_resampled, y_resampled)

        # Rank features
        Corr_indices = np.argsort(scores)[-Top:][::-1]
        ranked_features = X_resampled.columns[Corr_indices]

        # Store top features as a set
        top_features.append(set(ranked_features))

    return top_features


##############################
# 🔹 Run Correlation stability
##############################
limit=np.ceil(((X.shape[1])**0.5)/5)
top_features_corr = corr_feature_selection(X, y, top_ratio=limit, n_bootstraps=10)
stability_score_corr = stability_jaccard(top_features_corr)

print("Selected features per bootstrap run:", top_features_corr)
print("Jaccard stability of Correlation Coefficient (bootstrapping):", round(stability_score_corr,2))






import math
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.feature_selection import chi2
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations

##############################
# 🔹 Chi² scores with discretization
##############################
def chi2_scores(X, y, n_bins=10):
    """
    Calculate chi² scores for each feature in X with respect to target y.
    Continuous features are discretized into bins first.

    Args:
        X (pd.DataFrame): Feature matrix (numeric features allowed).
        y (pd.Series or np.array): Target vector (categorical labels).
        n_bins (int): Number of bins for discretization.

    Returns:
        np.array: Chi² scores for each feature.
    """
    # Discretize continuous features
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X_binned = discretizer.fit_transform(X)

    # Compute chi²
    scores, _ = chi2(X_binned, y)
    return scores

##############################
# 🔹 Jaccard stability utils
##############################
def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(selected_feature_sets):
    """Compute stability across multiple feature subsets using Jaccard index."""
    scores = []
    for A, B in combinations(selected_feature_sets, 2):
        A, B = set(A), set(B)
        scores.append(jaccard_index(A, B))
    return np.mean(scores) if scores else 0.0

##############################
# 🔹 Bootstrap Chi² stability
##############################
def chi2_feature_selection(X, y, top_ratio=0.5, n_bootstraps=10, n_bins=10):
    """
    Apply chi² filter with bootstrapping and return selected top features per run.
    """
    top_features = []
    Top= int(limit * X.shape[0])

    # Top = int(top_ratio * X.shape[1])

    for i in range(n_bootstraps):
        # Bootstrap resample
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42+i)

        # Compute chi² scores (with discretization)
        scores = chi2_scores(X_resampled, y_resampled, n_bins=n_bins)

        # Rank features
        Chi2_indices = np.argsort(scores)[-Top:][::-1]
        ranked_features = X_resampled.columns[Chi2_indices]

        # Store top features as a set
        top_features.append(set(ranked_features))

    return top_features


##############################
# 🔹 Run Chi² stability
##############################
limit=np.ceil(((X.shape[1])**0.5)/5)
top_features_chi2 = chi2_feature_selection(X, y, top_ratio=limit, n_bootstraps=10, n_bins=10)
stability_score_chi2 = stability_jaccard(top_features_chi2)

print("Selected features per bootstrap run:", top_features_chi2)
print("Jaccard stability of Chi² (bootstrapping):", round(stability_score_chi2,2))








import numpy as np
import pandas as pd
from sklearn.utils import resample
from itertools import combinations
from skrebate import ReliefF

##############################
# 🔹 ReliefF scores
##############################
def relieff_scores(X, y, n_neighbors=10):
    """
    Calculate ReliefF scores for each feature in X with respect to target y.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target vector.
        n_neighbors (int): Number of neighbors used in ReliefF.

    Returns:
        np.array: ReliefF feature scores.
    """
    fs = ReliefF(n_neighbors=n_neighbors, n_features_to_select=X.shape[1])
    fs.fit(X.values, np.array(y))
    return fs.feature_importances_

##############################
# 🔹 Jaccard stability utils
##############################
def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(selected_feature_sets):
    """Compute stability across multiple feature subsets using Jaccard index."""
    scores = []
    for A, B in combinations(selected_feature_sets, 2):
        A, B = set(A), set(B)
        scores.append(jaccard_index(A, B))
    return np.mean(scores) if scores else 0.0

##############################
# 🔹 Bootstrap ReliefF stability
##############################
def relieff_feature_selection(X, y, top_ratio=0.5, n_bootstraps=10, n_neighbors=10):
    """
    Apply ReliefF filter with bootstrapping and return selected top features per run.
    """
    top_features = []
    Top= int(limit * X.shape[0])
    # Top = int(top_ratio * X.shape[1])

    for i in range(n_bootstraps):
        # Bootstrap resample
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42+i)

        # Compute ReliefF scores
        scores = relieff_scores(X_resampled, y_resampled, n_neighbors=n_neighbors)

        # Rank features
        Relief_indices = np.argsort(scores)[-Top:][::-1]
        ranked_features = X_resampled.columns[Relief_indices]

        # Store top features as a set
        top_features.append(set(ranked_features))

    return top_features


##############################
# 🔹 Run ReliefF stability
##############################
limit=np.ceil(((X.shape[1])**0.5)/5)
top_features_relief = relieff_feature_selection(X, y, top_ratio=limit, n_bootstraps=10, n_neighbors=10)
stability_score_relief = stability_jaccard(top_features_relief)

print("Selected features per bootstrap run:", top_features_relief)
print("Jaccard stability of ReliefF (bootstrapping):", round(stability_score_relief,2))



 X=X.astype(np.float64) 












import numpy as np
import pandas as pd
from sklearn.utils import resample
from collections import Counter
from itertools import combinations

############################
# Jaccard index and stability
############################
def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(feature_sets):
    """Average pairwise Jaccard similarity across feature sets."""
    scores = []
    for A, B in combinations(feature_sets, 2):
        scores.append(jaccard_index(set(A), set(B)))
    return np.mean(scores) if scores else 0.0


############################
# Wrapper: run your SFFS once on (X, y)
############################
def run_SFFS_feature_ranking(X, y, labels):
    num_features = X.shape[1]
    significant_features = []

    for i in range(len(X)):
        row_i = X.iloc[i, :]
        X_new = X.drop(index=X.index[i]).reset_index(drop=True)
        y_new = y.drop(index=y.index[i]).reset_index(drop=True)
        X_subsets = {label: X_new[y_new == label].reset_index(drop=True) for label in labels}

        valid_normalized_variances = []
        valid_columns = []
        Feature = []

        for m in range(num_features):
            field_value = row_i[m]
            normalized_variances = {}

            for label in labels:
                diff = X_subsets[label].iloc[:, m] - field_value
                a, b = diff.min(), diff.max()
                if b == 0 and a == 0:
                    normalized_diff = diff * 0
                elif b == 0:
                    normalized_diff = diff.apply(lambda x: x / abs(a) if x < 0 else 0)
                elif a == 0:
                    normalized_diff = diff.apply(lambda x: x / b if x >= 0 else 0)
                else:
                    normalized_diff = diff.apply(lambda x: x / b if x >= 0 else x / abs(a))

                normalized_variances[label] = ((normalized_diff - 0) ** 2).sum() / (len(normalized_diff) - 1)

            best_label_norm = min(normalized_variances, key=lambda k: abs(normalized_variances[k]))

            # FIXED line
            if best_label_norm == y.iloc[i]:
                valid_normalized_variances.append(normalized_variances)
                valid_columns.append(m)

        abs_differences = [
            np.mean([abs(d[c1] - d[c2]) for c1, c2 in combinations(d.keys(), 2)])
            for d in valid_normalized_variances
        ]

        sorted_diff_columns = np.argsort(abs_differences)[::-1]
        sorted_valid_columns = [valid_columns[s] for s in sorted_diff_columns]
        Feature.append(sorted_valid_columns)
        significant_features.append(Feature)

    limit = np.ceil(((X.shape[1]) ** 0.5) / 5)
    topL_features = []
    for row in significant_features:
        features = row[0]
        top_features = features[:int(limit)]
        topL_features.append(top_features)

    all_features = [f for row in topL_features for f in row]
    feature_counts = Counter(all_features)
    ranked_features = [f for f, _ in sorted(feature_counts.items(), key=lambda x: (-x[1], x[0]))]

    return ranked_features

############################
# Bootstrap stability analysis
############################
def bootstrap_stability_SFFS(X, y, labels, n_bootstraps=10, top_ratio=None):
    ranked_sets = []

    for i in range(n_bootstraps):
        X_res, y_res = resample(X, y, replace=True, random_state=100+i)
        ranked_features = run_SFFS_feature_ranking(X_res, y_res, labels)

        if top_ratio is not None:   # take only top ratio of ranked features
            Top = int(top_ratio * len(ranked_features))
            ranked_features = ranked_features[:Top]

        ranked_sets.append(ranked_features)

    stability_score = stability_jaccard(ranked_sets)
    return stability_score, ranked_sets


############################
# Example usage
############################
# Full ranked_features
stability_full, ranked_sets_full = bootstrap_stability_SFFS(X, y, labels, n_bootstraps=10, top_ratio=None)
print("SFFS Stability (full ranked features):", round(stability_full, 2))

# # Top 50% ranked_features
# stability_half, ranked_sets_half = bootstrap_stability_SFFS(X, y, labels, n_bootstraps=10, top_ratio=0.5)
# print("SFFS Stability (top 50% ranked features):", round(stability_half, 2))


ranked_sets_half = []
for rset in ranked_sets_full:
    Top = int(0.5 * len(rset))  # 50% of length
    ranked_sets_half.append(rset[:Top])

stability_half = stability_jaccard(ranked_sets_half)

print("Stability (full ranked features):", round(stability_full,2))
print("Stability (top 50% of ranked features):", round(stability_half,2))





##################################
#################Ckeck non-repeated samples ratio################

import numpy as np
import pandas as pd
from sklearn.utils import resample
from itertools import combinations

##############################
# 🔹 Fisher score
##############################
def fisher_score(X, y):
    scores = []
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)

    for col in X.columns:
        numerator = 0
        denominator = 0
        for c in classes:
            x_c = X[y == c][col]
            mean_c = x_c.mean()
            var_c = x_c.var()
            n_c = len(x_c)
            numerator += n_c * (mean_c - overall_mean[col]) ** 2
            denominator += n_c * var_c
        score = numerator / (denominator + 1e-6)
        scores.append(score)

    return np.array(scores)

##############################
# 🔹 Jaccard stability utils
##############################
def jaccard_index(A, B):
    """Compute Jaccard index between two sets A and B."""
    return len(A & B) / len(A | B) if len(A | B) > 0 else 0.0

def stability_jaccard(selected_feature_sets):
    """Compute stability across multiple feature subsets using Jaccard index."""
    scores = []
    for A, B in combinations(selected_feature_sets, 2):
        A, B = set(A), set(B)
        scores.append(jaccard_index(A, B))
    return np.mean(scores) if scores else 0.0

##############################
# 🔹 Bootstrap Fisher stability
##############################
def fisher_feature_selection(X, y, top_ratio=0.5, n_bootstraps=10):
    """
    Apply Fisher score with bootstrapping and return selected top features per run.
    Also computes bootstrap stats: non-repeated ratio, rows, cols.
    """
    top_features = []
    stats = []

    Top = int(limit * X.shape[0])
    for i in range(n_bootstraps):
        # Bootstrap resample
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42+i)

        # --- Stats for this bootstrap ---
        n_rows = X_resampled.shape[0]
        n_cols = X_resampled.shape[1]
        unique_samples = len(np.unique(X_resampled.index))
        non_repeated_ratio = unique_samples / n_rows

        stats.append({
            "bootstrap": i+1,
            "rows": n_rows,
            "cols": n_cols,
            "non_repeated_ratio": round(non_repeated_ratio, 3)
        })

        # Compute Fisher scores
        scores = fisher_score(X_resampled, y_resampled)
        Fisher = np.argsort(scores)[-Top:][::-1]
        ranked_features = X_resampled.columns[Fisher]

        # Store top features as a set
        top_features.append(set(ranked_features))

    return top_features, pd.DataFrame(stats)

##############################
# 🔹 Run Fisher stability
##############################
limit = np.ceil(((X.shape[1])**0.5) / 5)
top_features_fisher, bootstrap_stats = fisher_feature_selection(X, y, top_ratio=limit, n_bootstraps=10)
stability_score_fisher = stability_jaccard(top_features_fisher)

print("Selected features per bootstrap run:", top_features_fisher)
print("Jaccard stability of Fisher (bootstrapping):", round(stability_score_fisher, 2))

# Show bootstrap stats
print("\nBootstrap stats:")
print(bootstrap_stats)
