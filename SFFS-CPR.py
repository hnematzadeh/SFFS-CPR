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
# Create class-specific subsets
labels = np.unique(y)
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
num_features = X.shape[1]
#############################   
########################## GLI
df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\GLI.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
num_features = X.shape[1]

###############SMK
df = pd.read_csv ('D:\DataS Projects\DataS\dataset2\SMK.csv', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
num_features = X.shape[1]
###################################################
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
num_features = X.shape[1]

#######################SRBCT-4c
df = pd.read_excel ('D:\DataS Projects\DataS\dataset2\SRBCT.xlsx', header=None)
le = LabelEncoder()
df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])
# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# Create class-specific subsets
labels = np.unique(y)
num_features = X.shape[1]

  
###########################################
###########################################
################DFFS#######################
###########################################
start_time = time.time()
subset_n = np.empty((len(X), num_features))
subset_n1 = np.empty((len(X), num_features))

predicted_n = np.empty(len(X))
predicted_n1 = np.empty(len(X))

significant_features = []
significant_variances = {}  # maps feature index to its variance dict
for i in range(len(X)):
    arr_n =np.empty( num_features,  dtype=int)
    
    best_n = np.empty( num_features)

    row_i = X.iloc[i, :]  #Remove one sample
    X_new = X.drop(index=X.index[i]).reset_index(drop=True)  #Work with the rest of the dataset (X_new, y_new)
    y_new = y.drop(index=y.index[i]).reset_index(drop=True)
    X_subsets = {label: X_new[y_new == label].reset_index(drop=True) for label in labels} #Organize it per class (X_subsets)

    all_normalized_variances=[]
    valid_normalized_variances=[]
    valid_columns=[]
    Feature=[]
    for m in range(num_features):
        field_value = row_i[m]

        variances = {}
        normalized_variances={}
        for label in labels:
                diff = X_subsets[label].iloc[:, m] - field_value
                a = diff.min()
                b = diff.max()
                if b == 0 and a == 0:
                     normalized_diff = diff * 0  # all values are the same
                elif b == 0:
                     normalized_diff = diff.apply(lambda x: x / abs(a) if x < 0 else 0)
                elif a == 0:
                     normalized_diff = diff.apply(lambda x: x / b if x >= 0 else 0)
                else:
                     normalized_diff = diff.apply(lambda x: x / b if x >= 0 else x / abs(a))
                
                normalized_variances[label] = ((normalized_diff - 0) ** 2).sum() / (len(normalized_diff) - 1)  #sample variance
                
        all_normalized_variances.append(normalized_variances)
        
        
        best_label_norm=min(normalized_variances, key=lambda k: abs(normalized_variances[k])) # predicts the feature-wise class label
        ###################
        if best_label_norm == y[i]:  # if the feature-wise predicted class label equals to the actual class label, the corresponding column (feature) must be saved.
            valid_normalized_variances.append(normalized_variances)
            valid_columns.append(m)  # store the corresponding column
        #####################
              
        
        best_n[m] = normalized_variances[best_label_norm] # best_n records the corresponding normalized_variances of the best_label_norm
        arr_n[m] = best_label_norm # arr_n records the predicted feature-wise class label
    
         
        
    ##################
    # abs_differences = [abs(d[0] - d[1]) for d in valid_normalized_variances]
    abs_differences = [
    np.mean([abs(d[c1] - d[c2]) for c1, c2 in combinations(d.keys(), 2)])
    for d in valid_normalized_variances
    ]

    # Step 2: Returns the indices that would sort abs_differences in ascending order (smallest → largest).
    sorted_diff_columns = np.argsort(abs_differences)[::-1]

    # Step 3: Get corresponding column indices 
    sorted_valid_columns = [valid_columns[s] for s in sorted_diff_columns]
    
    Feature.append(sorted_valid_columns)
    significant_features.append(Feature)
    #################
    # if i==196:
    #   print("196 valid columns", valid_columns)
    #   print(all_normalized_variances)
    
    predicted_n[i] = np.bincount(arr_n.astype(int)).argmax() # feature-wise predicted label for sample i
    subset_n[i,:] = np.argsort(best_n)   
accuracy = np.mean(y == predicted_n)
Acc=round(accuracy, 2) # Accuracy of EFC with full features
end_time = time.time()
execution_time = round(end_time - start_time)
print(f"Execution time: {execution_time:.6f} seconds")    


# Assuming significant_features is a list of lists (e.g., list of feature index arrays)
lengths = [len(np.ravel(row)) for row in significant_features]  # flatten each row just in case
average_length = sum(lengths) / len(lengths)

print("Average number of features per sample:", round(average_length, 2))
print("MIN number of features per sample:", min(lengths))
print("MAX number of features per sample:", max(lengths))

########################################################


############## LOAD and SAVE significant features calculated by DFFS
#######################################################################

file_path = r'D:\Papers\Explainable Feature_based Classification\Results\Covid\significant_features.pkl'
### For Saving
with open(file_path, 'wb') as f:
    pickle.dump(significant_features, f)   
### For Loading    
with open(file_path, 'rb') as f:
    significant_features = pickle.load(f)

###########################################################
############  SFFS ########################################
############  TOP limit Features in each row  #############

limit=np.ceil(((X.shape[1])**0.5)/5)
topL_features = []

for row in significant_features:
    features = row[0]               # unwrap inner list
    top_features = features[:int(limit)]    # keep top 10 only
    topL_features.append(top_features)

print("First row original length:", len(significant_features[0][0]))
print("First row kept length:", len(topL_features[0]))


######
###### ranked_features is the frequency of features appeared in top10_features
######
from collections import Counter

# Flatten properly (each row is already a list of features)
all_features = [feature for row in topL_features for feature in row]

# Count occurrences
feature_counts = Counter(all_features)

# Sort by frequency (descending), then by feature ID if tie
ranked_features = [f for f, _ in sorted(feature_counts.items(), key=lambda x: (-x[1], x[0]))]

print("Feature counts:", feature_counts)
print("Ranked features:", ranked_features)



## for CPR
# X_reduced = X.copy()

##################### CPR ##########################################
####################################################################
X_reduced = X[ranked_features].copy()
## Sort each feature and reorder y
sorted_X = pd.DataFrame()
reordered_y = {}

for col in X_reduced.columns:
    sorted_idx = X_reduced[col].sort_values().index
    sorted_X[col] = X_reduced[col].sort_values().values
    reordered_y[col] = y[sorted_idx].values

print("Sorted features:\n", sorted_X)
print("Reordered response variables:\n", reordered_y)



####### (A.S.D)**1/3


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


all_labels = sorted({lbl for arr in reordered_y.values() for lbl in np.unique(arr)})

# Compute per-feature per-label Sep using cube-root Sep
results = {}
for feat_key, y_arr in reordered_y.items():
    per_label = {}
    for lbl in all_labels:
        per_label[lbl] = sep_b_cube_root(y_arr, lbl)
    per_label["aggregated"] = np.mean(list(per_label.values()))
    results[feat_key] = per_label

dff = pd.DataFrame.from_dict(results, orient="index")  # features x labels (+ aggregated)

# Per-label sortings
sortings = {lbl: dff[lbl].sort_values(ascending=False).index.tolist() for lbl in all_labels}
sortings["aggregated"] = dff["aggregated"].sort_values(ascending=False).index.tolist()




########## Top 50 features of SFFS-CPR
########## Accuracy calculation of top 50 features using sortings["aggregated"]


Top = 50  # compute accuracies for top 1..50 features
accuracies = []

# Take aggregated features (third row of sortings dict we built earlier)
aggregated_features = sortings["aggregated"]

# Ensure aggregated_features is a flat list of column indices (or names)
flat_aggregated_features = []
for f in aggregated_features:
    if isinstance(f, (list, tuple)):
        flat_aggregated_features.extend(f)
    else:
        flat_aggregated_features.append(f)

flat_aggregated_features = flat_aggregated_features[:Top]  # take top 'Top' features
dectree = tree.DecisionTreeClassifier(random_state=42)

# Loop over top 1, 2, ..., Top features
for k in range(1, Top + 1):
    selected_features = flat_aggregated_features[:k]
    y_pred = []

    # Leave-one-out cross-validation
    for i in range(len(X)):
        if np.issubdtype(type(selected_features[0]), np.integer):
            X_new = X.iloc[:, selected_features]
        else:
            X_new = X[selected_features]

        X_test = X_new.iloc[i:i+1, :]
        X_train = X_new.drop(index=i)
        y_test = y.iloc[i]
        y_train = y.drop(index=i)

        dectree.fit(X_train, y_train)
        y_pred.append(dectree.predict(X_test)[0])

    accuracy = np.mean(y == y_pred)
    accuracies.append(accuracy)

# Round for display
accuracies_rounded = [round(acc, 2) for acc in accuracies]
print(f"Accuracies for top 1..{Top} aggregated features:", accuracies_rounded)
print("Mean accuracy:", np.mean(accuracies_rounded))
print("Top 10:", round(np.mean(accuracies_rounded[0:10]),2))
print("Top 20:", round(np.mean(accuracies_rounded[0:20]),2))
print("Top 30:", round(np.mean(accuracies_rounded[0:30]),2))
print("Top 40:", round(np.mean(accuracies_rounded[0:40]),2))
print("Top 50:", round(np.mean(accuracies_rounded[0:50]),2))
#####################################################
#####################################################
#####################################################






### With all  features selected by DFFS

# new=subset_n[:, :1]
new = significant_features
y = y.astype(int)
y_pred = []
dectree = tree.DecisionTreeClassifier(random_state=42)
for i in range(len(X)):

    feature_indices = np.array(new[i])[0]  # indices for sample i
    feature_indices=np.array(new[i]).flatten()
    feature_indices=np.array(feature_indices).flatten()
    X_new =  X.iloc[:,feature_indices]  
    X_test=X_new.iloc[i:i+1, :]
    X_train = X_new.drop(index=i)                   
    y_test = y.iloc[i]  
    y_train = y.drop(index=i)     
    dectree.fit(X_train, y_train)
    y_pred.append(dectree.predict(X_test)[0])
accuracy = np.mean(y == y_pred)
Acc=round(accuracy, 2)    

# Convert y to list or numpy array to align with y_pred
y_true = y.tolist()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Precision, Recall, F1-Score 
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", Acc)
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1 Score:", round(f1, 2))



#### With Top featuresby DFFS

Top=5
new = significant_features
y = y.astype(int)
y_pred = []
dectree = tree.DecisionTreeClassifier(random_state=42)
for i in range(len(X)):

       new_n =  new[i][0][:Top]
       feature_indices=np.array(new_n).flatten()
       X_new =  X.iloc[:,feature_indices]  # select only top 5 features for sample i
       X_test=X_new.iloc[i:i+1, :]
       X_train = X_new.drop(index=i)                   # All rows except i
       y_test = y.iloc[i]  
       y_train = y.drop(index=i)     
       dectree.fit(X_train, y_train)
       y_pred.append(dectree.predict(X_test)[0])
accuracy = np.mean(y == y_pred)
Acc=round(accuracy, 2)    
print(Acc)  


##########
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
        # To avoid divide-by-zero
        score = numerator / (denominator + 1e-6)
        scores.append(score)
    
    return np.array(scores)

# Calculate scores and get top 50 features
scores = fisher_score(X, y)
Fisher = np.argsort(scores)[-50:][::-1]
ranked_features = X.columns[Fisher]

print("Top 50 feature indices:", Fisher)
print("Top 50 feature names:", ranked_features.tolist())

##### MUTUAL INFORMATION##############
from sklearn.feature_selection import mutual_info_classif

def mutual_information_scores(X, y):
    """
    Calculate Mutual Information scores for each feature in X with respect to target y.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of Mutual Information scores.
    """
    # mutual_info_classif computes the score for each feature
    scores = mutual_info_classif(X, y, random_state=42)
    return scores

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate Mutual Information scores for each feature
scores = mutual_information_scores(X, y)

# Get the indices of the top 50 features based on their scores
# np.argsort returns the indices that would sort an array
MI_indices = np.argsort(scores)[-50:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[MI_indices]

print("Top 50 feature indices:", MI_indices)
print("Top 50 feature names:", ranked_features.tolist())
########################
########################
def decision_tree_feature_importance(X, y):
    """
    Calculate feature importances using a Decision Tree Classifier.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of feature importance scores.
    """
    # Initialize Decision Tree with fixed random_state for reproducibility
    dectree = tree.DecisionTreeClassifier(random_state=42)
    
    # Fit the tree to the data
    dectree.fit(X, y)
    
    # Get feature importances
    return dectree.feature_importances_

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate Decision Tree feature importances
scores = decision_tree_feature_importance(X, y)

# Get the indices of the top 50 features based on their importance scores
DT_indices = np.argsort(scores)[-50:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[DT_indices]

print("Top 50 feature indices:", DT_indices)
print("Top 50 feature names:", ranked_features.tolist())
#######################################################
#######################################################
from sklearn.ensemble import RandomForestClassifier
def random_forest_feature_importance(X, y):
    """
    Calculate feature importances using a Random Forest Classifier.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of feature importance scores.
    """
    # Initialize Random Forest with a fixed random_state for reproducibility
    rf_model = RandomForestClassifier(random_state=42)
    
    # Fit the model to the data
    rf_model.fit(X, y)
    
    # Get feature importances
    return rf_model.feature_importances_

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate Random Forest feature importances
scores = random_forest_feature_importance(X, y)

# Get the indices of the top 50 features based on their importance scores
RF_indices = np.argsort(scores)[-50:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[RF_indices]

print("Top 50 feature indices:", RF_indices)
print("Top 50 feature names:", ranked_features.tolist())

#######################################################
#######################################################

import xgboost as xgb

def xgboost_feature_importance(X, y):
    """
    Calculate feature importances using an XGBoost Classifier.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of feature importance scores.
    """
    # Initialize XGBoost Classifier with a fixed random_state for reproducibility
    # eval_metric='logloss' is added to avoid a common warning in XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Fit the model to the data
    xgb_model.fit(X, y)
    
    # Get feature importances
    return xgb_model.feature_importances_

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate XGBoost feature importances
scores = xgboost_feature_importance(X, y)

# Get the indices of the top 50 features based on their importance scores
XGB_indices = np.argsort(scores)[-50:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[XGB_indices]

print("Top 50 feature indices:", XGB_indices)
print("Top 50 feature names:", ranked_features.tolist())

########################################################
########################################################

def correlation_feature_importance(X, y):
    """
    Calculate feature importances using absolute Pearson correlation with the target.
    
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series or np.array): The target vector.
        
    Returns:
        np.array: An array of feature importance scores (absolute correlations).
    """
    # Ensure y is a Pandas Series for compatibility
    y = pd.Series(y)
    
    # Compute correlation of each feature with the target
    correlations = X.apply(lambda col: col.corr(y))
    
    # Take absolute value so both positive and negative correlations are treated equally
    importance_scores = correlations.abs().values
    
    return importance_scores

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate correlation-based feature importances
scores = correlation_feature_importance(X, y)

# Get the indices of the top 50 features based on correlation strength
corr_indices = np.argsort(scores)[-50:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[corr_indices]

print("Top 50 feature indices:", corr_indices)
print("Top 50 feature names:", ranked_features.tolist())

#######################################################
#######################################################


from sklearn.feature_selection import chi2
from sklearn.preprocessing import KBinsDiscretizer

def chi2_feature_importance(X, y, n_bins=10, strategy="quantile"):
    """
    Calculate feature importances using Chi-Square test with discretization.
    
    Args:
        X (pd.DataFrame): The feature matrix (numeric features).
        y (pd.Series or np.array): The categorical target vector.
        n_bins (int): Number of bins to discretize numeric features.
        strategy (str): Binning strategy - "uniform", "quantile", or "kmeans".
        
    Returns:
        np.array: An array of feature importance scores (Chi2 statistics).
    """
    # Discretize numerical features into bins
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    X_binned = kb.fit_transform(X)
    
    # Perform chi-square test
    chi2_stats, p_values = chi2(X_binned, y)
    
    return chi2_stats

# Assuming X is a pandas DataFrame and y is a Series or array
# X, y = ...  # Your data should be loaded here

# Calculate Chi-Square feature importances
scores = chi2_feature_importance(X, y, n_bins=10, strategy="quantile")

# Get the indices of the top 50 features
chi2_indices = np.argsort(scores)[-50:][::-1]

# Get the names of the top 50 features
ranked_features = X.columns[chi2_indices]

print("Top 50 feature indices:", chi2_indices)
print("Top 50 feature names:", ranked_features.tolist())

#######################################################
#######################################################
from skrebate import ReliefF


start_time = time.time()
def reliefF_scores(X, y):
    """
    Calculate ReliefF scores for feature selection (classification).
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target vector (categorical).
        n_neighbors (int): Number of neighbors to consider (default=10).
        
    Returns:
        np.array: ReliefF feature importance scores.
    """
    # Ensure features are float (ReliefF requirement)
    X_float = X.astype(float)

    relief = ReliefF(n_features_to_select=X.shape[1])
    relief.fit(X_float.values, y)
    return relief.feature_importances_

# Example usage
scores = reliefF_scores(X, y)
relief_indices = np.argsort(scores)[-50:][::-1]
ranked_features = X.columns[relief_indices]
end_time = time.time()
execution_time = round(end_time - start_time)
print("Top 50 ReliefF feature indices:", relief_indices)
print("Top 50 ReliefF feature names:", ranked_features.tolist())
print(f"Execution time: {execution_time:.6f} seconds") 



##########################  ANOVA F-test
from sklearn.feature_selection import f_classif
start_time = time.time()
def anova_f_scores(X, y):
    """
    Compute ANOVA F-test scores for each feature (continuous X, categorical y).
    """
    F, p_values = f_classif(X, y)
    return F, p_values

# Example usage
scores, p_values = anova_f_scores(X, y)

# Get top 50 features
anova_indices = np.argsort(scores)[-50:][::-1]
ranked_features = X.columns[anova_indices]
end_time = time.time()
execution_time = round(end_time - start_time)
print("Top 50 ANOVA F-test feature indices:", anova_indices)
print("Top 50 ANOVA F-test feature names:", ranked_features.tolist())
print(f"Execution time: {execution_time:.6f} seconds") 

#######################################################
#######################################################
#######################################################
Top = 50  # compute accuracies for top 1..50 features
accuracies = []

# Ensure sorted_features is flat list of column indices (or names)
flat_sorted_features = []
for f in ranked_features:
    if isinstance(f, list) or isinstance(f, tuple):
        flat_sorted_features.extend(f)
    else:
        flat_sorted_features.append(f)

flat_sorted_features = flat_sorted_features[:Top]  # take top 'Top' features
dectree = tree.DecisionTreeClassifier(random_state=42)

# Loop over top 1, 2, ..., Top features
for k in range(1, Top + 1):
    selected_features = flat_sorted_features[:k]
    y_pred = []

    # Leave-one-out cross-validation
    for i in range(len(X)):
        X_new = X.iloc[:, selected_features] if np.issubdtype(type(selected_features[0]), np.integer) else X[selected_features]
        X_test = X_new.iloc[i:i+1, :]
        X_train = X_new.drop(index=i)
        y_test = y.iloc[i]
        y_train = y.drop(index=i)

        dectree.fit(X_train, y_train)
        y_pred.append(dectree.predict(X_test)[0])

    accuracy = np.mean(y == y_pred)
    accuracies.append(accuracy)

# Round for display
accuracies_rounded = [round(acc, 2) for acc in accuracies]
print(f"Accuracies for top 1..{Top} features:", accuracies_rounded)
print(np.mean(accuracies_rounded))
print("Top 10:", round(np.mean(accuracies_rounded[0:10]),2))
print("Top 20:", round(np.mean(accuracies_rounded[0:20]),2))
print("Top 30:", round(np.mean(accuracies_rounded[0:30]),2))
print("Top 40:", round(np.mean(accuracies_rounded[0:40]),2))
print("Top 50:", round(np.mean(accuracies_rounded[0:50]),2))


#### Accuracy with all available features using leave one out on original datasets
y_pred=[]
dectree = tree.DecisionTreeClassifier(random_state=42)
for i in range(len(X)):
    X_train = X.drop(index=i)
    y_train = y.drop(index=i)
    X_test = X.iloc[i:i+1, :]
    y_test = y.iloc[i]

    dectree.fit(X_train, y_train)
    y_pred.append(dectree.predict(X_test)[0])

# Compute accuracy
loo_accuracy = np.mean(y == y_pred)
print("Leave-one-out accuracy on the original dataset:", round(loo_accuracy, 2))



#################### Plots##############################
import matplotlib.pyplot as plt
accuracies = [0.76, 0.73, 0.77, 0.76, 0.77, 0.74, 0.74, 0.74, 0.74, 0.79,
              0.74, 0.79, 0.81, 0.76, 0.84, 0.89, 0.89, 0.87, 0.84, 0.85,
              0.89, 0.89, 0.85, 0.87, 0.90, 0.85, 0.85, 0.85, 0.82, 0.87,
              0.87, 0.82, 0.84, 0.84, 0.76, 0.84, 0.81, 0.84, 0.84, 0.85,
              0.84, 0.82, 0.85, 0.85, 0.81, 0.81, 0.81, 0.82, 0.81, 0.82]

# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.84

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)  # all x-axis values with larger size
plt.yticks(fontsize=26)
# Labels and title
plt.xlabel("Number of Features",fontsize=36)
plt.ylabel("Accuracy",fontsize=36)
plt.title("Colon",fontsize=36)
plt.legend(fontsize=22)
plt.ylim(0.65,0.9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


###########  CNS #######################
# Accuracies from the prompt
accuracies = [
    0.75, 0.67, 0.68, 0.75, 0.67, 0.78, 0.7, 0.8, 0.8, 0.8, 0.82, 0.83, 0.78, 0.8, 0.85,
    0.77, 0.77, 0.75, 0.83, 0.77, 0.77, 0.73, 0.75, 0.77, 0.77, 0.68, 0.67,
    0.63, 0.7, 0.73, 0.73, 0.72, 0.75, 0.73, 0.73,
    0.75, 0.7, 0.75, 0.73, 0.63, 0.7, 0.65, 0.7,
    0.68, 0.65, 0.73, 0.7, 0.7, 0.68, 0.68
]

# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.67

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Labels and title
plt.xlabel("Number of Features", fontsize=36)
plt.ylabel("Accuracy", fontsize=36)
plt.title("CNS", fontsize=36)
plt.legend(fontsize=22)
plt.ylim(0.60,1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


###########  GLI #######################
# Accuracies from the prompt
accuracies = [0.86, 0.85, 0.88, 0.82, 0.87, 0.89, 0.88, 0.84, 0.88, 0.88, 0.87, 0.85, 0.87, 0.85, 0.88, 0.91, 0.89, 0.88, 0.89, 0.87, 0.87, 0.91, 0.91, 0.91, 0.89, 0.89, 0.91, 0.89, 0.91, 0.89, 0.89, 0.89, 0.89, 0.89, 0.92, 0.89, 0.89, 0.91, 0.91, 0.91, 0.89, 0.92, 0.93, 0.91, 0.89, 0.91, 0.89, 0.94, 0.91, 0.91]
# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.86

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Labels and title
plt.xlabel("Number of Features", fontsize=36)
plt.ylabel("Accuracy", fontsize=36)
plt.title("GLI", fontsize=36)
plt.legend(fontsize=22)
plt.ylim(0.75,1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

######################ُSMK#######################

accuracies = [0.6, 0.68, 0.6, 0.63, 0.64, 0.63, 0.63, 0.67, 0.66, 0.63, 0.7, 0.63, 0.61, 0.63, 0.58, 0.68, 0.67, 0.67, 0.64, 0.66, 0.67, 0.66, 0.7, 0.73, 0.68, 0.67, 0.67, 0.66, 0.67, 0.66, 0.68, 0.64, 0.7, 0.66, 0.69, 0.68, 0.66, 0.69, 0.7, 0.73, 0.72, 0.71, 0.71, 0.71, 0.74, 0.75, 0.74, 0.72, 0.75, 0.71]
# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.55

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Labels and title
plt.xlabel("Number of Features", fontsize=36)
plt.ylabel("Accuracy", fontsize=36)
plt.title("SMK", fontsize=36)
plt.legend(fontsize=22)
plt.ylim(0.4,0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

##############Covid########################
accuracies = [0.55, 0.52, 0.59, 0.6, 0.63, 0.65, 0.65, 0.64, 0.64, 0.63, 0.63, 0.65, 0.65, 0.66, 0.65, 0.67, 0.66, 0.68, 0.67, 0.65, 0.63, 0.65, 0.66, 0.65, 0.59, 0.58, 0.59, 0.56, 0.6, 0.59, 0.59, 0.59, 0.61, 0.62, 0.61, 0.66, 0.69, 0.69, 0.66, 0.66, 0.69, 0.69, 0.67, 0.68, 0.68, 0.67, 0.65, 0.67, 0.66, 0.63]
# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.68

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Labels and title
plt.xlabel("Number of Features", fontsize=36)
plt.ylabel("Accuracy", fontsize=36)
plt.title("Covid-19", fontsize=36)
plt.legend(fontsize=22)
plt.ylim(0.5,0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##############Leukemia###########################
accuracies = [0.82, 0.93, 0.96, 0.94, 0.9, 0.9, 0.92, 0.88, 0.88, 0.9, 0.93, 0.92, 0.86, 0.89, 0.9, 0.89, 0.88, 0.83, 0.89, 0.86, 0.89, 0.83, 0.9, 0.9, 0.86, 0.9, 0.88, 0.83, 0.86, 0.88, 0.9, 0.9, 0.9, 0.9, 0.9, 0.89, 0.9, 0.86, 0.89, 0.89, 0.86, 0.9, 0.88, 0.89, 0.89, 0.88, 0.9, 0.89, 0.9, 0.88]
# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.82

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Labels and title
plt.xlabel("Number of Features", fontsize=36)
plt.ylabel("Accuracy", fontsize=36)
plt.title("Leukemia", fontsize=36)
plt.legend(fontsize=22)
# plt.ylim(0.7,1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#################MLL###########################
accuracies = [0.86, 0.92, 0.93, 0.96, 0.94, 0.96, 0.97, 0.94, 0.96, 0.96, 0.96, 0.94, 0.96, 0.94, 0.96, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97, 0.96, 0.96, 0.94, 0.93, 0.93, 0.94, 0.93, 0.97, 0.94, 0.94, 0.96, 0.93, 0.93, 0.93, 0.93, 0.93, 0.94, 0.96, 0.9, 0.9, 0.97, 0.96, 0.94, 0.94, 0.96, 0.93, 0.89, 0.93, 0.97]
# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.92

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Labels and title
plt.xlabel("Number of Features", fontsize=36)
plt.ylabel("Accuracy", fontsize=36)
plt.title("MLL", fontsize=36)
plt.legend(fontsize=22)
# plt.ylim(0.5,0.7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##############SRBCT#######################
accuracies = [0.61, 0.81, 0.82, 0.83, 0.84, 0.89, 0.89, 0.87, 0.89, 0.87, 0.89, 0.87, 0.88, 0.89, 0.88, 0.89, 0.93, 0.88, 0.9, 0.88, 0.86, 0.84, 0.87, 0.87, 0.83, 0.88, 0.86, 0.89, 0.87, 0.89, 0.86, 0.87, 0.87, 0.88, 0.88, 0.88, 0.87, 0.92, 0.9, 0.89, 0.87, 0.8, 0.86, 0.87, 0.89, 0.9, 0.86, 0.87, 0.9, 0.77]
# X values: number of features
num_features = np.arange(1, len(accuracies) + 1)

# Average accuracy
avg_acc = np.mean(accuracies)

# Initial accuracy (given)
init_acc = 0.80

# Plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', linestyle='-', color='b', label="Accuracy by top features")

# Horizontal lines
plt.axhline(y=avg_acc, color='g', linestyle='-', linewidth=2, label=f"Average Accuracy ({avg_acc:.2f})")
plt.axhline(y=init_acc, color='r', linestyle='-', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Bigger ticks
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)

# Labels and title
plt.xlabel("Number of Features", fontsize=36)
plt.ylabel("Accuracy", fontsize=36)
plt.title("SRBCT", fontsize=36)
plt.legend(fontsize=22)
# plt.ylim(0.5,0.7)
# plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


############### DFFS analysis

##### Colon
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Data
accuracies = [0.77, 0.87, 0.87, 0.89, 0.90]
num_features = np.arange(1, 6)
init_acc = 0.84

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Increase figure size to create significantly more space between labels
plt.figure(figsize=(12, 12))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=38)

# Adjust y-axis (accuracy) ticks and labels to have more space
ax.set_yticks(np.arange(0.70, 0.95, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.70, 0.95, 0.1)], fontsize=16)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=32
)
plt.title("Colon", fontsize=48, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("radar_plot_more_spaced.png")

############## CNS###########################

import matplotlib.pyplot as plt
import numpy as np
from math import pi

# New Data
accuracies = [0.57, 0.63, 0.6, 0.57, 0.57]
num_features = np.arange(1, 6)
init_acc = 0.67

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Increase figure size to create significantly more space between labels
plt.figure(figsize=(12, 12))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=38)

# Adjust y-axis (accuracy) ticks and labels to have more space
ax.set_yticks(np.arange(0.5, 0.75, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.5, 0.75, 0.1)], fontsize=20)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=32
)
plt.title("CNS", fontsize=48, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


################ GLI ####################


# Data from previous turn
accuracies = [0.86, 0.93, 0.94, 0.95, 0.94]
num_features = np.arange(1, 6)
init_acc = 0.86

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Use a large figure size for clarity
plt.figure(figsize=(14, 14))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=38)

# Increase space between radars by adjusting y-axis ticks
ax.set_yticks(np.arange(0.80, 1.05, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.80, 1.05, 0.1)], fontsize=18)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=38
)
plt.title("GLI", fontsize=56, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

##############  SMK ###############



# Data
accuracies = [0.60, 0.64, 0.70, 0.72, 0.68]
num_features = np.arange(1, 6)
init_acc = 0.55

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Use a large figure size for clarity
plt.figure(figsize=(14, 14))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=38)

# Adjust y-axis (accuracy) ticks and labels to have more space
ax.set_yticks(np.arange(0.55, 0.75, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.55, 0.75, 0.1)], fontsize=24)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=38
)
plt.title("SMK", fontsize=56, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


########### Covid_19 ##########################


# Data from previous turn
accuracies = [0.53, 0.61, 0.66, 0.62, 0.67]
num_features = np.arange(1, 6)
init_acc = 0.68

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Use a large figure size for clarity
plt.figure(figsize=(16, 16))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=38)

# Increase space between radars by adjusting y-axis ticks step
ax.set_yticks(np.arange(0.50, 0.75, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.50, 0.75, 0.1)], fontsize=26)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=42
)
plt.title("Covid_19", fontsize=56, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


############  Leukemia  ########################

accuracies = [0.79, 0.90, 0.92, 0.90, 0.92]
num_features = np.arange(1, 6)
init_acc = 0.82

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Increase figure size for more space and better clarity
plt.figure(figsize=(18, 18))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=42)

# Adjust y-axis (accuracy) ticks and labels to have more space
ax.set_yticks(np.arange(0.75, 0.95, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.75, 0.95, 0.1)], fontsize=32)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=54
)
plt.title("Leukemia", fontsize=70, pad=20)


################## MLL #########################

# New Data
accuracies = [0.82, 0.93, 0.93, 0.96, 0.97]
num_features = np.arange(1, 6)
init_acc = 0.92

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Use a large figure size for clarity
plt.figure(figsize=(16, 16))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=38)

# Adjust y-axis (accuracy) ticks and labels to have more space
ax.set_yticks(np.arange(0.80, 1.00, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.80, 1.00, 0.1)], fontsize=24)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=42
)
plt.title("MLL", fontsize=58, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

##########  SRBCT #############################


# New Data
accuracies = [0.65, 0.67, 0.72, 0.75, 0.83]
num_features = np.arange(1, 6)
init_acc = 0.80

# Update labels to handle singular and plural "Feature"
labels = [f"{n} Feature" if n == 1 else f"{n} Features" for n in num_features]

# Close the polygon for radar plot
values = accuracies + [accuracies[0]]
angles = np.linspace(0, 2 * pi, len(values), endpoint=True)

# Use a large figure size for clarity
plt.figure(figsize=(16, 16))
ax = plt.subplot(111, polar=True)

# Plot accuracy polygon
ax.plot(angles, values, 'b-o', linewidth=2, label="Accuracy with DFFS")
ax.fill(angles, values, 'skyblue', alpha=0.4)

# Plot initial accuracy as a circle
ax.plot(angles, [init_acc] * len(values), 'r--', linewidth=2, label=f"Initial Accuracy ({init_acc:.2f})")

# Set the radar labels and their positions
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=38)

# Adjust y-axis (accuracy) ticks and labels to have more space
ax.set_yticks(np.arange(0.60, 0.90, 0.1))
ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.60, 0.90, 0.1)], fontsize=22)

# Move legend under the radar chart and adjust layout
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=28)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,              # stack items vertically
    frameon=False,
    fontsize=42
)
plt.title("SRBCT", fontsize=56, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



######## Box plot analysis #####################


# Data provided
data = {
    "Colon": [1625,443,1654,347,1687,281,1661,351,1546,427,1444,520,1607,289,1523,
              268,1773,236,1714,338,1558,510,1737,284,1543,1769,1731,1409,1319,1359,
              1277,1670,1713,1440,1716,1738,1487,1764,359,1570,1732,493,643,1416,
              1137,1369,1309,559,1535,477,358,1389,1586,573,346,1555,1503,1800,1563,
              462,1676,393],
    "CNS": [1080,1206,1120,1295,1117,1133,1073,1050,1022,1430,1185,880,1205,899,960,
            1211,937,1093,1100,1018,997,5971,5882,5734,5986,5799,6022,5931,5915,5617,
            5850,5946,5703,5885,5521,5768,5731,5712,5976,5762,5844,5731,5851,5731,
            6060,6011,5802,5871,5568,5952,6021,6040,5799,6039,5627,5906,5888,6021,
            6039,5749],
    "GLI": [5482,5591,5812,4972,4912,5653,4612,4729,5370,5498,5653,5589,5536,5409,
            3980,4923,4803,4798,4747,5151,5116,4789,4601,5263,5187,5883,18254,17497,
            18295,18500,16596,18465,18623,17897,17854,17177,17081,16847,16703,18518,
            17203,18027,17655,17534,17350,17393,18071,16589,17203,17998,18288,18357,
            18116,16490,18537,17230,17680,17845,16906,17205,17984,16577,17988,16926,
            18000,18125,17814,18044,17518,17495,18230,18398,17517,17980,16578,18346,
            18458,17309,17583,17147,17617,18062,17099,18022,18593],
    "SMK": [9154,12316,12365,12160,12020,12423,12413,12183,12861,12932,12230,11869,
            12481,12552,10344,12257,12099,11767,12672,12406,12593,11784,12233,12051,
            12106,12559,12040,12559,12881,12267,12569,11844,12648,12044,12696,12062,
            11388,12606,12313,9494,12320,11284,12102,11225,12415,12272,12463,9733,
            12717,11025,11161,8973,12186,12452,10349,10120,9807,11608,10536,12032,
            10313,12101,10460,12056,11442,11305,9927,11545,11076,9741,10772,10955,
            10424,11996,11180,12037,11805,11561,12594,10466,11722,10411,10045,11220,
            9245,11606,11511,11436,11819,11805,7212,7814,8472,7657,7806,7249,9060,
            7806,8780,8589,7621,8085,7812,7791,7202,8220,7455,8085,9427,7720,8493,
            9133,10009,7932,7758,7962,9599,9354,8923,10354,8862,10718,8714,10822,
            9211,9516,10478,9023,8677,10829,10169,8770,10712,8720,8070,8685,9815,
            8739,9447,8667,9792,8933,10682,9381,8715,10046,10783,9913,8897,8897,
            9773,9590,8395,7821,10335,10343,10159,7880,9766,8033,9276,10114,10424,
            7951,8995,7769,8787,9376,8139,9045,9446,8878,9609,7748,9214,8008,8252,
            9659,8811,7939,8099,10215,9548,8196,10396,9071,10236],
    "Covid_19": [4287, 3913, 6353, 3115, 8678, 1938, 2383, 4444, 2475, 5335, 6568,
              6790, 8406, 1579, 5452, 4023, 6493, 4002, 9156, 8234, 8153, 6683,
              1790, 6494, 1122, 1307, 1676, 6356, 5994, 6130, 4858, 4598, 6136, 5955,
              5559, 4561, 5646, 4701, 6348, 7487, 2675, 5987, 6249, 6009, 5152, 6071,
              5603, 8337, 4737, 5697, 5788, 8918, 3904, 5769, 7943, 6652, 9073, 4422,
              5525, 5194, 4321, 5872, 5533, 5182, 8599, 3114, 2611, 5587, 3793, 5268, 
              1921, 5166, 5901, 5531, 3880, 6483, 2326, 8579, 4613, 5263, 2458, 4621, 
              5817, 6239, 4987, 5319, 4778, 6125, 8758, 2532, 4577, 3729, 5446, 4150,
              6047, 1624, 7967, 4798, 8966, 8196, 6246, 4880, 6209, 9097, 4699, 2248,
              4981, 7240, 4013, 6343, 6691, 2927, 1652, 5530, 8434, 3220, 2394, 4154,
              6851, 5454, 4541, 4085, 896, 5090, 2971, 8392, 6149, 8713, 4575, 5074, 
              6401, 9039, 6263, 6644, 8597, 9085, 1911, 4418, 3265, 6122, 7527, 1397,
              3091, 2712, 3770, 8758, 2197, 6429, 6479, 9117, 5506, 8303, 1279, 1225,
              4128, 5524, 1833, 2008, 8871, 5953, 4135, 8932, 4069, 4717, 4836, 5107, 
              1055, 5043, 6158, 5422, 5491, 6409, 9217, 5225, 4405, 8306, 5605, 8696,
              8936, 8935, 8795, 9356, 8849, 9168, 8987, 8809, 8222, 8681, 4749, 8693,
              8862, 9142, 4936, 9265, 8704, 4593, 8061, 9052, 9299, 7234, 8948, 8567,
              8861, 8096, 9353, 8983, 9069, 8360, 3900, 9300, 9176, 7308, 8256, 4295,
              8303, 8578, 8020, 8865, 7854, 8424, 8408, 9201, 8856, 9559, 8893, 7710, 
              9207, 8962, 9087, 9404, 8979, 9008, 9094, 6165],
    "Leukemia": [4754,4685,5255,4921,5054,5129,4990,5355,5151,4957,4336,2297,2243,
                 1910,2404,1963,5104,4864,1782,2199,5154,1791,2163,2069,2376,2511,
                 1948,1690,74,4962,5057,5204,5196,5128,4686,97,171,4650,5217,119,
                 4554,4381,186,88,102,5109,5243,102,5149,5042,4210,5149,5222,4697,
                 5036,4954,113,4916,5052,5225,4378,2274,1729,2298,2274,2325,2637,
                 2304,2161,2374,2378,2287],
    "MLL": [4573,4813,4811,4470,2218,1976,2202,5794,6851,6470,6831,6160,7045,6567,
            6846,4973,5139,4161,4745,4673,5029,5214,4973,4987,4776,5046,4920,4427,
            4812,4191,5036,4022,4971,5202,4328,2531,2545,2324,2555,2651,2700,2351,
            2436,2762,2093,2644,2168,2326,2400,2186,2208,1991,6685,6324,6351,6508,
            6207,6307,6232,5062,6116,6318,6218,6304,6717,7169,6882,6866,7056,5771,
            6594,6918],
    "SRBCT": [990,984,1013,921,1010,1062,940,1095,1105,1135,1120,1099,1062,1079,
              910,913,920,1062,1068,910,961,1012,1059,137,143,157,158,176,119,165,
              163,385,384,384,456,485,425,428,447,451,461,483,479,800,812,785,737,
              823,778,816,881,801,820,877,831,881,799,797,832,803,824,833,891,411,
              821,345,961,948,134,976,854,1040,1123,1078,895,123,866,410,380,446,
              394,132,1010]
}

# Original feature sizes
original_sizes = {
    "Colon": 2000,
    "CNS": 7129,
    "GLI": 22283,
    "SMK": 19993,
    "Leukemia": 7129,
    "Covid_19": 15979,
    "MLL": 12582,
    "SRBCT": 2308
}

# Compute proportions
proportions = {k: np.array(v) / original_sizes[k] for k, v in data.items()}

# Plot
plt.figure(figsize=(12,6))
plt.boxplot(proportions.values(), labels=proportions.keys(), patch_artist=True)
plt.ylabel("Proportion of selected features",fontsize=22)
plt.xlabel("Datasets",fontsize=22)
# plt.title("Distribution of Selected Features per Sample (as proportion of total)")
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


##############  SFFS  top features  #################
from sklearn.model_selection import LeaveOneOut
Top = 50 
accuracies = []

# Select the top 50 columns from X_reduced
# Assuming X_reduced is a pandas DataFrame, we can select the first 50 columns.
X_top_50 = X_reduced.iloc[:, :Top]

# Initialize the Decision Tree Classifier with a fixed random_state
dectree = tree.DecisionTreeClassifier(random_state=42)

# Use LeaveOneOut cross-validation strategy
loo = LeaveOneOut()

# Loop through the columns from 1 to 50
for k in range(1, Top + 1):
    # Select the first k columns from the top 50
    X_selected = X_top_50.iloc[:, :k]
    y_pred = []
    y_true = []

    # Iterate through the Leave-One-Out splits
    for train_index, test_index in loo.split(X_selected):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit the model and make a prediction
        dectree.fit(X_train, y_train)
        y_pred.append(dectree.predict(X_test)[0])
        y_true.append(y_test.iloc[0])

    # Calculate and store the accuracy for the current number of features
    accuracy = accuracy_score(y_true, y_pred)
    accuracies.append(accuracy)

# Round the accuracies for display
accuracies_rounded = [round(acc, 2) for acc in accuracies]
print(f"Accuracies for top 1..{Top} features:", accuracies_rounded)
print("Mean accuracy:", round(np.mean(accuracies_rounded), 2))
print("Top 10 mean accuracy:", round(np.mean(accuracies_rounded[0:10]), 2))
print("Top 20 mean accuracy:", round(np.mean(accuracies_rounded[0:20]), 2))
print("Top 30 mean accuracy:", round(np.mean(accuracies_rounded[0:30]), 2))
print("Top 40 mean accuracy:", round(np.mean(accuracies_rounded[0:40]), 2))
print("Top 50 mean accuracy:", round(np.mean(accuracies_rounded[0:50]), 2))


############# Coparison of SFFSCPR and SFFS 
data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.77, 0.77, 0.77, 0.78, 0.79],
    # 'SFFS-CPR': [0.75, 0.79, 0.82, 0.82, 0.82],
    'CPR' : [0.76, 0.75, 0.77, 0.77, 0.79],
    'MPR': [0.79,0.79,0.78,	0.77,0.77]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.84

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.6, 0.85)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("Colon",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()

###  CNS ####
#############
data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.59, 0.59, 0.57, 0.55, 0.53],
    # 'SFFS-CPR': [0.74, 0.77, 0.75, 0.74, 0.73],
    'CPR':[0.75,	0.73,	0.73,	0.72,	0.73],
    'MPR':[0.75,	0.75,	0.74,	0.73,	0.7]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.67

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.4, 0.8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("CNS",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()

#### GLI ######
###############
data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.81, 0.80, 0.79, 0.79, 0.78],
    # 'SFFS-CPR': [0.88, 0.88, 0.87, 0.87, 0.86],
    'CPR':[0.87,	0.87,	0.87,	0.87,	0.87],
    'MPR':[0.84,	0.84,	0.85,	0.86,	0.87]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.86

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.7, 1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("GLI",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()


#### SMK ######
###############
data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.64, 0.62, 0.63, 0.64, 0.63],
    # 'SFFS-CPR': [0.64, 0.64, 0.65, 0.66, 0.67],
    'CPR':[0.68,	0.69,	0.68,	0.69,	0.69],
    'MPR':[0.66,	0.65,	0.65,	0.65,	0.66]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.55

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.5, 0.8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("SMK",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()

##############
#### Covid####
data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.61, 0.66, 0.67, 0.67, 0.67],
    # 'SFFS-CPR': [0.61, 0.63, 0.63, 0.63, 0.64],
    'CPR':[0.59,	0.62,	0.63,	0.64,	0.63],
    'MPR':[0.59,	0.61,	0.63,	0.64,	0.65]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.68

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.55, 0.7)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("Covid_19",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()


##############
#### Leukemia ####
data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.81, 0.84, 0.84, 0.85, 0.86],
    # 'SFFS-CPR': [0.90, 0.89, 0.89, 0.89, 0.89],
    'CPR':[0.89,	0.88,	0.88,	0.88,	0.88],
    'MPR':[0.84,	0.85,	0.85,	0.86,	0.86]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.82

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.7, 0.95)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("Leukemia",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()


##############
#### MLL ####
data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.8, 0.84, 0.86, 0.87, 0.89],
    # 'SFFS-CPR': [0.94, 0.95, 0.95, 0.94, 0.94],
    'CPR':[0.93,	0.94,	0.94,	0.95,	0.95],
    'MPR':[0.88,	0.90,	0.90,	0.92,	0.93]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.92

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.7, 1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("MLL",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()
##############
### SRBCT ####

data = {
    'Features': [10, 20, 30, 40, 50],
    # 'SFFS': [0.76, 0.81, 0.83, 0.84, 0.85],
    # 'SFFS-CPR': [0.83, 0.86, 0.86, 0.87, 0.87],
    'CPR':[0.81,	0.84,	0.85,	0.85,	0.87],
    'MPR':[0.79,	0.83,	0.81,	0.85,	0.84]
}
df_data = pd.DataFrame(data)

# Original accuracy for the Colon dataset
original_accuracy = 0.80

# Create the plot
plt.figure(figsize=(10, 6))

# Plot SFFS and SFFS-CPR accuracies
# plt.plot(df_data['Features'], df_data['SFFS'], marker='o', label='SFFS', color='blue',markersize=14)
# plt.plot(df_data['Features'], df_data['SFFS-CPR'], marker='x', label='SFFS-CPR', color='red',markersize=14 )
plt.plot(df_data['Features'], df_data['CPR'], marker='s', label='CPR', color='purple', markersize=14)
plt.plot(df_data['Features'], df_data['MPR'], marker='^', label='MPR', color='orange', markersize=14)
# Plot original accuracy as a horizontal dashed line
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Before feature selection')

# Add titles, labels, and legend
# plt.title('Accuracy for Colon Dataset with SFFS and SFFS-CPR')
plt.xlabel('Number of features', fontsize=36)
plt.ylabel('Accuracy',fontsize =36)
plt.xticks(df_data['Features'], fontsize=26)
plt.yticks(fontsize=26)
plt.ylim(0.7, 1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=22)
plt.title("SRBCT",fontsize =36)
plt.tight_layout()

# Display the plot
plt.show()







#####################  INITIAL ACCURACY WITH ALL AVAILABLE FEATURES
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

# Initialize lists to collect predictions
y_true, y_pred = [], []

# Decision Tree classifier with fixed random state
clf = tree.DecisionTreeClassifier(random_state=42)

# Perform LOOCV
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    y_true.append(y_test.values[0])
    y_pred.append(y_hat[0])

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")




##################### ACCURACY WITH TOP FEATURES BY SFFS

Top = 5
new = significant_features
y = y.astype(int)
y_true, y_pred = [], []

dectree = tree.DecisionTreeClassifier(random_state=42)

for i in range(len(X)):
    # pick top features for this sample
    new_n = new[i][0][:Top]
    feature_indices = np.array(new_n).flatten()

    # build dataset with selected features
    X_new = X.iloc[:, feature_indices]

    # split train/test (leave-one-out)
    X_test = X_new.iloc[i:i+1, :]
    X_train = X_new.drop(index=i)
    y_test = y.iloc[i]
    y_train = y.drop(index=i)

    # fit model
    dectree.fit(X_train, y_train)

    # predict and collect results
    y_true.append(y_test)
    y_pred.append(dectree.predict(X_test)[0])

# compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


######################per,rec,f1score


import matplotlib.pyplot as plt
import numpy as np

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.84, 0.84, 0.84, 0.84]
dffs = [0.87, 0.87, 0.87, 0.87]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("Colon", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0.5,0.9)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()


#####################CNS#######################


metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.67,	0.67,	0.67,	0.67]
dffs = [0.58,	0.54,	0.54,	0.54]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("CNS", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0,1)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()

######################GLI#######################


metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.86,	0.86,	0.86,	0.86]
dffs = [0.91,	0.92,	0.86,	0.88]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("GLI", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0.5,1)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()

################SMK################################

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.55,	0.54,	0.54,	0.54]
dffs = [0.70,	0.70,	0.7,	0.7]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("SMK", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0,0.8)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()

#######################Covid###############

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.68,	0.67,	0.68,	0.67]
dffs = [0.72,	0.69,	0.67,	0.68]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("Covid-19", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0.3,0.8)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()


#################Leukemia####################


metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.82,	0.82,	0.82,	0.82]
dffs = [0.90,	0.89,	0.90,	0.89]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("Leukemia", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0.5,1)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()


#################MLL#########################

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.92,	0.93,	0.92,	0.92]
dffs = [0.96,	0.97,	0.95,	0.96]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("MLL", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0.7,1)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()

####################SRBCT#############################


metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
initial = [0.80,	0.81,	0.80,	0.80]
dffs = [0.88,	0.86,	0.90,	0.87]

x = np.arange(len(metrics))  # positions for bars
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, initial, width, label="Before feature selection")
bars2 = ax.bar(x + width/2, dffs, width, label="After DFFS")

# Labels and formatting
ax.set_ylabel("Percentage", fontsize=32)
ax.set_title("SRBCT", fontsize=32)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18)
plt.yticks(fontsize=16)

ax.set_ylim(0.7,1)  # since scores are between 0 and 1
ax.legend(fontsize=18, loc="lower right")  # legend in bottom-right

plt.tight_layout()
plt.show()





################ Correlation analysis

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
        # To avoid divide-by-zero
        score = numerator / (denominator + 1e-6)
        scores.append(score)
    
    return np.array(scores)

# Calculate scores and get top 50 features
scores = fisher_score(X, y)
Fisher = np.argsort(scores)[-50:][::-1]
ranked_features = X.columns[Fisher]

print("Top 50 feature indices:", Fisher)
print("Top 50 feature names:", ranked_features.tolist())



####CPR

scores = np.array(sortings["aggregated"])
feature_names = np.array(X.columns)  # or your list of feature names
# Get indices of top 50 scores
top50_indices = np.argsort(scores)[-50:][::-1]

# Get top 50 feature names
ranked_features = feature_names[top50_indices].tolist()
selected_X=X[ranked_features]


### SFFS

selected_X=X[np.array(ranked_features)[:50]]


import numpy as np
import pandas as pd

# Assuming X is a pandas DataFrame and ranked_features contains the top features
selected_X = X[ranked_features]

# Compute correlation matrix
corr_matrix = selected_X.corr().values

# Extract upper triangle without the diagonal
upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
upper_tri_values = corr_matrix[upper_tri_indices]

# Calculate average pairwise correlation
avg_pairwise_corr = np.mean(upper_tri_values)
avg_absolute_pairwise_corr = np.mean(np.abs(upper_tri_values))
# print("Average pairwise correlation between top 50 features:", round(avg_pairwise_corr,2))
print("Average ABSOLUTE pairwise correlation between top 50 features:", round(avg_absolute_pairwise_corr, 2))