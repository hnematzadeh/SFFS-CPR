# SFFS-CPR

SFFS-CPR is a hybrid method that integrates two filters: Static Feature-wise Feature Selection (SFFS), which generalizes the Dynamic Feature-wise Feature Selection (DFFS), and frequency-based Constructive Pattern Recognition (CPR).

--> The file **significant_features.pkl** contains the set of features dynamically selected by the Dynamic Feature-wise Feature Selection (DFFS) method for each dataset. To save you time, we have prepared this file in advance. Therefore, you can skip lines 112–206. It is in the "DFFS selected features folder". Simply update the file path in line 214 to match your local system, and load the corresponding significant_features.pkl file using lines 219–220. However, if you wish to apply DFFS to your own dataset, you must first load your dataset and then execute lines 219–220. The datasets used in the paper have already been preloaded in advance, spanning lines 18–105.

--> SFFS utilizes the significant_features—a set of features selected by DFFS—and generalizes them by selecting a subset that is applicable across the entire dataset, as implemented in lines 226–253.

--> CPR begins at line 262, where the response variable is reordered based on the sorted values of each feature, as implemented in lines 262–273. CPR performs feature sorting in lines 280–355.

--> **Plots and visualizations of the SFFS-CPR ** method are presented , where the accuracy of the top 50 features is computed

    - The accuracy of the top 50 features is computed in lines 364–412, corresponding to Figure 4 in the paper.
    
    - The accuracy, precision, recall, and F1 score of DFFS are evaluated through per-sample feature selection, as implemented in lines 425–458, corresponding to Figure 6 in the paper.
    
    - The accuracy of DFFS using the top 5 features is computed in lines 464–482, corresponding to Figure 5 in the paper.
    

    
