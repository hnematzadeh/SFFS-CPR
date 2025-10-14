# SFFS-CPR

SFFS-CPR is a hybrid method that integrates two filters: Static Feature-wise Feature Selection (SFFS), which generalizes the Dynamic Feature-wise Feature Selection (DFFS), and frequency-based Constructive Pattern Recognition (CPR).

--> The file **significant_features.pkl** contains the set of features dynamically selected by the Dynamic Feature-wise Feature Selection (DFFS) method for each dataset. To save you time, we have prepared this file in advance. Therefore, you can skip lines 112–206. It is in the "DFFS selected features folder". Simply update the file path in line 214 to match your local system, and load the corresponding significant_features.pkl file using lines 219–220. However, if you wish to apply DFFS to your own dataset, you must first load your dataset and then execute lines 219–220. The datasets used in the paper have already been preloaded in advance, spanning lines 18–105.

--> SFFS utilizes the significant_features—a set of features selected by DFFS—and generalizes them by selecting a subset that is applicable across the entire dataset, as implemented in lines 226–253.

--> CPR begins at line 262, where the response variable is reordered based on the sorted values of each feature, as implemented in lines 262–273. CPR performs feature sorting in lines 280–355.

**Supplementary codes**

--> Lines 486–850 compute the accuracy of the top 50 features selected by existing filters, including Fisher Score, Mutual Information, ReliefF, and others.

--> Lines 854–867 compute the prediction accuracy using all available features, evaluated with the Leave-One-Out strategy.

--> The correlation analysis presented in the paper is implemented in lines 2318–2383.

**Plots and visualizations of the SFFS-CPR:** 

    - The accuracy of the top 50 features is computed in lines 364–412 and plotted in lines  871-1139, corresponding to Figure 4 in the paper.
    
    - The accuracy, precision, recall, and F1 score of DFFS are evaluated through per-sample feature selection, as implemented in lines 425–458, and plotted in lines 2080-2310 corresponding to Figure 6 in the paper.
    
    - The accuracy of DFFS using the top 5 features is computed in lines 464–482, and plotted in lines 1144 1517, corresponding to Figure 5 in the paper.

    - The box plot analysis is plotted in lines 1525-1622, corresponding to Figure 7 in the paper.

    - Lines 1672–1992 generate plots corresponding to Figure 2 and Figure 3 in the paper.
    

    
