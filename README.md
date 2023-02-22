# Linking prediction
## Dialogue Graph Auto Construction based on data with a regular structure

Goal: Extract regular structures from the data by building a dialogue graph
    
Tasks: 
* Cluster dialog data using embeddings of pre-trained models (BERT, ConveRT, S-BERT…)
* Evaluate the quality of clustering using intent’s labeling of Multi-WoZ dataset 
* Linking clusters of dialogs using naive approaches (Estimation of Probabilities by Frequency Models)
* Try other approaches (Deep Neural Networks) for linking clusters and improve the naive approach

Approaches for linking clusters:
- Wooden (CatBoost)
- Message Passing (GAT)
- GTN
- HGT
