# causal-mml
Code repository for an FIT3144 research project on causal structure learning using Minimal Message Length. For now, there will only be implementations of a variety of scoring metrics used for structure learning, most of which are outlined in O'Donnell's 2010 thesis 'Flexible Causal Discovery with MML'.

### Usage
Cd into the repo and run the data_fetcher executable. The script can be changed to retrieve any other dataset from `cmu-phil/example-causal-datasets`. Just make sure the data is discrete, as most scoring metrics investigated in this repo require the data to be discrete.

## Score based learning
Score based learning is an approach used to learn a Bayesian/Causal network given some data. The process generally involves two parts: The use of a scoring function and a search phase. Scoring functions are used as a way of evaluating a network structure given a dataset, while the search is a way of finding a network structure with an optimal score. This search is performed over the space of all possible structures, given the dataset. Since these problems can be viewed independently, we will first examine a few scoring metrics.

### K2 metric

