from pathlib import Path
import pandas as pd
from math import ceil, prod
from scipy.special import comb
from numpy import log2
import bnlearn as bn

DATASET_PATH = "sachs/sachs.2005.discrete.txt"
dataset = pd.read_csv(Path(__file__).parent.parent / "./data/" / DATASET_PATH, sep="\t")

# Fix variable naming inconsistencies
dataset.rename(columns=str.capitalize, inplace=True)
dataset.rename(columns={'Plc': 'Plcg'}, inplace=True)

def _arity(var: str) -> int:
    """
    The arity of a variable or number of unique values the variable can take on

    :param var: A variable in a dataset
    :return: The arity of the variable
    """
    return len(dataset[var].unique()) # .unique() is effectively the same as set()

def _get_parents(adjmat: pd.DataFrame, var: str) -> list[str]:
    """
    Get a list of parent variables for a particular variable in a BN

    :param adjmat: Adjacency matrix describing the BN model structure
    :param var: Variable to get the parents for
    :return: A list of parent variables by name
    """
    # For the column defined by var, return the indices of all rows mapping to True
    return adjmat.index[adjmat[var] == True].tolist()

def max_likelihood_per_var(var: str, parents: list[str], dataset: pd.DataFrame) -> float:
    """
    K2 score for a node in a BN given the dataset and the parents of the node

    :param var: The variable
    :param parents: The parents of that variable
    :param dataset: The dataset
    :return: The maximum likelihood score of that node/var given its parents
    """

    # Get the possible values that the variable can take on
    var_values = dataset[var].unique()
    log_score = 0.0

    # Group the dataset by the parents of the variable
    groups = dataset.groupby(parents) if parents else [(None, dataset)]

    for _, group in groups:
        # N(pa_i)
        n_pa = len(group)
        for value in var_values:
            # Get number of samples where X_i = x_i and parents take on a parent instantiation
            n_x_pa = group[var].value_counts().get(value, 0)
            # Skip if 0. By convention, 0 * log2(0) = 0
            if n_x_pa == 0:
                continue
            # Compute log score
            log_score += n_x_pa * log2(n_x_pa / n_pa)
    return log_score

def max_likelihood(model: dict, dataset: pd.DataFrame) -> float:
    """
    Max_likelihood score for an entire model given a dataset

    :param model: The BN model
    :param data: The dataset
    :return: The maximum likelihood score
    """

    adjmat = model["adjmat"]
    return -sum(
        max_likelihood_per_var(var, _get_parents(adjmat, var), dataset)
        for var in dataset.columns
    )

def _mdl_per_node(var: str, parents: list[str], dataset: pd.DataFrame) -> float:
    """
    MDL - Lam and Bacchus, taken from section 3.4.3 of O'Donnell's thesis
    Computes MDL for a node and its parents given a dataset. Implementation can be read off from the equation

    :param var: The variable
    :param parents: The parents of that variable
    :param dataset: The dataset
    :return: The MDL score per node
    """
    return (
            len(parents) * log2(len(dataset.columns)) +
            ceil(log2(_arity(var) - 1)) *
            prod(_arity(pa_var) for pa_var in parents)
    )

def mdl(model: dict, dataset: pd.DataFrame) -> float:
    """
    MDL - Lam and Bacchus, taken from section 3.4.3 of O'Donnell's thesis
    Computes minimal description length for a model given a dataset.
    This is done by computing MDL for each node and its parents using _mdl_per_node
    and computing the maximum likelihood score of the network

    :param model: The BN model
    :param dataset: The dataset
    :return: The MDL score
    """

    adjmat = model["adjmat"]
    return sum(
        _mdl_per_node(var, _get_parents(adjmat, var), dataset)
        for var in dataset.columns
    ) + max_likelihood(model, dataset)


if __name__ == "__main__":
    model = bn.import_DAG('sachs')

    # Fix variable naming inconsistencies
    model["adjmat"].columns = model["adjmat"].columns.str.capitalize()
    model["adjmat"].index = model["adjmat"].index.str.capitalize()
    print(mdl(model, dataset))
