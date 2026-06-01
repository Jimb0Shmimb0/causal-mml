from pathlib import Path
import pandas as pd
from scipy.special import gammaln
from scipy.special import comb
from numpy import log
import bnlearn as bn

DATASET_PATH = "sachs/sachs.2005.discrete.txt"
dataset = pd.read_csv(Path(__file__).parent.parent / "./data/" / DATASET_PATH, sep="\t")

# Preprocessing for this particular dataset: Fix variable naming inconsistencies
dataset.rename(columns=str.capitalize, inplace=True)
dataset.rename(columns={'Plc': 'Plcg'}, inplace=True)

def _num_dags(n: int) -> int:
    """
    Recursive formula for the number of possible DAGs that can be produced from n variables
    :param n: number of variables/nodes
    :return: number of possible DAGS
    """
    if n == 0 or n == 1:
        return 1
    return sum(((-1)**(k + 1)) * comb(n, k) * (2 ** (k * (n - k))) * _num_dags(n - k) for k in range(1, n + 1))

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

def _k2_node_score(var: str, parents: list[str], dataset: pd.DataFrame) -> float:
    """
    K2 score for a node in a BN given the dataset and the parents of the node

    :param var: The variable
    :param parents: The parents of that variable
    :param dataset: The dataset
    :return: The k2 log score
    """
    si = _arity(var)
    log_score = 0.0

    # Group the dataset by the parents of the variable.
    groups = dataset.groupby(parents) if parents else [(None, dataset)]
    for _, group in groups:
        # print(group)
        Sij = len(group) # Number of instances where parents take on a particular parent instantiation
        # gamma(x + 1) = x!
        log_score += gammaln(si) - gammaln(Sij + si)
        # Go through each value that the variable can take on
        for val in dataset[var].unique():
            # Count the number of observations that the variable in question takes on a particular value and
            # the variables parents take on a particular instantiation
            alpha_ijk = (group[var] == val).sum()
            log_score += gammaln(alpha_ijk + 1)
    return log_score

def k2_metric(model: dict, data: pd.DataFrame) -> float:
    """
    K2 score for an entire model given a dataset

    :param model: The BN model
    :param data: The dataset
    :return: The K2 log metric
    """
    adjmat = model["adjmat"]
    return log(1 / _num_dags(len(data.columns))) + sum(
        _k2_node_score(var, _get_parents(adjmat, var), dataset)
        for var in dataset.columns
    )

if __name__ == "__main__":
    # Import a pre-existing model
    model = bn.import_DAG('sachs')

    # Preprocessing for this particular dataset: Fix variable naming inconsistencies
    model["adjmat"].columns = model["adjmat"].columns.str.capitalize()
    model["adjmat"].index = model["adjmat"].index.str.capitalize()
    print(k2_metric(model, dataset))
