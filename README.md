![Header](img/header.png)

# Summary
 - [Project](#project)
   - [Description](#description)
   - [Related publications](#related-publications)
 - [Example](#example)
 - [Installation](#installation)
 - [Versions and tags](#versions-and-tags)
 - [Citation](#citation)


# Project

This is an open source implementation of the Logical Analysis of Data (LAD) Algorithm.

### Description

Logical Analysis of Data (LAD) is a rule-based machine learning algorithm based on ideas from Optimization and Boolean Function Theory. The LAD methodology was originally conceived by Peter L. Hammer, from Rutgers University, and has been described and developed in a number of papers since the late 80's. It has also been applied to classification problems arising in areas such as Medicine, Economics, and Bioinformatics. A list with representative publications about LAD will be made available here shortly.

LAD algorithm consists of dectecting hidden patterns capable of distinguishing observations in one class from all the other observations. The patterns are human readable which are used for reasoning of the decisions made by the classifier. 

### Related publications
 - **Maximum Patterns in Datasets**. Bonates, T.O., P.L. Hammer, A. Kogan. Discrete Applied Mathematics, vol. 156(6), 846-861, 2008. [Link](https://www.sciencedirect.com/science/article/pii/S0166218X07002089)
 
 - **An Implementation of Logical Analysis of Data**. Boros, E., P.L. Hammer, T. Ibaraki, A. Kogan, E. Mayoraz, I. Muchnik. IEEE Transactions on Knowledge and Data Engineering, vol 12(2), 292-306, 2000. [Link](https://ieeexplore.ieee.org/abstract/document/842268?casa_token=y2NyWCbn7SsAAAAA:LCrKLdntpx-5GRNVdtU4F-Cnfs4VqsfWZTspa_yvgy_acfHvZjoZt_ZXKtHAOdiZGioUiNAN6m4FwQ)

 - **Classificação Supervisionada de Dados via Otimização e Funções Booleanas**. Gomes, V.S.D., T. O. Bonates. Anais do II Workshop Técnico-Científico de Computação, p.21-27, Mossoró, RN, Brazil, 2011.

# Example
As the code was implemented following sklean's classifiers documentation, its usage is quite straightforward. See the code below.

```py
from lad.lad import LADClassifier

from sklearn import datasets
from sklearn.model_selection import cross_val_score

# Dataset
X, y = datasets.load_iris(return_X_y=True)

# Classifier
lad = LADClassifier()

# CV
scores = cross_val_score(lad, X, y, cv=10, scoring="accuracy")
```

> The current version of lad doesn't implement a score function!

Please, refer to the [examples.py](examples.py) file for another example.

# Installation

Choose one of the following in order to install this classifier.

1. Install it with pip:

    ```sh
    $ python -m pip install git+https://github.com/Bha-Gu/lad.git#egg=lad
    ```

2. Clone this repository and use the setup file to install:

    ```sh
    $ git clone https://github.com/Bha-Gu/lad.git
    ```
    ```sh
    $ sudo python setup.py install
    ```

# Citation

In case you want to cite this project:

```bibtex
@software{Bha Gu,
    author = {Bha Gu},
    month = {7},
    title = {{Logical Analysis of Data a Python Implementation}},
    version = {0.9.0},
    year = {2024}
}
```
