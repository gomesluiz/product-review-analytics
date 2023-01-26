Product Review Analytics
==============================

A pos-doctorate research project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


Papers
===================
* [Natural language processing: state of the art, current trends and challenges](https://link.springer.com/content/pdf/10.1007/s11042-022-13428-4.pdf)
* [Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks](https://arxiv.org/abs/1708.06025)
* [The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)

Text augmentation
===================
* [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)
* [Adaptive Ranking-based Sample Selection for Weakly Supervised Class-imbalanced Text Classification](https://paperswithcode.com/paper/ars2-adaptive-ranking-based-sample-selection)

Datasets
===================
* Datasets em português - AiLab | UNB - [Link](https://forum.ailab.unb.br/t/datasets-em-portugues/25 ) 

How-to
===================
* [Can we take of benefit of using transfer learning while training a word2vec models?](https://datascience.stackexchange.com/questions/10642/can-we-take-of-benefit-of-using-transfer-learning-while-training-a-word2vec-mode)
* [Gensim – Vectorizing Text and Transformations](https://dzone.com/articles/gensim-vectorizing-text-and-transformations)
* [Text Vectorization and Transformation Pipelines](https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html)
  
--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
