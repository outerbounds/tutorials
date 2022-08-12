# NLP With Metaflow

Welcome to the GitHub repo that supports the [NLP with Metaflow tutorial]()!  This repo contains supporting files so you can run the tutorial yourself and see the code.

## Setup

To run the tutorial, you need to install the required depdendencies via conda. We have included a conda environment in the form of a [env.yml](./env.ml) file for you to use.  You can install the environemnent via the following command:

> We are using `mamba` instead of `conda` because it is significantly faster. However you can use `conda` if you want to.

```bash
mamba env create -f env.yml
conda activate mf-tutorial-nlp
```

## Running the Tutorial

1. Lesson 1: Building The Model
    - [Tutorial page]()
    - [Notebook](nlp-1.ipynb)
2. Lesson 2: Creating A Baseline Flow
    - [Tutorial page]()
    - [baselineflow.py](baselineflow.py)
3. Lesson 3: Branching & Training The Model
   - [Tutorial page]()
   - [branchflow.py](branchflow.py)
4. Lesson 4: Tagging & Model Evaluation
   - [Tutorial page]()
   - [nlpflow.py](nlpflow.py)
5. Lesson 5: Using & Retrieving Your Model
   - [Tutorial page]()
   - [Notebook](nlp-5.ipynb)
   - [predflow.py](predflow.py)

