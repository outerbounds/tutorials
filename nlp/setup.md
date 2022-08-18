---
title: NLP Tutorial Setup
slug: /docs/nlp-tutorial-setup
tags: [orchestration]
sidebar_label: Setup
pagination_label: Setup
id: nlp-tutorial-setup
description: Setting up your environement for the NLP tutorial
category: data science
---
 

## Clone This Repository

```
git clone https://github.com/outerbounds/tutorials.git
```

## Install Requirements With Conda

To run the tutorial, you need to install the required depdendencies via conda. We have included a conda environment in the form of a [env.yml](./env.ml) file for you to use.  You can install the environemnent via the following command:

> We are using [`mamba`](https://mamba.readthedocs.io/en/latest/) instead of `conda` because it is significantly faster. However you can use `conda` if you want to.

```
cd tutorials/nlp
mamba env create -f env.yml
conda activate mf-tutorial-nlp
```

## Running The Tutorials

Please follow the instructions in each lesson for running either the associated python script or jupyter notebook.