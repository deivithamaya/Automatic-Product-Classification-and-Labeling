# Final-Project Anyone-AI Automated-product categorization for e-commerce with AI

> Sentiment Analysis on Movies Reviews

## The Business problem


## About the data

In this project, we will work exclusively with two files: `products.json` and `categories.json`.


ion**.

## Technical aspects

To Start with our Extraction Data Analisys you will have to primary interact with the Jupyter notebook provided, called `EDA.ipynb`. This notebook will guide you through all the steps you have to follow the different parts of the project. This Notebok will create the datasets to feed the models later.

## Install

A `requirements.txt` file is provided with all the needed Python libraries for running this project. For installing the dependencies just run:

```console
$ pip install -r requirements.txt
```

*Note:* We encourage you to install those inside a virtual environment.

## Run Project

It doesn't matter if you are inside or outside a Docker container, in order to execute the project you need to launch a Jupyter notebook server running:

```bash
$ cd project
$ jupyter notebook
```

Then, inside the file `AnyoneAI - Sprint Project 05.ipynb`, you can see the project statement, description and also which parts of the code you must complete in order to solve it.

## Code Style

Following a style guide keeps the code's aesthetics clean and improves readability, making contributions and code reviews easier. Automated Python code formatters make sure your codebase stays in a consistent style without any manual work on your end. If adhering to a specific style of coding is important to you, employing an automated to do that job is the obvious thing to do. This avoids bike-shedding on nitpicks during code reviews, saving you an enormous amount of time overall.

We use [Black](https://black.readthedocs.io/) for automated code formatting in this project, you can run it with:

```console
$ black --line-length=88 .
```
