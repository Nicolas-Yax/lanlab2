# Lanlab
Simple to learn and to run LLM framework

## What is lanlab ?
Lanlab is a framework that aims at providing convenient and simple tools to help researchers build their own language model pipeline. As research in language model is developing rapidly, it is important to have a flexible framework that can be easily extended to support new models and new tasks. Lanlab is built around very simple and general concepts, and it is easy to extend it to support new models and new tasks.

The baseline code already includes many functionalities such as local LLM hosting and an optimized pipeline to batch inputs and queries to remote models. This makes it so that users don't have to worry about most of the technical stuff and can only focus on what matters in their research project.

## Install
Clone the repository
```git clone https://github.com/Nicolas-Yax/lanlab```

Install the required libraries
```pip install -r requirements.txt```

### Setup OPENAI

Put an OPENAI API key
```[key] >> .api_openai```

### Setup HuggingFace

Download the models in a folder and update the link to the folder in lanlab/core/module/models/hf_models;py -> HFMODELSPATH

## Setup Palm

To be implemented in later version.

## How to learn the framework
A tutorial notebook is available at the root of the respository. It can be done in less than 15 minutes and explains most of what is necessary to use the framework. In addition two notebooks crt.ipynb and logscores.ipynb are exemples of research projects lead by the HRL team using this framework (https://arxiv.org/ftp/arxiv/papers/2309/2309.12485.pdf).

## Problems with the framework ?
If you find bugs or aren't sure about something in the framework please open an issue. I'll try to answer your questions, add tutorials, solve bugs as soon as I can.

## Contributions
This framework is made to be as simple as possible to learn. Contributions are always welcome as long as they make things simpler for users while being very simple to learn to use.
