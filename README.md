# AI-CHATBOT
Intent-Based Chatbot (PyTorch + NLTK)

This is a simple intent chatbot I built using PyTorch and NLTK. It reads training data from intents.json, learns to classify intents, then replies with a response from the matching intent. It also supports a function mapping example (like the stocks intent).

Files in this repo

main.py
The main code for training and running the chatbot

intents.json
The intents dataset (tags, patterns, responses)

- Requirements

Python 3

torch

nltk

numpy

Install:
pip install torch nltk numpy

- NLTK setup

If you get an NLTK error the first time you run it, you’ll need these downloads:

punkt

wordnet

main.py usually downloads them automatically if they’re missing.

- How to train

In main.py, uncomment the training part (the section with train_model and save_model), then run:
python main.py

- How to run the chatbot

After you train and save the model files, keep the inference section uncommented and run:
python main.py

Type /quit to exit.

- Customize the chatbot

Edit intents.json to add new tags, patterns, or responses. After any edits, retrain the model so it learns the new patterns.

Notes

This is a beginner-friendly chatbot (bag-of-words + a small neural network), mainly for learning how intent classification works.
