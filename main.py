import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings=None, confidence_threshold=0.6, fallback_response=None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []  # (tokenized_pattern, tag) pairs
        self.vocabulary = []  # all unique words from patterns
        self.intents = []  # list of tags
        self.intents_responses = {}  # tag -> responses

        self.function_mappings = function_mappings  # tag -> function
        self.confidence_threshold = confidence_threshold
        self.fallback_response = fallback_response or "Sorry, I didn't understand that. Can you rephrase?"

        self.X = None  # training inputs (bags)
        self.y = None  # training labels (intent indices)

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        # convert token list into 0/1 vector based on vocabulary
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        # read intents.json and build documents + vocabulary
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        # turn documents into X (bags) and y (intent indices)
        bags = []
        indices = []

        for pattern_words, tag in self.documents:
            bag = self.bag_of_words(pattern_words)
            intent_index = self.intents.index(tag)

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        # convert numpy arrays to torch tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        # save weights + input/output sizes
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        # rebuild model with correct sizes, then load weights
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

    def process_message(self, input_message):
        # handle empty input
        if not input_message or not input_message.strip():
            return self.fallback_response

        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(bag_tensor)

        # softmax gives probabilities for each intent
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

        confidence = confidence.item()
        predicted_intent = self.intents[predicted_class.item()]

        # if the model isn't confident, use fallback
        if confidence < self.confidence_threshold:
            return self.fallback_response

        # if intent has a function, run it and return its text
        if self.function_mappings and predicted_intent in self.function_mappings:
            result = self.function_mappings[predicted_intent]()
            if isinstance(result, str) and result.strip():
                return result

        # otherwise pick a random response for that intent
        if self.intents_responses.get(predicted_intent):
            return random.choice(self.intents_responses[predicted_intent])

        return self.fallback_response


def get_stocks():
    # example "dynamic" response (pretend portfolio)
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
    picks = random.sample(stocks, 3)
    return f"Here are your stocks: {', '.join(picks)}"


if __name__ == '__main__':
    # make sure NLTK data exists (so it doesn't crash)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    # load the assistant + trained model
    assistant = ChatbotAssistant(
        'intents.json',
        function_mappings={'stocks': get_stocks},
        confidence_threshold=0.6
    )
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    # simple chat loop
    while True:
        message = input('Enter your message:')

        if message == '/quit':
            break

        print(assistant.process_message(message))
