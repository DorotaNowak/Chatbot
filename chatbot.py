import random
import json

import torch

from net import Net
from utils import set_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "saved_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = Net(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def chatbot_response(msg):
    msg = tokenize(msg)
    X = set_of_words(msg, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    print(probs)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.8:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "Nie rozumiem pytania. Jeśli potrzebujesz " \
               "pomocy możesz skontaktować się z nami pod numerem " \
               "telefonu 513-445-988 codziennie od 8:00 do 16:00."
