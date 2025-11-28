import torch
import json
from model.model import NeuralNet
from model.nltk_utils import tokenize, bag_of_words
import logging
import random

logging.basicConfig(level=logging.INFO, filename="logs/chat.log",
                    format="%(asctime)s %(message)s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

data = torch.load("model/data.pth", map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(sentence):
    sentence = sentence.lower().strip()
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).to(device).float().unsqueeze(0)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.6:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return tag, response
    else:
        return "fallback", "Извини, не уверен — можешь переформулировать?"

if __name__ == "__main__":
    print("Запущен локальный чат. Вводи 'exit' для выхода.")
    while True:
        sentence = input("Ты: ")
        if sentence.lower() in ["exit", "quit", "выход"]:
            break
        tag, resp = get_response(sentence)
        print("Бот:", resp)
        logging.info(f"User: {sentence}")
        logging.info(f"Bot (tag={tag}): {resp}")
