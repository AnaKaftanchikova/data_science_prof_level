import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from nltk_utils import tokenize, lemmatize, bag_of_words
from model import NeuralNet
import random

# Логирование в файл
logging.basicConfig(level=logging.INFO,
                    filename="logs/train.log",
                    filemode="a",
                    format="%(asctime)s %(levelname)s: %(message)s")

with open('data/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend([lemmatize(tok) for tok in w])
        xy.append((w, tag))

# уникальные слова
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# создаём датасет
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# параметры
batch_size = 32
hidden_size = 16
input_size = len(all_words)
output_size = len(tags)
learning_rate = 0.001
dropout = 0.3
num_epochs = 300

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

logging.info("Start training. input_size=%d, output_size=%d, epochs=%d", input_size, output_size, num_epochs)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device).float()
        labels = labels.to(device).long()

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], loss={loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], loss={loss.item():.4f}")

# сохранить
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "model/data.pth"
torch.save(data, FILE)
logging.info("Training complete. Model saved to %s", FILE)
print("Training complete. File saved to", FILE)
