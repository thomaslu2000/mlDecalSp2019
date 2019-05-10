import json
import pickle
import os
from utils import *
from model import NLP_LSTM
import torch
from torch.utils.data import DataLoader, TensorDataset


unlabeled = json.load(open('testDataset.json', 'r'))

titles, texts = extract(unlabeled, False)

word_map = pickle.load(open("word_map.p", "rb"))

tokens = tokenize_text(texts)

int_texts = text_to_ints(tokens, word_map)

params = {"id": "3", "len_feat": 200, "batch size": 10, "epochs": 4, "embedding size": 300, "hidden size": 256, "layers": 2, "lr": 0.002}

features = process_data(int_texts, params["len_feat"])
dummy_answers = np.zeros(shape=[len(features)])
batch_size=params["batch size"]
data_test = DataLoader(TensorDataset(torch.from_numpy(features), torch.from_numpy(dummy_answers)), shuffle=True, batch_size=batch_size)

vocab_size = len(word_map) + 1  # +1 for the 0 padding
embedding_size = params["embedding size"]
hidden_size = params["hidden size"]
layers = params["layers"]
save_path = "save-" + params["id"]
model = NLP_LSTM(vocab_size, embedding_size, hidden_size, layers)

model.load_state_dict(torch.load(save_path))
model.eval()

test_losses = []  # track loss
num_correct = 0

# init hidden state
hidden_state = model.make_hidden_states(batch_size)

model.eval()
# iterate over test data
i = 0
for in_x, out_y in data_test:
    i += 1
    print(i)
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    hidden_state = tuple([each.data for each in hidden_state])

    # get predicted outputs
    in_x = in_x.type(torch.LongTensor)
    if in_x.shape[0] == batch_size:
        output, hidden_state = model(in_x, hidden_state)

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(out_y.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

# accuracy over all test data
test_acc = num_correct / len(data_test.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
print()
