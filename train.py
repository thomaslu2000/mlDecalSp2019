import json
import pickle
import os
from utils import *
from model import NLP_LSTM
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset

labeled = json.load(open('trainingDataset.json', 'r'))

ratings, titles, texts = extract(labeled, True)

if os.path.isfile("word_map.p"):
    word_map = pickle.load(open("word_map.p", "rb"))
else:
    print("making new word map")
    word_map = make_word_to_int_mapping(texts + titles, 2500)
    pickle.dump(word_map, open("word_map.p", "wb"))

tokens = tokenize_text(texts)


int_texts = text_to_ints(tokens, word_map)
# loop for diff hyper-parameter
hyperparams = [
    {"id": "1", "len_feat": 200, "batch size": 25, "epochs": 5, "embedding size": 400, "hidden size": 256, "layers": 2, "lr": 0.002},
    {"id": "2", "len_feat": 200, "batch size": 25, "epochs": 4, "embedding size": 500, "hidden size": 300, "layers": 3, "lr": 0.005},
    {"id": "3", "len_feat": 200, "batch size": 10, "epochs": 4, "embedding size": 300, "hidden size": 256, "layers": 2, "lr": 0.002},
    {"id": "4", "len_feat": 200, "batch size": 10, "epochs": 5, "embedding size": 300, "hidden size": 256, "layers": 2, "lr": 0.001}
]

test_accuracies = []

positive_texts = []
negative_texts = []
positive_labels = []
negative_labels = []
for i in range(len(ratings)):
    if ratings[i] == 1:
        positive_labels.append(ratings[i])
        positive_texts.append(int_texts[i])
    else:
        negative_labels.append(ratings[i])
        negative_texts.append(int_texts[i])

# k fold validation
for params in hyperparams:
    len_feat = params["len_feat"]
    good_features = process_data(positive_texts, len_feat)
    bad_features = process_data(negative_texts, len_feat)
    # features = process_data(int_texts, len_feat)

    validation_chunks = 10

    batch_size = params["batch size"]

    # data_pieces = split_array(features, validation_chunks)
    # label_pieces = split_array(ratings, validation_chunks)
    good_data_pieces = split_array(good_features, validation_chunks)
    good_label_pieces = split_array(positive_labels, validation_chunks)
    bad_data_pieces = split_array(bad_features, validation_chunks)
    bad_label_pieces = split_array(negative_labels, validation_chunks)
    test_idx = random.randint(0, validation_chunks)
    input_test = np.vstack([good_data_pieces.pop(test_idx), bad_data_pieces.pop(test_idx)])
    output_test = np.hstack([good_label_pieces.pop(test_idx), bad_label_pieces.pop(test_idx)])
    data_test = DataLoader(TensorDataset(torch.from_numpy(input_test), torch.from_numpy(output_test)), shuffle=True, batch_size=batch_size)

    vocab_size = len(word_map) + 1  # +1 for the 0 padding
    embedding_size = params["embedding size"]
    hidden_size = params["hidden size"]
    layers = params["layers"]
    save_path = "save-" + params["id"]
    model = NLP_LSTM(vocab_size, embedding_size, hidden_size, layers)
    criterion = nn.BCELoss()

    if os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path))
        model.eval()
    else:

        from matplotlib import pyplot as plt

        lr = params["lr"]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        epochs = params["epochs"]

        iteration_num = 0
        data_step = 10
        clip = 5

        training_losses = []
        validation_losses = []
        validation_accuracies = []

        for e in range(epochs):
            data_size = min(len(good_data_pieces), len(bad_data_pieces))
            validation_idx = random.randint(0, data_size - 1)
            good_train_x , good_valid_x = select_one(good_data_pieces, validation_idx)
            bad_train_x , bad_valid_x = select_one(bad_data_pieces, validation_idx)
            good_y, good_valid_y = select_one(good_label_pieces, validation_idx)
            bad_y, bad_valid_y = select_one(bad_label_pieces, validation_idx)
            # print(good_y.shape)
            # print(bad_y.shape)

            data_size = min(len(good_train_x), len(bad_train_x))
            valid_size = min(len(good_valid_x), len(bad_valid_x))
            # make sure to SHUFFLE your data
            data_train = DataLoader(TensorDataset(torch.from_numpy(np.vstack([good_train_x[:data_size], bad_train_x[:data_size]])),
                                                  torch.from_numpy(np.concatenate([good_y[:data_size], bad_y[:data_size]]))), shuffle=True, batch_size=batch_size)
            data_validation = DataLoader(TensorDataset(torch.from_numpy(np.vstack([good_valid_x[:valid_size], bad_valid_x[:valid_size]])),
                                                       torch.from_numpy(np.concatenate([good_valid_y[:valid_size], bad_valid_y[:valid_size]]))), shuffle=True, batch_size=batch_size)

            # initialize hidden state
            hidden_state = model.make_hidden_states(batch_size)

            # batch loop
            for in_x, out_y in data_train:
                iteration_num += 1

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                hidden_state = tuple([each.data for each in hidden_state])

                # zero accumulated gradients
                model.zero_grad()

                # get the output from the model
                in_x = in_x.type(torch.LongTensor)
                if in_x.shape[0] == batch_size:
                    output, hidden_state = model(in_x, hidden_state)

                    loss = criterion(output.squeeze(), out_y.float())
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

                    if iteration_num % data_step == 0:
                        val_h = model.make_hidden_states(batch_size)
                        val_losses = []
                        model.eval()
                        num_correct = 0
                        for in_x, out_y in data_validation:
                            # Creating new variables for the hidden state, otherwise
                            # we'd backprop through the entire training history
                            val_h = tuple([each.data for each in val_h])

                            in_x = in_x.type(torch.LongTensor)
                            if in_x.shape[0] == batch_size:
                                output, val_h = model(in_x, val_h)
                                val_loss = criterion(output.squeeze(), out_y.float())

                                val_losses.append(val_loss.item())

                                # convert output probabilities to predicted class (0 or 1)
                                pred = torch.round(output.squeeze())  # rounds to the nearest integer

                                # compare predictions to true label
                                correct_tensor = pred.eq(out_y.float().view_as(pred))
                                correct = np.squeeze(correct_tensor.numpy())
                                num_correct += np.sum(correct)

                        model.train()
                        training_losses.append(loss.item())
                        validation_losses.append(np.mean(val_losses))
                        validation_accuracies.append(num_correct / len(data_validation.dataset))

                        print("id: {} | ".format(params["id"]),
                              "Epoch: {}/{} | ".format(e+1, epochs),
                              "Step: {} | ".format(iteration_num),
                              "Loss: {:.6f} | ".format(loss.item()),
                              "Val Loss: {:.6f} | ".format(np.mean(val_losses)),
                              "Accuracy: {:.3f}".format(num_correct / len(data_validation.dataset))
                              )

        torch.save(model.state_dict(), save_path)
        plt.plot(training_losses)
        plt.plot(validation_losses)
        plt.plot(validation_accuracies)
        plt.legend(["Training Losses", "Validation Losses", "Accuracy"])
        plt.savefig("" + params["id"] + '-training.png')
        plt.clf()

    test_losses = []  # track loss
    num_correct = 0

    # init hidden state
    hidden_state = model.make_hidden_states(batch_size)

    model.eval()
    # iterate over test data
    i = 0
    for in_x, out_y in data_test:
        i += 1
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        hidden_state = tuple([each.data for each in hidden_state])

        # get predicted outputs
        in_x = in_x.type(torch.LongTensor)
        if in_x.shape[0] == batch_size:
            output, hidden_state = model(in_x, hidden_state)

            # calculate loss
            test_loss = criterion(output.squeeze(), out_y.float())
            test_losses.append(test_loss.item())

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer

            # compare predictions to true label
            correct_tensor = pred.eq(out_y.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy())
            num_correct += np.sum(correct)

    # accuracy over all test data
    test_acc = num_correct / len(data_test.dataset)
    test_accuracies.append(test_acc)
    print("Test accuracy: {:.3f}".format(test_acc))
    print()

print(test_accuracies)
