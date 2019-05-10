import torch.nn as nn
# https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948


class NLP_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, layers_num):
        super().__init__()

        self.layers_num = layers_num
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, layers_num, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)  # output size is 1
        self.act = nn.Sigmoid()

    def forward(self, x, hidden_state):
        batches = x.size(0)
        word_embeddings = self.word_embedding(x)
        out, hidden_out = self.lstm(word_embeddings, hidden_state)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.linear(out)
        out = self.act(out)
        out = out.view(batches, -1)
        out = out[:, -1]
        return out, hidden_out

    def make_hidden_states(self, batches):
        weight = next(self.parameters()).data
        new_states = (weight.new(self.layers_num, batches, self.hidden_size).zero_(),
                  (weight.new(self.layers_num, batches, self.hidden_size).zero_()))
        return new_states
