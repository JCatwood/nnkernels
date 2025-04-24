import torch

class LSTMKernelSD(torch.nn.Module):
    def __init__(self, locsDim, nHidden):
        super(LSTMKernelSD, self).__init__()
        self.n_hidden = nHidden
        self.n_feature = locsDim
        self.lstm = torch.nn.LSTM(locsDim, nHidden, batch_first=True, bidirectional=True)
        self.hidden2pred = torch.nn.Linear(nHidden * 2, 1)

    def forward(self, locs, h0=None, c0=None):
        if h0 is None or c0 is None:
            lstm_out, (h_out, c_out) = self.lstm(locs)
        else:
            lstm_out, (h_out, c_out) = self.lstm(locs, (h0, c0))
        out = torch.exp(self.hidden2pred(lstm_out[:, -1, :])).reshape((-1,))
        return out, h_out, c_out

class LSTMKernelMean(torch.nn.Module):
    def __init__(self, locsNyDim, nHidden):
        super(LSTMKernelMean, self).__init__()
        self.n_hidden = nHidden
        self.n_feature = locsNyDim
        self.lstm = torch.nn.LSTM(locsNyDim, nHidden, batch_first=True, bidirectional=True)
        self.hidden2pred = torch.nn.Linear(nHidden * 2, 1)

    def forward(self, locsNy, h0=None, c0=None):
        if h0 is None or c0 is None:
            lstm_out, (h_out, c_out) = self.lstm(locsNy)
        else:
            lstm_out, (h_out, c_out) = self.lstm(locsNy, (h0, c0))
        out = self.hidden2pred(lstm_out[:, -1, :]).reshape((-1,))
        return out, h_out, c_out

