from torch import nn



class SiameseNetwork(nn.Module):

    def __init__(self, dim, ks, len_f, len_t):
        super(SiameseNetwork, self).__init__()
        self.dim_out = dim
        self.cnn_out_channel = 8
        self.len_f = len_f
        self.len_t = len_t
        self.kernel_size = ks

        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=self.kernel_size, padding=0),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, self.cnn_out_channel, kernel_size=self.kernel_size, padding=0),
            nn.BatchNorm1d(self.cnn_out_channel),
            nn.Sigmoid(), )

        self.fc1 = nn.Sequential(
            nn.Linear(self.cnn_out_channel * (self.len_f - 2 * (self.kernel_size - 1)), self.dim_out),)

        self.fc2 = nn.Sequential(
            nn.Linear(self.cnn_out_channel * (self.len_t - 2 * (self.kernel_size - 1)), self.dim_out),)

    def forward_f_encoder(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward_t_encoder(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc2(output)
        return output

    def forward(self, input1, input2):
        output_f = self.forward_f_encoder(input1)
        output_t = self.forward_t_encoder(input2)
        return output_f, output_t


class NETA(nn.Module):
    def __init__(self, dim, ks, len_f):
        super(NETA, self).__init__()
        self.kernel_size = ks
        self.fc1 = nn.Sequential(nn.Linear(8 * (len_f - 2 * (self.kernel_size - 1)), dim))
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=self.kernel_size, padding=0),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=self.kernel_size, padding=0),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h_a = self.cnn1(x)
        h_a = h_a.view(h_a.size()[0], -1)
        v_a = self.fc1(h_a)
        return v_a


class NETB(nn.Module):
    def __init__(self, dim, ks, len_t):
        super(NETB, self).__init__()
        self.kernel_size = ks
        self.fc2 = nn.Sequential(nn.Linear(8 * (len_t - 2 * (self.kernel_size - 1)), dim))
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=self.kernel_size, padding=0),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=self.kernel_size, padding=0),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h_b = self.cnn1(x)
        h_b = h_b.view(h_b.size()[0], -1)
        v_b = self.fc2(h_b)
        return v_b


