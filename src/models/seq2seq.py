import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encodes the input sequence into a hidden state using an LSTM.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        :param x: (batch_size, x_seq_len, input_size)
        :return: outputs, (h, c)
                 outputs -> (batch_size, x_seq_len, hidden_size)
                 h, c -> (num_layers, batch_size, hidden_size)
        """
        outputs, (h, c) = self.lstm(x)
        return outputs, (h, c)


class Decoder(nn.Module):
    """
    Decodes from the Encoder's hidden state to produce a sequence, step by step.
    """

    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Decoder LSTM expects the previous output as input:
        self.lstm = nn.LSTM(
            output_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        :param x: (batch_size, 1, output_size) the previous step's prediction
                  (or the true value if teacher forcing)
        :param hidden: (h, c) from encoder or from previous decoder step
        :return: out -> (batch_size, 1, output_size), (h, c)
        """
        out, (h, c) = self.lstm(x, hidden)  # out: (batch_size, 1, hidden_size)
        pred = self.fc(out)  # pred: (batch_size, 1, output_size)
        return pred, (h, c)


class Seq2Seq(nn.Module):
    """
    Combines Encoder and Decoder. In a single forward pass, it:
      1) Feeds input sequence to the encoder to get final hidden state.
      2) Iterates over the target sequence in the decoder, optionally applying teacher forcing.
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, teacher_forcing_ratio: float = 0.5):
        """
        :param x: (batch_size, x_seq_len, input_size)
        :param y: (batch_size, y_seq_len, output_size)
        :param teacher_forcing_ratio: probability to feed the true previous y instead of model's prediction
        :return: predictions -> (batch_size, y_seq_len, output_size)
        """
        batch_size, y_len, output_dim = y.shape

        # Encode source sequence
        _, hidden = self.encoder(x)  # hidden = (h, c)

        # Prepare a tensor to store decoder outputs
        outputs = torch.zeros(batch_size, y_len, output_dim).to(x.device)

        # The decoder's first input is typically the first "target" step
        # or a special start token. We'll use y[:, 0, :] as a start.
        dec_input = y[:, 0].unsqueeze(1)  # shape: (batch_size, 1, output_dim)

        for t in range(1, y_len):
            # Forward through decoder one step
            pred, hidden = self.decoder(
                dec_input, hidden
            )  # pred: (batch_size, 1, output_dim)

            outputs[:, t] = pred.squeeze(1)

            # Decide if we use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                # Use the actual next target as input
                dec_input = y[:, t].unsqueeze(1)  # shape: (batch_size, 1, output_dim)
            else:
                # Use our own prediction as next input
                dec_input = pred  # shape: (batch_size, 1, output_dim)

        return outputs
