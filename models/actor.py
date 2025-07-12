import torch
import torch.nn as nn

class ActorLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=48, num_actions=2, use_layernorm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.init_forget_bias(self.lstm, bias_val=0.5)

        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_size)

        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),                         
            nn.Linear(hidden_size, num_actions)
        )

        self._init_weights()

    def init_forget_bias(self, lstm, bias_val=0.5):
        for name, param in lstm.named_parameters():
            if 'bias_ih' in name:
                hidden_size = param.shape[0] // 4
                forget_gate = slice(hidden_size, 2 * hidden_size)
                with torch.no_grad():
                    param[forget_gate].fill_(bias_val)

    def _init_weights(self):
        for layer in self.policy:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, hx=None, clamp_c_val=10.0):
        output, (h, c) = self.lstm(x, hx)
        c = torch.clamp(c, min=-clamp_c_val, max=clamp_c_val)
        last_output = output[:, -1, :]
        if self.use_layernorm:
            last_output = self.ln(last_output)
        logits = self.policy(last_output)
        return logits, (h, c), output