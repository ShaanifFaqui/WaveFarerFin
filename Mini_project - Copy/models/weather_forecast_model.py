import torch
import torch.nn as nn

class WeatherForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(WeatherForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.view(-1, 72, 3)  # Adjust output shape to your needs (72 hours * 3 features)