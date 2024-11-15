"""
﷽
author: @anbarsanti
"""

import torch
from torch.utils.data import Dataset, DataLoader

## ===================================================================================
## ================================ KALMAN FILTER ====================================
## ===================================================================================


## ===================================================================================
## ================== KLT (Kanade - Lucas - Tomasi) TRACKER  =========================
## ===================================================================================

## ===================================================================================
## ================================ DEEP SORT TRACKER ================================
## ===================================================================================


## ===================================================================================
## ======================================== LSTM ====================================
## ===================================================================================

# Prepare the data in a sequence format
class OBBSequenceDataset(Dataset):
    def __init__(self, obb_predictions, sequence_length):
        self.obb_predictions = obb_predictions
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.obb_predictions) - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence = self.obb_predictions[idx:idx+self.sequence_length]
        return torch.tensor(sequence, dtype=torch.float32)

# Define the LSTM model to process the OBB sequences
class OBBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(OBBLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Train the LSTM model
def train_lstm(model, train_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, seq in enumerate(train_loader):
            outputs = model(seq)
            loss = criterion(outputs, seq[:, -1, :])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def smooth_obb_predictions(model, obb_predictions, sequence_length):
    model.eval()
    smoothed_predictions = []

    with torch.no_grad():
        for i in range(len(obb_predictions) - sequence_length + 1):
            seq = torch.tensor(obb_predictions[i:i+sequence_length], dtype=torch.float32).unsqueeze(0)
            smoothed_obb = model(seq)
            smoothed_predictions.append(smoothed_obb.squeeze().tolist())

    return smoothed_predictions

# Assuming you have your OBB predictions in a list called 'obb_predictions'
dataset = OBBSequenceDataset(obb_predictions, sequence_length=10)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

input_size = 5  # Assuming each OBB has 5 parameters (x, y, w, h, angle)
hidden_size = 64
num_layers = 2
output_size = 5

model = OBBLSTM(input_size, hidden_size, num_layers, output_size)
train_lstm(model, train_loader, num_epochs=50, learning_rate=0.001)

# Use the LSTM model to smooth your OBB predictions:
smoothed_obbs = smooth_obb_predictions(model, obb_predictions, sequence_length=10)


