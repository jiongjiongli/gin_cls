from types import SimpleNamespace
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool


import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.optim as optim



def make_convolution(in_channels, out_channels):
    return GINConv(nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    ))

class GINClassification(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GINClassification, self).__init__()
        self.conv1 = make_convolution(in_channels, hidden_channels)
        self.conv2 = make_convolution(hidden_channels, hidden_channels)
        self.conv3 = make_convolution(hidden_channels, out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        return self.classifier(x)

    def extract_embedding(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        return x



def train_model(model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    num_epochs,
    model_dir_path):
    best_model_path = model_dir_path / 'best_model.pth'
    best_test_accuracy = 0.0
    best_f1 = 0.0

    log_data = {'Epoch': [], 'Average Loss': [], 'Test Accuracy': [], 'F1 Score': []}

    for epoch in tqdm(range(num_epochs), desc='Training'):
        total_loss = 0
        total_samples = 0

        # Training loop
        model.train()
        for data in train_loader:
            x, edge_index, y = data.x, data.edge_index, data.y
            optimizer.zero_grad()
            out = model(x, edge_index, data.batch)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs

        avg_loss = total_loss / total_samples

        # Test loop
        model.eval()
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for data in test_loader:
                x, edge_index, y = data.x, data.edge_index, data.y
                out = model(x, edge_index, data.batch)
                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                true_labels.extend(y.numpy())
                predicted_labels.extend(predicted.numpy())

        test_accuracy = correct / total
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.5f}, Test Accuracy: {test_accuracy:.5f}, F1 Score: {f1:.5f}')

        # Log data
        log_data['Epoch'].append(epoch + 1)
        log_data['Average Loss'].append(avg_loss)
        log_data['Test Accuracy'].append(test_accuracy)
        log_data['F1 Score'].append(f1)

        # Save the best model based on test accuracy and F1-score
        if test_accuracy > best_test_accuracy or (test_accuracy == best_test_accuracy and best_f1 < f1):
            tqdm.write("$$$ best model is updated according to accuracy! $$$")
            best_test_accuracy = test_accuracy
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)

    # Load the best model for evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Convert log_data to DataFrame
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(model_dir_path / 'train_log.csv', index=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config_dict = dict(
    root_dir_path = Path(r"/content"),
    dataset_name = "MUTAG",
    batch_size = 64,
    num_epochs = 20,
    device = device,

    val_size=0.1,
    test_size=0.2,
    seed=17,

    lr = 0.001,
    hidden_channels=1000,
    out_channels=100,
)

config = SimpleNamespace(**model_config_dict)

dataset_root_path = config.root_dir_path / "datasets" / config.dataset_name
model_dir_path = config.root_dir_path / "models" / config.dataset_name

dataset_root_path.mkdir(parents=True, exist_ok=True)
model_dir_path.mkdir(parents=True, exist_ok=True)

dataset = TUDataset(root=dataset_root_path,
    name=config.dataset_name,
    use_node_attr=True)

# Split dataset into training and test sets
trainval_dataset, test_dataset = train_test_split(dataset,
    test_size=config.test_size,
    random_state=config.seed)

train_dataset, val_dataset = train_test_split(trainval_dataset,
    test_size=config.val_size / (1 - config.test_size),
    random_state=config.seed)

# # Draw graphs from the test dataset before training
# draw_random_graph_samples(test_dataset, dataset_name, dataset.num_classes)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)

model = GINClassification(in_channels=dataset.num_node_features,
    hidden_channels=config.hidden_channels,
    out_channels=config.out_channels,
    num_classes=dataset.num_classes)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

train_model(model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    num_epochs=config.num_epochs,
    model_dir_path=model_dir_path)
