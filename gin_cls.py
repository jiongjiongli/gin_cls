from types import SimpleNamespace
import os
from pathlib import Path
import random
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re

import networkx as nx

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix)

import torch.nn as nn
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool


def create_dataset(config):
    dataset_root_path = config.root_dir_path / "datasets" / config.dataset_name
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    dataset = TUDataset(root=dataset_root_path,
        name=config.dataset_name,
        use_node_attr=True)

    return dataset

def create_dataloaders(dataset, config):
    trainval_dataset, test_dataset = train_test_split(dataset,
        test_size=config.test_size,
        random_state=config.seed)

    train_dataset, val_dataset = train_test_split(trainval_dataset,
        test_size=config.val_size / (1 - config.test_size),
        random_state=config.seed)

    # # Draw graphs from the test dataset before training
    # draw_random_graph_samples(test_dataset, dataset_name, dataset.num_classes)

    train_dataloader = DataLoader(train_dataset,
        batch_size=config.batch_size,
        shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    dataloaders = dict(
        train=train_dataloader,
        val=val_dataloader,
        test=test_dataloader,
        )

    return dataloaders


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
        self.head = nn.Linear(out_channels, num_classes)

        # self.res1 = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()
        # self.res2 = nn.Identity()  # hidden_channels -> hidden_channels, same dim
        # self.res3 = nn.Linear(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()
        # self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # out1 = self.conv1(x, edge_index)
        # x = out1 + self.res1(x)
        # x = self.relu(x)
        x = self.conv1(x, edge_index)

        # out2 = self.conv2(x, edge_index)
        # x = out2 + self.res2(x)
        # x = self.relu(x)
        x = self.conv2(x, edge_index)

        # out3 = self.conv3(x, edge_index)
        # x = out3 + self.res3(x)
        # x = self.relu(x)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch=batch)
        return self.head(x)

    def extract_embedding(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        return x


class Trainer:
    def __init__(self, config, config_idx, fold, dataloaders, model):
        self.config = config
        self.config_idx = config_idx
        self.fold = fold
        self.model = model
        self.dataloaders = dataloaders

        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        config = self.config

        model_dir_path = config.root_dir_path / "models" / config.dataset_name
        model_dir_path.mkdir(parents=True, exist_ok=True)
        best_model_path = model_dir_path / f"config_{self.config_idx + 1:03d}_fold_{self.fold + 1:03d}_best_model.pth"
        eval_metrics_file_path = model_dir_path / f"config_{self.config_idx + 1:03d}_fold_{self.fold + 1:03d}_eval_metrics.csv"

        best_epoch = 0
        best_val_acc = 0.0
        best_test_acc = 0.0
        all_eval_metrics = []

        pbar = tqdm(range(config.num_epochs))

        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch + 1:03d}")
            train_loss = self.train_one_epoch(epoch, pbar)
            val_metrics  = self.test_one_epoch("val",  epoch, pbar)
            test_metrics = self.test_one_epoch("test", epoch, pbar)

            eval_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["avg_loss"],
                "test_loss": test_metrics["avg_loss"],

                "val_acc": val_metrics["accuracy"],
                "test_acc": test_metrics["accuracy"],

                "val_precision": val_metrics["precision"],
                "test_precision": test_metrics["precision"],

                "val_recall": val_metrics["recall"],
                "test_recall": test_metrics["recall"],

                "val_f1": val_metrics["f1"],
                "test_f1": test_metrics["f1"],
            }

            all_eval_metrics.append(eval_metrics)

            val_loss = val_metrics["avg_loss"]
            test_loss = test_metrics["avg_loss"]
            val_acc = val_metrics["accuracy"]
            test_acc = test_metrics["accuracy"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

            pbar.set_postfix({
                "train_loss": f"{train_loss:.2f}",
                "val_loss":   f"{val_loss:.2f}",
                "test_loss":  f"{test_loss:.2f}",
                "val_acc":    f"{val_acc:.2%}",
                "test_acc":   f"{test_acc:.2%}",
                "best_epoch": f"{best_epoch + 1:03d}",
                "best_val_acc": f"{best_val_acc:.2%}",
                "best_test_acc": f"{best_test_acc:.2%}",
                })

        eval_metrics_df = pd.DataFrame(all_eval_metrics)
        eval_metrics_df.to_csv(eval_metrics_file_path, index=False)

        # self.run_eval(best_model_path,
        #             best_epoch,
        #             best_val_acc,
        #             best_test_acc)

        result = {
            "config_idx": self.config_idx,
            "fold": self.fold,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc
        }

        return result

    def run_eval(self,
                model_path,
                best_epoch,
                best_val_acc,
                best_test_acc,
                pbar=None):
        model.load_state_dict(torch.load(model_path))
        val_metrics  = self.test_one_epoch("val",  best_epoch, pbar)
        test_metrics = self.test_one_epoch("test", best_epoch, pbar)

        val_loss  = val_metrics["avg_loss"]
        test_loss = test_metrics["avg_loss"]
        val_acc   = val_metrics["accuracy"]
        test_acc  = test_metrics["accuracy"]
        val_cm    = val_metrics["cm"]
        test_cm   = test_metrics["cm"]

        print("\n")
        print("=" * 80)
        print("Best Result:")
        print("-" * 80)
        print({
            "best_epoch": f"{best_epoch + 1:03d}",
            "val_loss":   f"{val_loss:.2f}",
            "test_loss":  f"{test_loss:.2f}",
            "val_acc":    f"{val_acc:.2%}",
            "test_acc":   f"{test_acc:.2%}",
            "best_val_acc": f"{best_val_acc:.2%}",
            "best_test_acc": f"{best_test_acc:.2%}",
            "val_cm": val_cm,
            "test_cm": test_cm,
            })
        print("=" * 80)

    def train_one_epoch(self, epoch, pbar):
        config = self.config
        model = self.model
        train_dataloader = self.dataloaders["train"]
        device = config.device
        optimizer = self.optimizer
        criterion = self.criterion

        model = model.to(device)
        model.train()

        running_loss = 0.0
        num_samples = 0

        for step, data in enumerate(train_dataloader):
            x, edge_index, y = data.x, data.edge_index, data.y
            optimizer.zero_grad()
            out = model(x.to(device),
                        edge_index.to(device),
                        data.batch.to(device))
            loss = criterion(out.to(device), y.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.num_graphs
            num_samples += data.num_graphs

        avg_loss = running_loss / num_samples

        return avg_loss

    @torch.no_grad()
    def test_one_epoch(self, split, epoch, pbar):
        config = self.config
        model = self.model
        dataloader = self.dataloaders[split]
        device = config.device
        criterion = self.criterion

        model = model.to(device)
        model.eval()

        running_loss = 0.0
        num_samples = 0
        all_preds = []
        all_labels = []

        for step, data in enumerate(dataloader):
            x, edge_index, y = data.x, data.edge_index, data.y

            out = model(x.to(device),
                        edge_index.to(device),
                        data.batch.to(device))
            loss = criterion(out.to(device), y.to(device))
            scores, preds = torch.max(out, 1)

            running_loss += loss.item() * data.num_graphs
            num_samples += data.num_graphs

            preds = preds.cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_loss = running_loss / num_samples

        accuracy  = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall    = recall_score(all_labels, all_preds, zero_division=0)
        f1        = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        eval_metrics = dict(
            epoch=epoch,
            avg_loss=avg_loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cm=cm,
            )

        return eval_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_config_dict = dict(
    root_dir_path = Path(r"/content"),
    dataset_name = "MUTAG",
    # batch_size
    num_epochs = 40,
    device = device,

    val_size=0.1,
    test_size=0.2,
    seed=17,

    # lr
    # hidden_channels
    # out_channels
    num_folds = 10,
)

batch_sizes = [32, 64]
learning_rates = [0.001, 0.01]
hidden_channels_list = [500, 1000,]
out_channels_list = [100, 200,]

config_dicts = []

for batch_size, learning_rate, hidden_channels, out_channels in product(
    batch_sizes, learning_rates, hidden_channels_list, out_channels_list):

    config_dict = dict(
        **common_config_dict,

        batch_size = batch_size,

        lr = learning_rate,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    config_dicts.append(config_dict)

common_config = SimpleNamespace(**common_config_dict)
dataset = create_dataset(common_config)
dataset_indices = np.arange(len(dataset))
trainval_idx, test_idx = train_test_split(dataset_indices,
            test_size=common_config.test_size,
            random_state=common_config.seed)

all_results = []


for config_idx, input_config_dict in enumerate(config_dicts):
    print(f"Config: {config_idx + 1} / {len(config_dicts)}")
    config = SimpleNamespace(**input_config_dict)

    best_results = []

    kf = KFold(n_splits=config.num_folds,
        shuffle=True,
        random_state=config.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_idx)):
        print(f"Fold {fold + 1}/{config.num_folds}")

        train_dataset = dataset[train_idx]
        val_dataset   = dataset[val_idx]
        test_dataset  = dataset[test_idx]

        dataset_info = dict(
            train=len(train_dataset),
            val=len(val_dataset),
            test=len(test_dataset)
        )

        print(f"dataset_info: {dataset_info}")

        train_dataloader = DataLoader(train_dataset,
            batch_size=config.batch_size,
            shuffle=True)
        val_dataloader = DataLoader(val_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset, shuffle=False)

        dataloaders = dict(
            train=train_dataloader,
            val=val_dataloader,
            test=test_dataloader,
            )

        # dataloaders = create_dataloaders(dataset, config)

        model = GINClassification(in_channels=dataset.num_node_features,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            num_classes=dataset.num_classes)

        trainer = Trainer(config, config_idx, fold, dataloaders, model)

        best_result = trainer.train()
        best_results.append(best_result)

    best_results_df = pd.DataFrame(best_results)
    mean_val_acc = best_results_df["best_val_acc"].mean()
    std_val_acc  = best_results_df["best_val_acc"].std()
    mean_test_acc = best_results_df["best_test_acc"].mean()
    std_test_acc  = best_results_df["best_test_acc"].std()
    print("\n")
    print(f"config: {config}")
    print(f"{config.num_folds}-Fold CV Validation Acc: {mean_val_acc:.2%} ± {std_val_acc:.2%} Test Acc: {mean_test_acc:.2%} ± {std_test_acc:.2%}")

    result = {"config": config,
        "mean_val_acc": mean_val_acc,
        "std_val_acc": std_val_acc,
        "mean_test_acc": mean_test_acc,
        "std_test_acc": std_test_acc
    }

    all_results.append(result)

all_results_df = pd.DataFrame(all_results)





def show_best_result(all_results_df):
    result_idx = all_results_df["mean_val_acc"].idxmax()
    best_result = all_results_df.loc[result_idx]
    display(best_result)

    config = best_result["config"]
    mean_val_acc = best_result["mean_val_acc"]
    std_val_acc = best_result["std_val_acc"]
    mean_test_acc = best_result["mean_test_acc"]
    std_test_acc = best_result["std_test_acc"]

    print(result_idx)
    display(config)
    print(f"{config.num_folds}-Fold CV Validation Acc: {mean_val_acc:.2%} ± {std_val_acc:.2%} Test Acc: {mean_test_acc:.2%} ± {std_test_acc:.2%}")
    return result_idx



def show_metrics(eval_metrics_file_path):
    match_result = re.match(r"config_([0-9]+)_fold_([0-9]+)_eval_metrics", eval_metrics_file_path.stem)

    assert match_result, eval_metrics_file_path.stem
    fold = int(match_result.group(2))

    eval_metrics = pd.read_csv(eval_metrics_file_path)

    sns.set(style="whitegrid")

    # Find the epoch with the highest validation accuracy
    best_idx = eval_metrics['val_acc'].idxmax()
    best_epoch = eval_metrics['epoch'][best_idx]
    best_val_acc = eval_metrics['val_acc'][best_idx]
    best_test_acc = eval_metrics['test_acc'][best_idx]

    # Plot Accuracy with highlight
    plt.figure(figsize=(10, 8))
    plt.plot(eval_metrics['epoch'], eval_metrics['val_acc'], label='Validation Accuracy', marker='o')
    plt.plot(eval_metrics['epoch'], eval_metrics['test_acc'], label='Test Accuracy', marker='o')

    # Highlight best epoch
    y_offset = 0.01
    plt.scatter(best_epoch, best_val_acc, color='red', s=100, zorder=5, label=f'Best Validation Epoch')
    plt.text(best_epoch, best_val_acc + y_offset, f'Val: {best_val_acc:.2%}', color='red', ha='center')
    plt.scatter(best_epoch, best_test_acc, color='red', s=100, zorder=5)
    plt.text(best_epoch, best_test_acc + y_offset , f'Test: {best_test_acc:.2%}', color='red', ha='center')

    plt.title(f'Fold {fold} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


result_idx = show_best_result(all_results_df)

eval_metrics_file_paths = Path("/content/models/MUTAG").glob(f"config_{result_idx + 1:03d}_fold_*_eval_metrics.csv")
eval_metrics_file_paths = list(eval_metrics_file_paths)
eval_metrics_file_paths.sort()

for eval_metrics_file_path in eval_metrics_file_paths:
    print(eval_metrics_file_path)
    show_metrics(eval_metrics_file_path)


best_result = all_results_df.loc[result_idx]
config = best_result["config"]
config_idx = len(config_dicts) + 1
fold = 0

train_dataset = dataset[trainval_idx]
val_dataset   = dataset[trainval_idx]
test_dataset  = dataset[test_idx]

dataset_info = dict(
    train=len(train_dataset),
    val=len(val_dataset),
    test=len(test_dataset)
)

print(f"dataset_info: {dataset_info}")

train_dataloader = DataLoader(train_dataset,
    batch_size=config.batch_size,
    shuffle=True)
val_dataloader = DataLoader(val_dataset, shuffle=False)
test_dataloader = DataLoader(test_dataset, shuffle=False)

dataloaders = dict(
    train=train_dataloader,
    val=val_dataloader,
    test=test_dataloader,
    )

model = GINClassification(in_channels=dataset.num_node_features,
    hidden_channels=config.hidden_channels,
    out_channels=config.out_channels,
    num_classes=dataset.num_classes)

trainer = Trainer(config, config_idx, fold, dataloaders, model)

best_result = trainer.train()

eval_metrics_file_path = Path("/content/models/MUTAG") / f"config_{config_idx + 1:03d}_fold_{fold + 1:03d}_eval_metrics.csv"
show_metrics(eval_metrics_file_path)
