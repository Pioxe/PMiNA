from baza import wczytanie, czyszczenie
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
#przebudowany kod z https://github.com/Pioxe/PMN-26L_Piotr_K/tree/main/Task_3
# z:class_cifar10.py przerobione
class Telco_MLP(nn.Module):
    # Dynamiczne tworzenie warstw ukrytych na podstawie listy hidden_sizes (jak w CIFAR10_MLP)
    def __init__(self, input_size, num_classes=2, hidden_sizes=[512, 256, 128], dropout_rate=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        current_dim = input_size
        
        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(current_dim, h_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = h_size
            
        self.out = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x) 
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

def initialize_model(device, input_size, hidden_sizes, dropout_rate):
    # input_size, liczba kolumn po One-Hot Encodingu jest zmienna
    return Telco_MLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)

def setup_optimization(model, learning_rate=0.001, optimizer_type='adam'):
    criterion = nn.CrossEntropyLoss()
    opt_class = optim.Adam if optimizer_type.lower() == 'adam' else optim.SGD
    optimizer = opt_class(model.parameters(), lr=learning_rate)
    return criterion, optimizer

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status = f"Wykorzystywane urządzenie: {device.type.upper()}"
    if device.type == 'cuda':
        status += f" ({torch.cuda.get_device_name(0)})"
    print("\n" + "="*len(status))
    print(status)
    print("="*len(status) + "\n")
    return device

def get_telco_loaders(df, batch_size=64):
    
    # Zamiana zmiennych kategorycznych tekstowych na układ 0/1 (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Konwersja ewentualnych kolumn typu bool na int dla PyTocha
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
            
    X = df_encoded.drop('Churn', axis=1).values
    y = df_encoded['Churn'].values
    
    # Podział na zbiór treningowy i testowy (80% / 20%) ze stratyfikacją
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Skalowanie danych (StandardScaler) - kluczowe dla poprawnego działania sieci MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Tworzenie PyTorch DataLoaderów z obiektów TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    input_size = X_train.shape[1]
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False), \
           input_size

# z: engine.py

def run_step(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss, correct, total = 0.0, 0, 0
    
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if is_train: optimizer.zero_grad()
            outputs = model(inputs)
            current_loss = criterion(outputs, labels)
            if is_train:
                current_loss.backward()
                optimizer.step()
            
            loss += current_loss.item() * inputs.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
    return loss / len(loader.dataset), 100. * correct / total

def run_experiment(device, config, train_loader, test_loader, input_size):
    # 'input_size' do inicjalizacji modelu tabelarycznego
    model = initialize_model(device, input_size, config["hidden_layers"], config["dropout"])
    criterion, optimizer = setup_optimization(model, config["lr"], config["optimizer_type"])
    
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    for epoch in range(config["epochs"]):
        t_loss, t_acc = run_step(model, train_loader, criterion, device, optimizer)
        v_loss, v_acc = run_step(model, test_loader, criterion, device)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        print(f"Epoch {epoch+1:02d} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | Val Loss: {v_loss:.4f}")
    
    return history, model

def evaluate_best_model(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Rejestrowanie błędnych predykcji (zapis cech wejściowych zamiast obrazka)
            for i in range(len(preds)):
                if preds[i] != labels[i] and len(misclassified) < 10:
                    misclassified.append((inputs[i].cpu().numpy(), labels[i].item(), preds[i].item()))

    print("RAPORT KLASYFIKACJI (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return all_labels, all_preds, misclassified

def execute_series(device, base_cfg, param_name, values, train_loader, test_loader, input_size):
    """Wykonuje serię testów dla wybranego hiperparametru sieci."""
    results = {}
    for val in values:
        print(f"\n>>> TEST SERII: {param_name} = {val}")
        cfg = base_cfg.copy()
        cfg[param_name] = val
        key = str(val) if isinstance(val, list) else val
        results[key] = run_experiment(device, cfg, train_loader, test_loader, input_size)
    return results


# Z: graph.py


def plot_series_results(results_dict, param_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Zestawienie eksperymentów: Wpływ parametru [{param_name}] na sieć MLP", fontsize=14)

    for val, (history, _) in results_dict.items():
        ax1.plot(history['val_acc'], marker='o', label=f'{param_name}={val}')
        ax2.plot(history['val_loss'], marker='o', label=f'{param_name}={val}')

    ax1.set_title("Dokładność walidacyjna (Validation Accuracy)")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title("Strata walidacyjna (Validation Loss)")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Macierz Pomyłek dla Telco Churn (MLP)")
    plt.xlabel("Przewidywana klasa (Predykcja)")
    plt.ylabel("Rzeczywista klasa (Prawda)")
    plt.tight_layout()
    plt.show()

def plot_single_history_overfitting(history):
    
    epochs = len(history['train_acc'])
    plt.figure(figsize=(14, 5))
    
    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Trening (Train)', color='#1f77b4', lw=2)
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Walidacja (Validation)', color='#ff7f0e', lw=2)
    plt.title('Dokładność modelu: Widoczny Overfitting')
    plt.xlabel('Epoki')
    plt.ylabel('Dokładność (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Trening (Train)', color='#1f77b4', lw=2)
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Walidacja (Validation)', color='#ff7f0e', lw=2)
    plt.title('Funkcja straty (Loss Value)')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def main():
    raw_df = wczytanie()
    cleaned_df = czyszczenie(raw_df)
    
    target_classes = ['Zostaje (No)', 'Odchodzi (Yes)']
    device = get_device()
    
    
    base_config = {
        "batch_size": 64,
        "hidden_layers": [128, 64],  
        "dropout": 0.3,
        "lr": 0.005,
        "optimizer_type": "adam",
        "epochs": 30
    }
    
    train_loader, test_loader, input_features_dim = get_telco_loaders(cleaned_df, batch_size=base_config["batch_size"])
    print(f"Liczba cech wejściowych po kodowaniu One-Hot: {input_features_dim}")
    
    # przeprowadzenie głównego testu sieci i generowanie dowodu na overfitting
    print("\n" + "="*50)
    print("Główny eksperyment")
    print("="*50)
    history, best_model = run_experiment(device, base_config, train_loader, test_loader, input_features_dim)
    
    # Ewaluacja i macierz pomyłek
    y_true, y_pred, _ = evaluate_best_model(best_model, test_loader, device, target_classes)
    plot_single_history_overfitting(history)
    plot_confusion_matrix_heatmap(y_true, y_pred, target_classes)
    
    print("Seria eksperymentów (execute_series)")
    
    architektury_testowe = [
        [32],               
        [128, 64],          
        [512, 256, 128]     
    ]
    
    seria_wynikow = execute_series(
        device=device,
        base_cfg={**base_config, "epochs": 15},  
        param_name="hidden_layers",
        values=architektury_testowe,
        train_loader=train_loader,
        test_loader=test_loader,
        input_size=input_features_dim
    )
    
    plot_series_results(seria_wynikow, "hidden_layers")

if __name__ == "__main__":
    main()