import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def przygotuj_dane(df, test_size=0.2, random_state=42):
    """
    Przygotowuje dane do modelowania: usuwa ID, koduje zmienną docelową, 
    tworzy zmienne dummy i dzieli zbiór na treningowy i testowy.
    """
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Zwracamy również kolumny X, by móc z nich później skorzystać przy wykresach
    return X_train, X_test, y_train, y_test, X.columns

def optymalizuj_model(X_train, y_train, param_grid, cv=3, random_state=42):
    """
    Uruchamia GridSearchCV w celu znalezienia najlepszych hiperparametrów
    dla algorytmu Random Forest.
    """
    print("Rozpoczynam optymalizację modelu (GridSearch)...")
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(class_weight='balanced', random_state=random_state),
        param_grid=param_grid,
        scoring='recall', 
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    
    print("\nOptymalizacja zakończona.")
    print(f"Najlepsze znalezione parametry: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def ewaluacja_modelu(model, X_test, y_test):
    """
    Dokonuje predykcji, wyświetla raport klasyfikacji oraz rysuje macierz pomyłek.
    """
    y_pred = model.predict(X_test)

    print("Tabela metryk klasyfikacji (Classification Report):")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Zostaje (No)', 'Odchodzi (Yes)'])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title('Macierz Pomyłek zoptymalizowanego Lasu Losowego')
    plt.show()

def wizualizuj_waznosc_cech(model, feature_names, top_n=10):
    """
    Pobiera ważność cech z wytrenowanego modelu i rysuje poziomy wykres słupkowy.
    """
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    
    # Naprawiono FutureWarning z seaborn dodając hue i legend=False
    sns.barplot(
        x=feature_importances[indices], 
        y=feature_names[indices], 
        hue=feature_names[indices], 
        palette='magma', 
        legend=False
    )
    plt.title(f'Top {top_n} najważniejszych cech klienta przy decyzji o odejściu', fontsize=14)
    plt.xlabel('Względna ważność cechy wg algorytmu', fontsize=12)
    plt.ylabel('Cecha w zbiorze', fontsize=12)
    plt.tight_layout()
    plt.show()