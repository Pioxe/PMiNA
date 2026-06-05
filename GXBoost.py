import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def przygotuj_dane(df, test_size=0.2, random_state=42):
    # Przygotowuje dane do modelowania: usuwa ID, koduje zmienną docelową, 
    # tworzy zmienne dummy i dzieli zbiór na treningowy i testowy.
    # Struktura identyczna jak w lesie losowym dla zachowania spójności projektu.
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Zabezpieczenie nazw kolumn
    df_encoded.columns = [str(col).replace('[', '').replace(']', '').replace('<', '') for col in df_encoded.columns]

    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns

def optymalizuj_model(X_train, y_train, param_grid, cv=3, random_state=42):
    # Uruchamia GridSearchCV w celu znalezienia najlepszych hiperparametrów
    # dla algorytmu XGBoost. Uwzględnia niezbalansowanie klas za pomocą scale_pos_weight.
    print("Rozpoczynam optymalizację modelu XGBoost (GridSearch)...")
    
    # Obliczenie balansu klas (liczba zer / liczba jedynek) dla scale_pos_weight
    rozklad = y_train.value_counts()
    skala_wag = rozklad[0] / rozklad[1]

    grid_search = GridSearchCV(
        estimator=XGBClassifier(scale_pos_weight=skala_wag, random_state=random_state, eval_metric='logloss'),
        param_grid=param_grid,
        scoring='recall', 
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    
    print("\nOptymalizacja XGBoost zakończona.")
    print(f"Najlepsze znalezione parametry: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def ewaluacja_modelu(model, X_test, y_test):
    # Dokonuje predykcji, wyświetla raport klasyfikacji oraz rysuje macierz pomyłek.
    y_pred = model.predict(X_test)

    print("Tabela metryk klasyfikacji (Classification Report) - XGBoost:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Zostaje (No)', 'Odchodzi (Yes)'])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Oranges', values_format='d', ax=ax)
    plt.title('Macierz Pomyłek zoptymalizowanego XGBoost')
    plt.show()

def wizualizuj_waznosc_cech(model, feature_names, top_n=10):
    # Pobiera ważność cech z wytrenowanego modelu XGBoost i rysuje poziomy wykres słupkowy.
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        x=feature_importances[indices], 
        y=feature_names[indices], 
        hue=feature_names[indices], 
        palette='viridis', 
        legend=False
    )
    plt.title(f'Top {top_n} najważniejszych cech klienta wg algorytmu XGBoost', fontsize=14)
    plt.xlabel('Względna ważność cechy wg algorytmu', fontsize=12)
    plt.ylabel('Cecha w zbiorze', fontsize=12)
    plt.tight_layout()
    plt.show()