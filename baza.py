import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


def wczytanie():
    # Ładowanie danych
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "blastchar/telco-customer-churn",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    print("WCZYTANIE DANYCH")
    print(f"Liczba rekordów: {len(df)}")
    print("\nTYPY DANYCH (POCZĄTKOWE):")
    print(df.dtypes) 
    return df

def analiza_jakosci(df):
    print("\nANALIZA JAKOŚCI")
    
    # kopia do celów wizualizacji, żeby nie psuć oryginału przed czyszczeniem
    df_temp = df.copy()

    # Sprawdzanie ukrytych spacji
    if df_temp['TotalCharges'].dtype == 'object':
        # spacje na NaN, żeby heatmapa je "zobaczyła"
        df_temp['TotalCharges'] = pd.to_numeric(df_temp['TotalCharges'], errors='coerce')
        
        num_spaces = df_temp['TotalCharges'].isnull().sum()
        print(f"Liczba wierszy z ukrytymi brakami (spacjami) w TotalCharges: {num_spaces}")
        
        if num_spaces > 0:
            indices = df_temp[df_temp['TotalCharges'].isnull()].index
            print(df.loc[indices, ['customerID', 'tenure', 'TotalCharges']])

    # Teraz heatmapa pokaże żółte kreski tam, gdzie były spacje
    plt.figure(figsize=(10, 4))
    sns.heatmap(df_temp.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Mapa braków w danych (żółte kreski = wykryte braki/spacje)')
    plt.show()

def czyszczenie(df):
    print("\nCZYSZCZENIE DANYCH")
    
    #Konwersja TotalCharges na liczby (spacje staną się NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    print(f"Liczba wykrytych braków (NaN) po konwersji: {df['TotalCharges'].isnull().sum()}")

    #Wypełnienie braków zerem (nowi klienci)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    #Usunięcie zbędnych kolumn
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    #Mapowanie Churn na wartości liczbowe
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print("\nFINALNE TYPY KLUCZOWYCH DANYCH:")
    print(df[['TotalCharges', 'Churn']].dtypes)
    return df
def wykresy_podstawowe(df):
    print("ANALIZA I STATYSTYKI")
    
    print("\nOPIS CECH NUMERYCZNYCH:")
    print(df.describe().round(2))

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 5))

    # Wykres 1: Rozkład Churn
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='Churn', palette='viridis')
    plt.title('Rozkład rezygnacji (Churn)')

    # Wykres 2: Staż (tenure) a Churn
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Churn', y='tenure', palette='magma')
    plt.title('Staż klienta a odejście')

    plt.tight_layout()
    plt.show()

def mapa_korelacji_top12(df):
    print("\nGenerowanie szczegółowej mapy korelacji (Top 12)...")
    
    # Kodowanie zmiennych kategorycznych (one-hot encoding)
    df_encoded = pd.get_dummies(df)
    
    # z najwyższą wartością bezwzględną korelacji względem Churn
   
    top_12_features = df_encoded.corr()['Churn'].abs().sort_values(ascending=False).head(12).index
    
    #macierz korelacji 12x12
    plt.figure(figsize=(14, 12))
    corr_matrix_12x12 = df_encoded[top_12_features].corr()
    
    # mapa
    sns.heatmap(corr_matrix_12x12, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5,
                annot_kws={"size": 10}, # Rozmiar czcionki liczb w środku
                square=True)
    
    plt.title('Macierz korelacji 12x12 (Najważniejsze cechy)', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    # tylko kolumny numeryczne do korelacji
    corr_matrix = df.select_dtypes(include=['number']).corr()
    
    sns.heatmap(corr_matrix[['Churn']].sort_values(by='Churn', ascending=False), 
                annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Wpływ cech na odejście klienta (Churn)")
    plt.show()

def main():
    df = wczytanie_i_czyszczenie()
    wykresy_analiza(df)
if __name__ == "__main__":
    main()