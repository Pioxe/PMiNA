#link do bazy https://www.kaggle.com/datasets/blastchar/telco-customer-churn
from baza import wczytanie, czyszczenie, wykresy_podstawowe, analiza_jakosci, mapa_korelacji_top12, analiza_jakosci
df = wczytanie()

analiza_jakosci(df)
df = czyszczenie(df)
analiza_jakosci(df)
wykresy_podstawowe(df)
mapa_korelacji_top12(df)