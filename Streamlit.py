import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Memuat data
data_EEG = pd.read_csv("Epileptic Seizure Recognition.csv")  # Ganti "dataset.csv" dengan nama file dataset Anda
data_EEG.dropna(inplace=True)
data_EEG.reset_index(drop=True, inplace=True)
def convert_label(x):
    if (x > 1):
        return 0
    else:
        return x
data_EEG["y"] = data_EEG['y'].apply(convert_label)
X = data_EEG.iloc[:,1:179].values
y = data_EEG.iloc[:,179].values

# Fungsi untuk melakukan prediksi dan menampilkan grafik
def make_prediction_and_plot(model_name, model):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    st.subheader(f"{model_name} Prediction")
    fig, ax = plt.subplots()
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
    ax.set_title(f"{model_name} Prediction")
    st.pyplot(fig)

# Antarmuka Streamlit
def main():
    st.set_page_config(page_title="Prediksi Epilepsi", page_icon=":brain:", layout="wide")
    st.title(":brain: Prediksi Epilepsi Berdasarkan Data Sinyal EEG")

    # Tampilan pilihan model
    model_options = ["Support Vector Machine", "Random Forest", "Neural Network"]
    selected_model = st.selectbox("Pilih Model", model_options)

    # Menampilkan grafik prediksi sesuai model yang dipilih
    if selected_model == "Support Vector Machine":
        make_prediction_and_plot("Support Vector Machine", SVC())
    elif selected_model == "Random Forest":
        make_prediction_and_plot("Random Forest", RandomForestClassifier())
    elif selected_model == "Neural Network":
        make_prediction_and_plot("Neural Network", MLPClassifier())

if __name__ == "__main__":
    main()