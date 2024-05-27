import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense

# Fungsi untuk menampilkan informasi dataset
def display_data_info(data):
    st.write("## Informasi Dataset")
    st.write(data.describe())
    st.write("## Distribusi Kelas")
    st.bar_chart(data['y'].value_counts())

# Fungsi untuk menampilkan beberapa sampel EEG
def plot_some_samples(X, y, seizure_index_list, normal_index_list):
    st.write("## Contoh Sampel EEG")
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))

    for i in range(5):
        axes[i, 0].plot(X[seizure_index_list[i], :])
        axes[i, 0].set_title('Seizure EEG')
        axes[i, 1].plot(X[normal_index_list[i], :])
        axes[i, 1].set_title('Normal EEG')

    st.pyplot(fig)

# Fungsi untuk prediksi menggunakan model yang dilatih
def predict(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test) * 100
    return y_pred, accuracy

# Load dataset
@st.cache
def load_data():
    data_path = 'Epileptic Seizure Recognition.csv'
    data = pd.read_csv(data_path)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['y'] = data['y'].apply(lambda x: 0 if x > 1 else x)
    return data

data = load_data()

# Sidebar navigation
st.sidebar.title('Navigasi')
options = st.sidebar.radio('Pilih', ['Informasi Dataset', 'Visualisasi Sampel EEG', 'Prediksi'])

if options == 'Informasi Dataset':
    display_data_info(data)

elif options == 'Visualisasi Sampel EEG':
    X = data.iloc[:, 1:179].values
    y = data.iloc[:, 179].values

    seizure_index_list = [i for i in range(len(y)) if y[i] == 1]
    normal_index_list = [i for i in range(len(y)) if y[i] == 0]

    plot_some_samples(X, y, seizure_index_list, normal_index_list)

elif options == 'Prediksi':
    st.write("## Prediksi Epilepsi")
    X = data.iloc[:, 1:179].values
    y = data.iloc[:, 179].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model_choice = st.selectbox("Pilih Model", ["SVM", "Random Forest", "Neural Network"])

    if model_choice == "SVM":
        svm_clf = SVC()
        svm_clf.fit(X_train, y_train)
        y_pred, accuracy = predict(svm_clf, X_test, y_test)
        st.write(f"Accuracy SVM: {accuracy:.2f}%")
        
    elif model_choice == "Random Forest":
        rfc_clf = RandomForestClassifier()
        rfc_clf.fit(X_train, y_train)
        y_pred, accuracy = predict(rfc_clf, X_test, y_test)
        st.write(f"Accuracy Random Forest: {accuracy:.2f}%")

    elif model_choice == "Neural Network":
        model = Sequential()
        model.add(Dense(80, activation='relu', input_dim=178))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0)
        score = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Accuracy Neural Network: {score[1] * 100:.2f}%")

    st.write("### Prediksi untuk Satu Sampel")
    sample_index = st.number_input("Pilih Index Sampel", min_value=0, max_value=len(X_test)-1, value=0)
    sample = np.expand_dims(X_test[sample_index], 0)
    
    if model_choice == "SVM":
        pred = svm_clf.predict(sample)
    elif model_choice == "Random Forest":
        pred = rfc_clf.predict(sample)
    elif model_choice == "Neural Network":
        pred = (model.predict(sample) > 0.5).astype(int)

    st.write(f"Prediksi: {'Seizure' if pred[0] == 1 else 'Non-Seizure'}")
    st.write(f"Ground Truth: {'Seizure' if y_test[sample_index] == 1 else 'Non-Seizure'}")

    plt.plot(X_test[sample_index])
    plt.title(f"EEG Sampel Index {sample_index}")
    plt.xlabel('Sampel')
    plt.ylabel('uV')
    st.pyplot(plt.gcf())
