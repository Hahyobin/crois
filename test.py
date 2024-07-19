import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from scipy.signal import stft
from PIL import Image

# File listing function
def list_csv_files(folder_path):
    return glob(os.path.join(folder_path, '*.csv'))

# Data loading
def load_data(file_path):
    return pd.read_csv(file_path)

# Data merging
def merge_csv_files(file_list):
    data_frames = []
    for file in file_list:
        df = load_data(file)
        data_frames.append(df)
    merged_df = pd.concat(data_frames, ignore_index=True)
    return merged_df

def save_merged_csv(selected_files, output_filename):
    if selected_files:
        merged_df = merge_csv_files(selected_files)
        output_path = os.path.join(os.path.dirname(selected_files[0]), output_filename)
        merged_df.to_csv(output_path, index=False)
        return f"Saved merged file as {output_path}"
    else:
        return "No CSV files selected to merge."

# Data preprocessing
def preprocess_data(df, method):
    if method == 'dropna':
        df.dropna(inplace=True)
    elif method == 'interpolate':
        df.interpolate(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def check_for_duplicates(df):
    return df.duplicated().sum() > 0

def check_data_types(df):
    column_types = df.dtypes
    type_issues = {col: str(dtype) for col, dtype in column_types.items() if dtype not in [np.int64, np.float64, object]}
    return type_issues

def check_range(df, column, min_value, max_value):
    return ((df[column] < min_value) | (df[column] > max_value)).sum() > 0

def check_file_integrity_extended(df, method):
    results = {
        "no_missing": True,
        "no_duplicates": True,
        "valid_types": True,
        "range_issues": {}
    }

    try:
        df_preprocessed = preprocess_data(df.copy(), method)

        if df_preprocessed.isnull().sum().sum() > 0:
            results["no_missing"] = False

        if check_for_duplicates(df_preprocessed):
            results["no_duplicates"] = False

        type_issues = check_data_types(df_preprocessed)
        if type_issues:
            results["valid_types"] = False
            results["type_issues"] = type_issues

        for column in df_preprocessed.select_dtypes(include=[np.number]).columns:
            if check_range(df_preprocessed, column, -1000, 1000):
                results["range_issues"][column] = True

        return results
    except Exception as e:
        st.error(f"File error: {str(e)}")
        return None

# Data visualization functions
def create_plots2(df, column):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[column], label=f'{column}')
    ax.set_title(f'Line Plot of {column}')
    ax.set_xlabel('Index')
    ax.set_ylabel(column)
    ax.legend()
    plt.tight_layout()
    return fig

# statistics
def display_statistics(df, column):
    st.markdown(f"**Min:** {df[column].min()}")
    st.markdown(f"**Max:** {df[column].max()}")
    st.markdown(f"**Mean:** {df[column].mean()}")
    st.markdown(f"**Median:** {df[column].median()}")
    st.markdown(f"**Std:** {df[column].std()}")

# control chart
def create_control_chart(df, column):
    fig, ax = plt.subplots(figsize=(10, 5))
    cl = df[column].mean()
    ucl = cl + 3*df[column].std()
    lcl = cl - 3*df[column].std()
    ax.plot(df[column], marker='o', linestyle='-')
    ax.axhline(y=cl, color='green', label='CL')
    ax.axhline(y=ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(y=lcl, color='blue', linestyle='--', label='LCL')
    ax.set_title(f'Control Chart for {column}')
    ax.set_xlabel('Index')
    ax.set_ylabel(column)
    ax.legend()
    return fig

# pareto chart
def create_pareto_chart(df, column):
    data = df[column].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots()
    data.plot(kind='bar', ax=ax)
    ax.set_xticklabels([])
    ax.set_ylabel('Frequency')
    ax2 = ax.twinx()
    ax2.plot(data.cumsum() / data.sum() * 100, marker='D', color='green')
    ax2.set_ylim(0, 110)
    return fig

# scatter
def create_scatter_plot(df, columns):
    fig, ax = plt.subplots()
    ax.scatter(df[columns[0]], df[columns[1]], alpha=0.5)
    ax.set_title(f'Scatter Plot of {columns[0]} vs {columns[1]}')
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    return fig

# spectrogram
def spectrogram(df, column):
    signal = df[column]
    fs = 5000  # 샘플링 주파수
    nperseg = 256  # 각 세그먼트의 길이
    noverlap = 128  # 세그먼트 중첩
    nfft = 512  # FFT 점의 수
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), shading='auto')
    plt.title('STFT Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    return plt.gcf()

# histogram
def create_histogram(df, column):
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=30, alpha=0.7)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    return fig

# FFT
def create_fft(df, column):
    signal = df[column]
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal))
    fig, ax = plt.subplots()
    ax.plot(fft_freq, np.abs(fft_vals))
    ax.set_title(f'FFT of {column}')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    return fig

# Z-Score
def apply_z_score(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

# PCA
def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df)
    return pd.DataFrame(pca_data)

# Autoencoder
def apply_autoencoder(df, encoding_dim=3):
    input_dim = df.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(df, df, epochs=50, batch_size=256, shuffle=True, verbose=0)
    encoder = Model(input_layer, encoded)
    encoded_data = encoder.predict(df)
    return pd.DataFrame(encoded_data)

# plotting
def plot_data(df, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    if df.shape[1] <= 2:
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0], alpha=0.7)
        ax.set_title(f'{title} - Scatter Plot')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2' if df.shape[1] > 1 else 'Value')
    else:
        for col in df.columns:
            ax.plot(df[col], label=col)
        ax.set_title(f'{title} - Line Plot')
        ax.legend()
    st.pyplot(fig)

def main():
    image_url = "111.png"
    st.image(image_url, use_column_width=True)    
    st.markdown("""
        <style>
            .title {
                font-size: 50px;
                font-weight: bold;
                text-align: center;
                font-style: italic;
            }
        </style>
        <div class="title">Crois Data Mining</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<style>.subtitle{text-align:center;font-size:17px;font-style: italic;}</style><div class="subtitle">Data Processing & Analysis Tool</div>""", unsafe_allow_html=True)
    
    st.title('Data Processing & Analysis Tool')

    menu = ['Data Merging', 'File Checking', 'Data Cleaning', 'Analysis', 'Dimensionality Reduction']
    choice = st.sidebar.selectbox('Select Activity', menu)

    folder_path = st.sidebar.text_input('Enter folder path', '')

    if choice == 'Data Merging':
        st.subheader('Merge CSV Files')

        if folder_path:
            files = list_csv_files(folder_path)
            selected_files = st.multiselect('Select files to merge', files)
            output_filename = st.text_input('Enter name for merged file', 'merged.csv')
            if st.button('Merge'):
                result_msg = save_merged_csv(selected_files, output_filename)
                st.success(result_msg)
        else:
            st.warning('Please enter a valid folder path.')

    if choice == 'File Checking':
        st.subheader('Check File Integrity')

        if folder_path:
            files = list_csv_files(folder_path)
            selected_file = st.selectbox('Select file', files)
            method = st.selectbox('Select method to handle missing data', ['dropna', 'interpolate'])
            if st.button('Check'):
                df = load_data(selected_file)
                results = check_file_integrity_extended(df, method)
                if results:
                    st.write(f"No Missing Data: {results['no_missing']}")
                    st.write(f"No Duplicates: {results['no_duplicates']}")
                    st.write(f"Valid Data Types: {results['valid_types']}")
                    st.write(f"Range Issues: {results['range_issues']}")
                else:
                    st.error("Error checking file integrity.")
        else:
            st.warning('Please enter a valid folder path.')

    if choice == 'Data Cleaning':
        st.subheader('Data Cleaning')
        if folder_path:
            files = list_csv_files(folder_path)
            selected_file = st.selectbox('Select file', files)
            method = st.selectbox('Select method to handle missing data', ['dropna', 'interpolate'])
            if st.button('Clean Data'):
                df = load_data(selected_file)
                df_cleaned = preprocess_data(df, method)
                st.write(df_cleaned)
                st.success('Data cleaned successfully.')
        else:
            st.warning('Please enter a valid folder path.')

    if choice == 'Analysis':
        st.subheader('Data Analysis')
        if folder_path:
            files = list_csv_files(folder_path)
            selected_file = st.selectbox('Select file', files)
            if selected_file:
                df = load_data(selected_file)
                columns = df.columns.tolist()
                analysis_type = st.selectbox('Select analysis type', ['Statistics', 'Control Chart', 'Pareto Chart', 'Scatter Plot', 'Spectrogram', 'Histogram', 'FFT'])
                if analysis_type == 'Statistics':
                    column = st.selectbox('Select column for statistics', columns)
                    if column:
                        display_statistics(df, column)
                elif analysis_type == 'Control Chart':
                    column = st.selectbox('Select column for control chart', columns)
                    if column:
                        fig = create_control_chart(df, column)
                        st.pyplot(fig)
                elif analysis_type == 'Pareto Chart':
                    column = st.selectbox('Select column for pareto chart', columns)
                    if column:
                        fig = create_pareto_chart(df, column)
                        st.pyplot(fig)
                elif analysis_type == 'Scatter Plot':
                    scatter_columns = st.multiselect('Select two columns for scatter plot', columns)
                    if len(scatter_columns) == 2:
                        fig = create_scatter_plot(df, scatter_columns)
                        st.pyplot(fig)
                elif analysis_type == 'Spectrogram':
                    column = st.selectbox('Select column for spectrogram', columns)
                    if column:
                        fig = spectrogram(df, column)
                        st.pyplot(fig)
                elif analysis_type == 'Histogram':
                    column = st.selectbox('Select column for histogram', columns)
                    if column:
                        fig = create_histogram(df, column)
                        st.pyplot(fig)
                elif analysis_type == 'FFT':
                    column = st.selectbox('Select column for FFT', columns)
                    if column:
                        fig = create_fft(df, column)
                        st.pyplot(fig)
        else:
            st.warning('Please enter a valid folder path.')

    if choice == 'Dimensionality Reduction':
        st.subheader('Dimensionality Reduction')
        if folder_path:
            files = list_csv_files(folder_path)
            selected_file = st.selectbox('Select file', files)
            if selected_file:
                df = load_data(selected_file)
                columns = df.columns.tolist()
                dr_method = st.selectbox('Select dimensionality reduction method', ['Z-Score', 'PCA', 'Autoencoder'])
                if dr_method == 'Z-Score':
                    df_zscore = apply_z_score(df)
                    plot_data(df_zscore, 'Z-Score Normalized Data')
                elif dr_method == 'PCA':
                    n_components = st.slider('Select number of PCA components', 2, min(len(columns), 10))
                    df_pca = apply_pca(df, n_components)
                    plot_data(df_pca, 'PCA Reduced Data')
                elif dr_method == 'Autoencoder':
                    encoding_dim = st.slider('Select encoding dimension for Autoencoder', 2, min(len(columns), 10))
                    df_autoencoder = apply_autoencoder(df, encoding_dim)
                    plot_data(df_autoencoder, 'Autoencoder Reduced Data')
        else:
            st.warning('Please enter a valid folder path.')

if __name__ == '__main__':
    main()
