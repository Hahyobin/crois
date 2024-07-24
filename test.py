import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tempfile
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

def save_merged_csv(folder_path, output_filename):
    file_list = list_csv_files(folder_path)
    if file_list:
        merged_df = merge_csv_files(file_list)
        output_path = os.path.join(folder_path, output_filename)
        merged_df.to_csv(output_path, index=False)
        return f"Saved merged file as {output_path}"
    else:
        return "No CSV files found to merge."
    
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
                font-size: 60px;
                font-weight: bold;
                text-align: center;
                font-style: italic;
            }
        </style>
        <div class="title">Crois Data Mining</div>
    """, unsafe_allow_html=True)

    st.markdown("""<style>.subtitle{text-align:center;font-size:17px;font-style: italic;}</style><div class="subtitle">Data Processing and Analysis Application</div>""", unsafe_allow_html=True)
 
    image_url2 = "222.png"
    st.sidebar.image(image_url2, use_column_width=False)    
    st.sidebar.title("File and Settings")

    folder_path = st.sidebar.text_input('Folder Path', value='')

    file_list = list_csv_files(folder_path)
    if st.sidebar.button('Update File List'):
        file_list = list_csv_files(folder_path)
        file_names = [os.path.basename(file) for file in file_list]
        st.sidebar.write(file_names)    

    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)
    if st.sidebar.button('Merge and Save'):
        result_message = save_merged_csv(folder_path, 'merged_data.csv')
        st.sidebar.success(result_message)

    selected_files = st.sidebar.multiselect('Choose files from folder', file_list)
    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload your file")

    if selected_files:
        uploaded_file = None
    if uploaded_file:
        selected_files = []

    all_selected_files = selected_files + ([uploaded_file.name] if uploaded_file else [])

    st.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)

    if all_selected_files:
        total_removed_nones = 0
        method = st.selectbox('Select Preprocessing Method', ['dropna', 'interpolate'])
        for file_path in all_selected_files:
            df = load_data(file_path)
            if df is not None:
                df_preprocessed = preprocess_data(df.copy(), method)
                total_removed_nones += df.isnull().sum().sum()

                col1, col2 = st.columns(2)
                col1.metric(label="Data", value="Raw")
                col2.metric(label="Data", value="Preprocessed")

                with col1:
                    st.dataframe(df.head(), use_container_width=True)
                    st.text(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')
                    st.download_button(
                        label='Download csv',
                        data=df.to_csv(), 
                        file_name='Raw data.csv', 
                        mime='text/csv')

                with col2:
                    st.dataframe(df_preprocessed.head(), use_container_width=True)
                    st.text(f'Rows: {df_preprocessed.shape[0]}, Columns: {df_preprocessed.shape[1]}')
                    st.download_button(
                        label='Download csv',
                        data=df_preprocessed.to_csv(), 
                        file_name='Preprocessed data.csv', 
                        mime='text/csv')      
        st.write(f'Data preprocessing has been completed')
    
    st.header('1. Data Integrity test')
    st.caption('데이터 전처리 여부 및 정합성 테스트')

    if all_selected_files and st.button('Check'):
        all_good = True
        for file_path in all_selected_files:
            df = preprocess_data(load_data(file_path), method)
            if df is not None:
                integrity_results = check_file_integrity_extended(df, method)
                if integrity_results:
                    st.subheader("Integrity Check Results")
                    st.write("No missing values:", integrity_results["no_missing"])
                    st.write("No duplicate rows:", integrity_results["no_duplicates"])
                    st.write("Valid data types:", integrity_results["valid_types"])

                    if not integrity_results["no_missing"]:
                        st.warning(f"Missing data detected in {file_path}")
                        all_good = False
                    if not integrity_results["no_duplicates"]:
                        st.warning(f"Duplicate data detected in {file_path}")
                        all_good = False
                    if not integrity_results["valid_types"]:
                        st.warning(f"Invalid data types detected in {file_path}: {integrity_results['type_issues']}")
                        all_good = False
                    if integrity_results["range_issues"]:
                        for col, issue in integrity_results["range_issues"].items():
                            if issue:
                                st.warning(f"Data out of range detected in column {col} in {file_path}")
                                all_good = False
        if all_good:
            st.success('All files successfully passed integrity testing')

    st.header('2. Data Visualization')
    st.caption('데이터 시각화')
    if all_selected_files:
        df = load_data(all_selected_files[0])
        if df is not None:
            df_preprocessed = preprocess_data(df, method)

            df_list = [preprocess_data(load_data(file), method) for file in all_selected_files if load_data(file) is not None]

            visualization_type = st.selectbox("Choose Visualization", ['Line Plot', 'Control Chart', 'Pareto Chart', 'Spectrogram', 'Scatter', 'Histogram', 'FFT'])
            result = []

            if visualization_type != 'Scatter':
                column = st.selectbox('Select Column', df.columns)
                if st.button('Generate Visualization'):
                    if visualization_type == 'Line Plot':
                        fig = create_plots2(df, column)
                        result.append('Line Plot completed')
                    elif visualization_type == 'Control Chart':
                        fig = create_control_chart(df, column)
                        result.append('Control chart completed')
                    elif visualization_type == 'Pareto Chart':
                        fig = create_pareto_chart(df, column)
                        result.append('Pareto chart completed')
                    elif visualization_type == 'Spectrogram':
                        fig = spectrogram(df, column)
                        result.append('Spectrogram completed')
                    elif visualization_type == 'Histogram':
                        fig = create_histogram(df, column)
                        result.append('Histogram completed')
                    elif visualization_type == 'FFT':
                        fig = create_fft(df, column)
                        result.append('FFT completed')
                    st.pyplot(fig)
                    display_statistics(df, column)
                    if result:
                        st.success(' and '.join(result))
                    
            else:
                selected_columns = st.multiselect('Select two columns for Scatter Plot', df.columns)
                if len(selected_columns) == 2 and st.button('Generate Scatter Plot'):
                    fig = create_scatter_plot(df, selected_columns)
                    result.append('Scatter completed')
                    st.pyplot(fig)
                    if result:
                        st.success(' and '.join(result))

    st.header('3. Data Analysis')
    st.caption('Sidebar에서 분석 방법을 선택해주세요')
    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)
    st.sidebar.write('Select Method')
    use_z_score = st.sidebar.checkbox('Z-Score Normalization')
    use_pca = st.sidebar.checkbox('PCA')
    use_autoencoder = st.sidebar.checkbox('Autoencoder')

    if st.button('Analyzing Start'):
        if not all_selected_files:
            st.error('파일을 선택해야 합니다.')
        else:
            for file_path in all_selected_files:
                df = load_data(file_path)
                if df is not None:
                    df_preprocessed = preprocess_data(df.copy(), method)
                    results = []

                    if use_z_score:
                        df_z_score = apply_z_score(df_preprocessed)
                        plot_data(df_z_score, 'Z-Score Normalization')
                        results.append('Z-Score Normalization completed')

                    if use_pca:
                        df_pca = apply_pca(df_preprocessed)
                        plot_data(df_pca, 'PCA Analysis')
                        results.append('PCA Analysis completed')

                    if use_autoencoder:
                        df_autoencoder = apply_autoencoder(df_preprocessed)
                        plot_data(df_autoencoder, 'Autoencoder Analysis')
                        results.append('Autoencoder Analysis completed')

                    if results:
                        st.success(' and '.join(results))
                    else:
                        st.warning('No analysis method selected')

if __name__ == "__main__":
    main()
