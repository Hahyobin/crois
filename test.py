
def main():
    # 예제 GitHub Raw 이미지 URL
    image_url = "streamlitimage.png"
    st.image(image_url, use_column_width=True)    
    st.markdown("""
        <style>
            .title {
                font-size: 50px;
                font-weight: bold;
                color: white;
                text-align: center;
                font-style: italic;
            }
        </style>
        <div class="title">Crois Anomaly Detection</div>
    """, unsafe_allow_html=True)



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
import time
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

# Data integrity check
def check_file_integrity(df, method):
    try:
        preprocess_data(df.copy(), method)
        return True
    except Exception as e:
        st.error(f"File error: {str(e)}")
        return False

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

# 통계값 표시
def display_statistics(df, column):
    st.write(f"Min: {df[column].min()}")
    st.write(f"Max: {df[column].max()}")
    st.write(f"Mean: {df[column].mean()}")
    st.write(f"Median: {df[column].median()}")
    st.write(f"Std: {df[column].std()}")

# 컨트롤 차트 함수
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

# 파레토 차트 함수
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

# 산포도 생성
def create_scatter_plot(df, columns):
    fig, ax = plt.subplots()
    ax.scatter(df[columns[0]], df[columns[1]], alpha=0.5)
    ax.set_title(f'Scatter Plot of {columns[0]} vs {columns[1]}')
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    return fig

# 스펙트로그램 생성
def spectrogram(df, column):
    signal = df[column]
    fs = 1  # 샘플링 주파수
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

# 히스토그램 생성
def create_histogram(df, column):
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=30, alpha=0.7)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    return fig

# FFT 시각화
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

# Z-Score 정규화
def apply_z_score(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

# PCA 분석
def apply_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df)
    return pd.DataFrame(pca_data)

# Autoencoder
def apply_autoencoder(df, encoding_dim=3):
    input_dim = df.shape[1]  # 입력 특성의 수
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(df, df, epochs=50, batch_size=256, shuffle=True, verbose=0)
    encoder = Model(input_layer, encoded)
    encoded_data = encoder.predict(df)
    return pd.DataFrame(encoded_data)

# 결과 데이터 시각화
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
                color: white;
                text-align: center;
                font-style: italic;
            }
        </style>
        <div class="title">Crois Data Mining</div>
    """, unsafe_allow_html=True)

    image_url2 = "222.png"
    st.sidebar.image(image_url2, use_column_width=False)    
    st.sidebar.title("File and Settings")

    folder_path = st.sidebar.text_input('Folder Path', value='C:/Users/slsld/Streamlit')

    file_list = list_csv_files(folder_path)
    if st.sidebar.button('Update File List'):
        file_list = list_csv_files(folder_path)
        file_names = [os.path.basename(file) for file in file_list]
        st.sidebar.write(file_names)    

    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)
    if st.sidebar.button('Merge and Save CSV'):
        result_message = save_merged_csv(folder_path, 'merged_data.csv')
        st.sidebar.success(result_message)

    selected_files = st.sidebar.multiselect('Choose files from folder', file_list)
    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload your file")
    uploaded_file_list = []

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.getvalue())
        uploaded_file_list.append(tfile.name)
        selected_files = []

    all_selected_files = selected_files + uploaded_file_list

    st.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)

    if all_selected_files:
        total_removed_nones = 0
        method = st.radio('Select Preprocessing Method', ['dropna', 'interpolate'])
        for file_path in all_selected_files:
            df = load_data(file_path)
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
    st.caption('데이터 정합성 테스트')
    if all_selected_files and st.button('Check'):
        all_good = True
        for file_path in all_selected_files:
            df = preprocess_data(load_data(file_path), method)
            if not check_file_integrity(df, method):
                all_good = False
        if all_good:
            st.success('All files successfully passed integrity testing')

    st.header('2. Visualisation')
    st.caption('데이터 시각화')
    if all_selected_files:
        df = load_data(all_selected_files[0])
        df_preprocessed = preprocess_data(df, method)

        df_list = [preprocess_data(load_data(file), method) for file in all_selected_files]

        visualization_type = st.selectbox("Choose Visualization", ['Line Plot', 'Control Chart', 'Pareto Chart', 'Spectrogram', 'Scatter', 'Histogram', 'FFT'])

        if visualization_type != 'Scatter':
            column = st.selectbox('Select Column', df.columns)
            if st.button('Generate Visualization'):
                if visualization_type == 'Line Plot':
                    fig = create_plots2(df, column)
                elif visualization_type == 'Control Chart':
                    fig = create_control_chart(df, column)
                elif visualization_type == 'Pareto Chart':
                    fig = create_pareto_chart(df, column)
                elif visualization_type == 'Spectrogram':
                    fig = spectrogram(df, column)
                elif visualization_type == 'Histogram':
                    fig = create_histogram(df, column)
                elif visualization_type == 'FFT':
                    fig = create_fft(df, column)
                st.pyplot(fig)
                display_statistics(df, column)
        else:
            selected_columns = st.multiselect('Select two columns for Scatter Plot', df.columns)
            if len(selected_columns) == 2 and st.button('Generate Scatter Plot'):
                fig = create_scatter_plot(df, selected_columns)
                st.pyplot(fig)

    st.header('3. Data Analyzing')
    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)
    st.sidebar.write('Select Method')
    use_z_score = st.sidebar.checkbox('Z-Score Normalization')
    use_pca = st.sidebar.checkbox('PCA')
    use_autoencoder = st.sidebar.checkbox('Autoencoder')

    if st.button('Analyzing Start'):
        if not all_selected_files:
            st.error('파일을 선택해야 합니다.')
        else:
            # 각 파일에 대해 데이터 분석을 수행
            for file_path in all_selected_files:
                df = load_data(file_path)
                df_preprocessed = preprocess_data(df.copy(), method)  # 전처리된 데이터를 사용
                st.dataframe(df_preprocessed, use_container_width=True)
                results = []

                if use_z_score:
                    df_z_score = apply_z_score(df_preprocessed)
                    plot_data(df_z_score, 'Z-Score Normalization')
                    results.append(('Z-Score Normalization', df_z_score))

                if use_pca:
                    df_pca = apply_pca(df_preprocessed)
                    st.dataframe(df_pca, use_container_width=True)  # 전처리된 데이터를 사용하여 PCA 적용
                    plot_data(df_pca, 'PCA')
                    results.append(('PCA', df_pca))

                if use_autoencoder:
                    df_auto = apply_autoencoder(df_preprocessed)
                    st.dataframe(df_auto, use_container_width=True)
                    plot_data(df_auto, 'Autoencoder')
                    results.append(('Autoencoder', df_auto))

if __name__ == '__main__':
    main()

