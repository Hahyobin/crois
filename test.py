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

# File listing function
def list_csv_files(folder_path):
    return glob(os.path.join(folder_path, '*.csv'))

# Data loading
def load_data(file_path):
    return pd.read_csv(file_path)

# Data preprocessing
def preprocess_data(df):
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Data integrity check
def check_file_integrity(df):
    try:
        preprocess_data(df.copy())
        return True
    except Exception as e:
        st.error(f"File error: {str(e)}")
        return False

# Data visualization functions
def create_plots(df):
    num_columns = len(df.columns)
    fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 3))
    if num_columns == 1:
        axes = [axes]
    for i, column in enumerate(df.columns):
        axes[i].plot(df.index, df[column], label=f'{column}')
        axes[i].set_title(f'Line Plot of {column}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel(column)
        axes[i].legend()
    plt.tight_layout()
    return fig

def create_plots2(df, column):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[column], label=f'{column}')
    ax.set_title(f'Line Plot of {column}')
    ax.set_xlabel('Index')
    ax.set_ylabel(column)
    ax.legend()
    return fig

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
    if df.shape[1] <= 2:  # 저차원 데이터인 경우 산점도 사용
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0], alpha=0.7)
        ax.set_title(f'{title} - Scatter Plot')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2' if df.shape[1] > 1 else 'Value')
    else:  # 고차원 데이터인 경우 선 그래프 사용
        for col in df.columns:
            ax.plot(df[col], label=col)
        ax.set_title(f'{title} - Line Plot')
        ax.legend()
    st.pyplot(fig)
    
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

    # 사이드바 설정
    st.sidebar.title("File and Settings")
    folder_path = st.sidebar.text_input('Folder Path', value='C:/Users/')
    update_interval = st.sidebar.number_input('Update Interval (minutes)', min_value=1, value=1)

    file_list = list_csv_files(folder_path)

    # 파일 목록 초기화, 업데이트
    if st.sidebar.button('Update File List'):
        file_list = list_csv_files(folder_path)
        file_names = [os.path.basename(file) for file in file_list]
        st.sidebar.write(file_names)    

    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)

    # 파일 선택
    selected_files = st.sidebar.multiselect('Choose files from folder', file_list)

    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader("Upload your file")
    uploaded_file_list = []

    if uploaded_file is not None:
        # 임시 디렉토리에 파일 저장
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.getvalue())
        uploaded_file_list.append(tfile.name)

    all_selected_files = selected_files + uploaded_file_list

    st.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)

    # 데이터 시각화
    if all_selected_files:
        if st.button("View Data"):
            for file_path in all_selected_files:
                df = load_data(file_path)
                df_preprocessed = preprocess_data(df.copy())

                col1, col2 = st.columns(2)
                col1.metric(label="data", value="Raw")
                col2.metric(label="data", value="Preprocessed")

                with col1:
                    st.dataframe(df.head(), use_container_width=True)
                    st.text(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')
                    st.download_button(
                        label='Download csv',
                        data=df.to_csv(), 
                        file_name='Raw data.csv', 
                        mime='text/csv'
                    )

                with col2:
                    st.dataframe(df_preprocessed.head(), use_container_width=True)
                    st.text(f'Rows: {df_preprocessed.shape[0]}, Columns: {df_preprocessed.shape[1]}')
                    st.download_button(
                        label='Download csv',
                        data=df_preprocessed.to_csv(), 
                        file_name='Preprocessed data.csv', 
                        mime='text/csv'
                    )

    st.header('1. Data Integrity test')
    st.caption('데이터 정합성 테스트')
    if all_selected_files and st.button('Check'):
        all_good = True
        for file_path in all_selected_files:
            df = preprocess_data(load_data(file_path))
            if not check_file_integrity(df):
                all_good = False
        if all_good:
            st.success('All files successfully passed integrity testing')

    st.header('2. Visualisation')
    st.caption('데이터 시각화')
    if all_selected_files:
        df = load_data(all_selected_files[0])
        df_preprocessed = preprocess_data(df)

        df_list = [preprocess_data(load_data(file)) for file in all_selected_files]

        visualization_type = st.selectbox("Choose Visualization", ['Line Plot', 'Control Chart', 'Pareto Chart', 'Spectrogram', 'Scatter'])
        
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
                st.pyplot(fig)
        else:
            selected_columns = st.multiselect('Select two columns for Scatter Plot', df.columns)
            if len(selected_columns) == 2 and st.button('Generate Scatter Plot'):
                fig = create_scatter_plot(df, selected_columns)
                st.pyplot(fig)

    st.header('3. Model Training')
    st.sidebar.markdown('<hr style="border:1px solid gray;">', unsafe_allow_html=True)
    st.sidebar.write('Select Model')
    use_z_score = st.sidebar.checkbox('Z-Score Normalization')
    use_pca = st.sidebar.checkbox('PCA')
    use_autoencoder = st.sidebar.checkbox('Autoencoder')
         
    if st.button('Training Start'):
        if not all_selected_files:
            st.error('파일을 선택해야 합니다.')
        else:
            # 각 파일에 대해 모델 훈련을 수행
            for file_path in all_selected_files:
                df = preprocess_data(load_data(file_path))
                st.dataframe(df, use_container_width=True)           
                original_df = df.copy()
                #results = []
                
                if use_z_score:
                    df = apply_z_score(df)
                    plot_data(df, 'Z-Score Normalization')
                    results.append(('Z-Score Normalization', df))
                    
                if use_pca:
                    df_pca = apply_pca(original_df)
                    st.dataframe(df, use_container_width=True)           
                    plot_data(df_pca, 'PCA')
                    results.append(('PCA', df_pca))
                    
                if use_autoencoder:
                    df_auto = apply_autoencoder(original_df)
                    st.dataframe(df, use_container_width=True)           
                    plot_data(df_auto, 'Autoencoder')
                    results.append(('Autoencoder', df_auto))

if __name__ == '__main__':
    main()
