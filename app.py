# =====================================================================
# IMPORT LIBRARY
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE

# =====================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =====================================================================
st.set_page_config(page_title="Deteksi Dini Diabetes", page_icon="🩺", layout="wide")

# Daftar fitur klinis utama yang RELEVAN untuk deteksi diabetes
FITUR_KLINIS = [
    'Age', 'BMI', 'HighBP', 'HighChol', 'PhysActivity', 'Smoker',
    'Fruits', 'Veggies'
]

NAMA_FITUR_LIME = {
    'Age': 'Kelompok usia',
    'BMI': 'Indeks massa tubuh (BMI)',
    'HighBP': 'Tekanan darah tinggi',
    'HighChol': 'Kolesterol tinggi',
    'PhysActivity': 'Aktivitas fisik',
    'Smoker': 'Riwayat merokok',
    'Fruits': 'Konsumsi buah harian',
    'Veggies': 'Konsumsi sayuran harian'
}

KATEGORI_LIME = {
    'Age': [
        'Tidak diketahui', '18-24 tahun', '25-29 tahun', '30-34 tahun',
        '35-39 tahun', '40-44 tahun', '45-49 tahun', '50-54 tahun',
        '55-59 tahun', '60-64 tahun', '65-69 tahun', '70-74 tahun',
        '75-79 tahun', '80 tahun ke atas'
    ],
    'HighBP': ['Tidak', 'Ya'],
    'HighChol': ['Tidak', 'Ya'],
    'PhysActivity': ['Tidak', 'Ya'],
    'Smoker': ['Tidak', 'Ya'],
    'Fruits': ['Tidak', 'Ya'],
    'Veggies': ['Tidak', 'Ya']
}

JUMLAH_BIN_BMI = 10


def quantize_bmi(features, quantizer):
    quantized = features.copy()
    quantized['BMI'] = quantizer.transform(features[['BMI']]).ravel().astype(int)
    return quantized

@st.cache_resource(show_spinner="Sedang melatih model AI... Mohon tunggu sebentar (sekitar 1-2 menit).")
def train_model(csv_path, fitur_klinis):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # HANYA gunakan fitur klinis yang relevan
    X = df[fitur_klinis]
    y = df['Diabetes_012']
    
    # 2. Split Data (80% Train, 20% Test) dengan Stratified
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    bmi_quantizer = KBinsDiscretizer(
        n_bins=JUMLAH_BIN_BMI,
        encode='ordinal',
        strategy='quantile'
    )
    bmi_quantizer.fit(X_train[['BMI']])
    
    # 3. Penanganan Imbalanced Data (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    X_train_res = quantize_bmi(X_train_res, bmi_quantizer)
    X_test_quantized = quantize_bmi(X_test, bmi_quantizer)
    
    # 4. Pelatihan Model Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_res, y_train_res)
    
    y_test_pred = rf_model.predict(X_test_quantized)
    class_names = ['Tidak Diabetes', 'Prediabetes', 'Diabetes']
    evaluation = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision_macro': precision_score(
            y_test, y_test_pred, average='macro', zero_division=0
        ),
        'recall_macro': recall_score(
            y_test, y_test_pred, average='macro', zero_division=0
        ),
        'f1_macro': f1_score(
            y_test, y_test_pred, average='macro', zero_division=0
        ),
        'report': classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1, 2],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        ),
        'confusion_matrix': confusion_matrix(
            y_test, y_test_pred, labels=[0, 1, 2]
        )
    }

    return rf_model, X_train, X.columns, evaluation, bmi_quantizer

# =====================================================================
# FUNGSI UNTUK MEMBUAT PENJELASAN LIME (XAI)
# =====================================================================
def generate_lime_explanation(
    model, X_train, input_instance, feature_names, predicted_class, bmi_quantizer
):
    feature_names = list(feature_names)
    categorical_features = [
        index for index, name in enumerate(feature_names)
        if name in KATEGORI_LIME
    ]
    categorical_names = {
        index: KATEGORI_LIME[name]
        for index, name in enumerate(feature_names)
        if name in KATEGORI_LIME
    }

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=[NAMA_FITUR_LIME[name] for name in feature_names],
        class_names=['Non-Diabetes', 'Prediabetes', 'Diabetes'],
        mode='classification',
        categorical_features=categorical_features,
        categorical_names=categorical_names
    )
    
    def predict_proba(samples):
        samples_df = pd.DataFrame(samples, columns=feature_names)
        quantized_samples = quantize_bmi(samples_df, bmi_quantizer)
        return model.predict_proba(quantized_samples)

    exp = explainer.explain_instance(
        data_row=input_instance[feature_names].iloc[0].to_numpy(),
        predict_fn=predict_proba,
        labels=(int(predicted_class),),
        num_features=len(feature_names)
    )
    return exp

# =====================================================================
# FUNGSI UTAMA APLIKASI WEB
# =====================================================================
def main():
    st.title("🩺 Sistem Deteksi Dini Risiko Diabetes Melitus")
    st.markdown("""
    Aplikasi ini menggunakan model **Random Forest** untuk mendeteksi risiko Diabetes Melitus berdasarkan **data klinis utama**. 
    Dilengkapi dengan **Explainable AI (LIME)** untuk transparansi hasil prediksi.
    """)
    
    csv_file = 'diabetes_012_health_indicators_BRFSS2015.csv'
    
    if not os.path.exists(csv_file):
        st.error(f"⚠️ File dataset '{csv_file}' tidak ditemukan! Pastikan file berada di folder yang sama dengan app.py.")
        return

    # --- PROSES LOAD MODEL ---
    with st.spinner("Memuat model & memproses data..."):
        rf_model, X_train, feature_names, evaluation, bmi_quantizer = train_model(
            csv_file, FITUR_KLINIS
        )

    st.subheader("Evaluasi Model")
    st.caption(
        "Metrik dihitung dari data uji 20% yang tidak digunakan saat pelatihan. "
        "SMOTE hanya diterapkan pada data latih."
    )
    st.caption(
        f"Kuantisasi fitur: BMI diubah menjadi "
        f"{int(bmi_quantizer.n_bins_[0])} bin kuantil berdasarkan data latih."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Akurasi", f"{evaluation['accuracy']:.2%}")
    metric_columns[1].metric(
        "Presisi Macro", f"{evaluation['precision_macro']:.2%}"
    )
    metric_columns[2].metric(
        "Recall Macro", f"{evaluation['recall_macro']:.2%}"
    )
    metric_columns[3].metric("F1 Macro", f"{evaluation['f1_macro']:.2%}")

    with st.expander("Detail evaluasi per kelas"):
        report = pd.DataFrame(evaluation['report']).T
        report_rows = [
            'Tidak Diabetes', 'Prediabetes', 'Diabetes', 'macro avg',
            'weighted avg'
        ]
        st.dataframe(
            report.loc[
                report_rows, ['precision', 'recall', 'f1-score', 'support']
            ].round(3),
            width='stretch'
        )

        matrix = pd.DataFrame(
            evaluation['confusion_matrix'],
            index=['Aktual: Tidak Diabetes', 'Aktual: Prediabetes', 'Aktual: Diabetes'],
            columns=['Prediksi: Tidak Diabetes', 'Prediksi: Prediabetes', 'Prediksi: Diabetes']
        )
        st.write("Confusion matrix")
        st.dataframe(matrix, width='stretch')
    
    # --- SIDEBAR: INPUT PENGGUNA (HANYA FITUR KLINIS) ---
    st.sidebar.header("📋 Formulir Data Klinis Pasien")
    st.sidebar.write("Silakan isi data klinis di bawah ini:")
    
    input_data = {}
    
    # Mapping nama fitur agar lebih ramah dibaca (Human-Readable)
    label_map = {
        'Age': 'Kategori Usia (1=18-24, 13=80+)',
        'BMI': 'Indeks Massa Tubuh (BMI; tinggi dan berat terpisah tidak tersedia)',
        'HighBP': 'Tekanan Darah Tinggi (0=Tidak, 1=Ya; bukan nilai sistolik/diastolik)',
        'HighChol': 'Kolesterol Tinggi (0=Tidak, 1=Ya; bukan nilai HDL/LDL/trigliserida)',
        'PhysActivity': 'Aktivitas Fisik (30 hari terakhir; 0=Tidak, 1=Ya)',
        'Smoker': 'Riwayat Merokok (0=Tidak, 1=Ya)',
        'Fruits': 'Konsumsi Buah 1x atau lebih per hari (0=Tidak, 1=Ya)',
        'Veggies': 'Konsumsi Sayuran 1x atau lebih per hari (0=Tidak, 1=Ya)'
    }

    st.sidebar.caption(
        "Form ini hanya memakai fitur yang tersedia di dataset. "
        "Lingkar pinggang, riwayat keluarga/diabetes gestasional, hasil lab, "
        "profil lipid lengkap, CGM, citra, dan audio tidak tersedia."
    )

    for col in feature_names:
        min_val = int(X_train[col].min())
        max_val = int(X_train[col].max())
        default_val = int(X_train[col].mean())
        
        # Tampilkan label yang sudah di-mapping
        display_label = label_map.get(col, col)
        
        # Number input untuk semua fitur
        input_data[col] = st.sidebar.number_input(
            display_label, 
            min_value=min_val, 
            max_value=max_val, 
            value=default_val,
            step=1
        )
            
    # Tombol Prediksi
    predict_btn = st.sidebar.button("🔍 Deteksi Risiko Sekarang", type="primary")

    # --- LOGIKA PREDIKSI ---
    if predict_btn:
        request_started = time.perf_counter()
        st.session_state["prediction_run_id"] = (
            st.session_state.get("prediction_run_id", 0) + 1
        )
        request_id = st.session_state["prediction_run_id"]

        # 1. Siapkan data input
        input_df = pd.DataFrame([input_data])
        input_quantized = quantize_bmi(input_df, bmi_quantizer)
        
        # 2. Prediksi Model
        model_started = time.perf_counter()
        prediction = rf_model.predict(input_quantized)[0]
        prediction_proba = rf_model.predict_proba(input_quantized)[0]
        model_elapsed_ms = (time.perf_counter() - model_started) * 1000
        
        label_hasil = {0: "Tidak Diabetes", 1: "Prediabetes", 2: "Diabetes"}
        hasil_teks = label_hasil.get(prediction, "Tidak Diketahui")
        
        # 3. Tampilkan Hasil
        st.header("📊 Hasil Analisis")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 2:
                st.error(f"🚨 **HASIL: {hasil_teks}**")
                st.write(f"Probabilitas Risiko Diabetes: **{prediction_proba[2]*100:.2f}%**")
                st.warning("Disarankan untuk segera berkonsultasi dengan tenaga medis profesional.")
            elif prediction == 1:
                st.warning(f"⚠️ **HASIL: {hasil_teks}**")
                st.write(f"Probabilitas Prediabetes: **{prediction_proba[1]*100:.2f}%**")
                st.info("Tingkatkan pola hidup sehat dan waspadai risiko diabetes di masa depan.")
            else:
                st.success(f"✅ **HASIL: {hasil_teks}**")
                st.write(f"Probabilitas Tidak Diabetes: **{prediction_proba[0]*100:.2f}%**")
                st.info("Tetap jaga pola makan sehat dan rutin berolahraga.")
                
        with col2:
            st.info("""
            **Catatan Penting:**
            Sistem ini adalah alat bantu *screening* awal. 
            Hasil prediksi bukanlah diagnosis medis final.
            """)
            
        # 4. Generate & Tampilkan XAI (LIME)
        st.markdown("---")
        st.subheader("🧠 Penjelasan Prediksi (Explainable AI - LIME)")
        st.write("Grafik di bawah ini menunjukkan fitur klinis mana yang paling mempengaruhi hasil prediksi pasien ini.")
        
        with st.spinner("Menghasilkan penjelasan LIME..."):
            lime_started = time.perf_counter()
            exp = generate_lime_explanation(
                rf_model,
                X_train,
                input_df,
                feature_names,
                prediction,
                bmi_quantizer
            )
            
            explanation = exp.as_list(label=int(prediction))
            explanation_labels = [item[0] for item in explanation]
            explanation_weights = [item[1] for item in explanation]
            colors = [
                '#4472C4' if weight >= 0 else '#ED7D31'
                for weight in explanation_weights
            ]

            fig, ax = plt.subplots(
                figsize=(max(10, len(explanation_labels) * 1.4), 6)
            )
            ax.bar(explanation_labels, explanation_weights, color=colors)
            ax.axhline(0, color='#555555', linewidth=0.8)
            ax.grid(axis='y', linestyle='--', alpha=0.35)
            ax.set_axisbelow(True)
            ax.set_title(f"Faktor yang memengaruhi hasil: {hasil_teks}")
            ax.set_xlabel("Fitur dan nilai pasien")
            ax.set_ylabel("Kontribusi lokal LIME")
            ax.tick_params(axis='x', labelrotation=35)
            plt.setp(ax.get_xticklabels(), ha='right')
            ax.legend(handles=[
                Patch(facecolor='#4472C4', label='Mendukung hasil prediksi'),
                Patch(facecolor='#ED7D31', label='Mengurangi dukungan')
            ])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            lime_elapsed_ms = (time.perf_counter() - lime_started) * 1000

            st.caption(
                "Biru mendukung kelas hasil prediksi, sedangkan oranye "
                "menguranginya. Penjelasan ini khusus untuk data yang dimasukkan."
            )
            
            st.write("Detail Probabilitas Kelas:")
            st.write(f"- Non-Diabetes: {prediction_proba[0]*100:.2f}%")
            st.write(f"- Prediabetes: {prediction_proba[1]*100:.2f}%")
            st.write(f"- Diabetes: {prediction_proba[2]*100:.2f}%")

        server_elapsed_ms = (time.perf_counter() - request_started) * 1000
        st.caption(
            f"Uji prediksi #{request_id} | Waktu model: {model_elapsed_ms:.3f} ms | "
            f"Waktu LIME: {lime_elapsed_ms:.3f} ms | "
            f"Total pemrosesan server: {server_elapsed_ms:.3f} ms"
        )

if __name__ == "__main__":
    main()