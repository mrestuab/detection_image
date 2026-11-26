import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="AI Image Analytics Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Judul Utama
st.title("🔍 AI vs Real Image Analytics Dashboard")
st.markdown("""
**Analisis Karakteristik Visual dan Distribusi Fitur pada Citra Asli dan Citra Hasil Generasi AI**  
*Laporan Internship 1 — Data Analytics & Exploratory Data Analysis (EDA)*
""")
st.divider()

# Load Dataset Fitur yang sudah diekstrak (Otomatis deteksi pemisah koma atau titik-koma dari Excel)
@st.cache_data
def load_data():
    csv_path = Path("outputs/extracted_features/dataset_features_4000.csv")
    if not csv_path.exists():
        csv_path = Path("../outputs/extracted_features/dataset_features_4000.csv")
    
    # engine='python' dan sep=None agar otomatis mendeteksi koma (,) atau titik-koma (;) dari Excel
    return pd.read_csv(csv_path, sep=None, engine='python')

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

# Sidebar Informasi
st.sidebar.header("📌 Informasi Proyek")
st.sidebar.info("""
- **Dataset:** CIFAKE Benchmark
- **Populasi Mentah:** 120.000 Citra (32x32 px)
- **Sampel Terkurasi:** 4.000 Citra (Seimbang)
- **Metode:** Stratified Random Sampling
- **Target Luaran:** Laporan Internship 1 & Artikel Jurnal
""")

# Menu Tab Utama
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Ringkasan Dataset", 
    "📈 Karakteristik Visual & Distribusi", 
    "🧪 Uji Signifikansi Statistik", 
    "🖼️ Live Image Inspector"
])

# ----------------- TAB 1: RINGKASAN DATASET -----------------
with tab1:
    st.subheader("Ringkasan Dataset Terkurasi")
    
    col1, col2, col3, col4 = st.columns(4)
    total_rows = len(df)
    total_real = len(df[df['label'] == 'REAL'])
    total_fake = len(df[df['label'] == 'FAKE'])
    
    col1.metric("Total Sampel", f"{total_rows:,} Citra")
    col2.metric("Citra Asli (REAL)", f"{total_real:,} (50%)")
    col3.metric("Citra AI (FAKE)", f"{total_fake:,} (50%)")
    col4.metric("Resolusi Citra", "32 x 32 px (JPEG)")
    
    st.markdown("### Tabel Fitur Citra Hasil Ekstraksi")
    st.dataframe(df.drop(columns=["filepath"], errors="ignore"), height=350)

# ----------------- TAB 2: KARAKTERISTIK VISUAL -----------------
with tab2:
    st.subheader("Perbandingan Distribusi Fitur Visual")
    
    pilihan_fitur = st.selectbox(
        "Pilih Fitur yang Ingin Dianalisis:",
        [
            "Kejenuhan Warna (saturation_mean)",
            "Tingkat Ketajaman (laplacian_var)",
            "Kontras Tekstur GLCM (glcm_contrast)",
            "Kehalusan Tekstur GLCM (glcm_homogeneity)",
            "Kecerahan Citra (brightness_mean)",
            "Ukuran File (file_size_kb)"
        ]
    )
    
    kolom_map = {
        "Kejenuhan Warna (saturation_mean)": "saturation_mean",
        "Tingkat Ketajaman (laplacian_var)": "laplacian_var",
        "Kontras Tekstur GLCM (glcm_contrast)": "glcm_contrast",
        "Kehalusan Tekstur GLCM (glcm_homogeneity)": "glcm_homogeneity",
        "Kecerahan Citra (brightness_mean)": "brightness_mean",
        "Ukuran File (file_size_kb)": "file_size_kb"
    }
    
    fitur_terpilih = kolom_map[pilihan_fitur]
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown(f"**Boxplot Perbandingan: {fitur_terpilih}**")
        fig_box, ax_box = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df, 
            x="label", 
            y=fitur_terpilih, 
            hue="label", 
            palette={"REAL": "#2ecc71", "FAKE": "#e74c3c"}, 
            width=0.4, 
            legend=False, 
            ax=ax_box
        )
        ax_box.set_xlabel("Kategori Citra")
        st.pyplot(fig_box)
        plt.close(fig_box)
        
    with col_chart2:
        st.markdown(f"**Distribusi Kepadatan (KDE): {fitur_terpilih}**")
        fig_kde, ax_kde = plt.subplots(figsize=(6, 4))
        data_plot = df[df[fitur_terpilih] < 10000] if fitur_terpilih == "laplacian_var" else df
        sns.kdeplot(
            data=data_plot, 
            x=fitur_terpilih, 
            hue="label", 
            fill=True, 
            common_norm=False,
            palette={"REAL": "green", "FAKE": "red"}, 
            alpha=0.4, 
            ax=ax_kde
        )
        st.pyplot(fig_kde)
        plt.close(fig_kde)

# ----------------- TAB 3: UJI STATISTIK -----------------
with tab3:
    st.subheader("Uji Hipotesis Statistik (Mann-Whitney U Test)")
    st.markdown("""
    Pengujian non-parametrik dua sisi (*two-sided*) untuk membuktikan signifikansi perbedaan distribusi 
    antara citra asli dan citra AI pada taraf signifikansi $\\alpha = 0.05$.
    """)
    
    stat_data = [
        {"Fitur": "file_size_kb", "Mean REAL": 0.90, "Mean FAKE": 0.90, "p-value": "4.7112e-01", "Signifikansi": "Tidak Signifikan (p >= 0.05)"},
        {"Fitur": "red_mean", "Mean REAL": 126.00, "Mean FAKE": 115.79, "p-value": "1.8293e-26", "Signifikansi": "Sangat Signifikan (p < 0.001)"},
        {"Fitur": "green_mean", "Mean REAL": 123.26, "Mean FAKE": 113.41, "p-value": "3.9997e-26", "Signifikansi": "Sangat Signifikan (p < 0.001)"},
        {"Fitur": "blue_mean", "Mean REAL": 112.71, "Mean FAKE": 98.82, "p-value": "2.3147e-29", "Signifikansi": "Sangat Signifikan (p < 0.001)"},
        {"Fitur": "brightness_mean", "Mean REAL": 137.05, "Mean FAKE": 129.49, "p-value": "8.0515e-19", "Signifikansi": "Sangat Signifikan (p < 0.001)"},
        {"Fitur": "saturation_mean", "Mean REAL": 72.07, "Mean FAKE": 92.73, "p-value": "6.4200e-43", "Signifikansi": "Sangat Signifikan (p < 0.001)"},
        {"Fitur": "glcm_contrast", "Mean REAL": 539.60, "Mean FAKE": 643.85, "p-value": "3.0174e-08", "Signifikansi": "Sangat Signifikan (p < 0.001)"},
        {"Fitur": "glcm_homogeneity", "Mean REAL": 0.14, "Mean FAKE": 0.18, "p-value": "2.3627e-39", "Signifikansi": "Sangat Signifikan (p < 0.001)"},
        {"Fitur": "laplacian_var", "Mean REAL": 2500.09, "Mean FAKE": 3804.07, "p-value": "9.1581e-103", "Signifikansi": "Sangat Signifikan (p < 0.001)"}
    ]
    st.table(pd.DataFrame(stat_data))
    
    st.markdown("### Matriks Korelasi Fitur (Spearman)")
    corr_img_path = Path("outputs/visualizations/04_feature_correlation_heatmap.png")
    if not corr_img_path.exists():
        corr_img_path = Path("../outputs/visualizations/04_feature_correlation_heatmap.png")
    if corr_img_path.exists():
        st.image(str(corr_img_path), caption="Matriks Korelasi Fitur Visual")

# ----------------- TAB 4: LIVE INSPECTOR -----------------
with tab4:
    st.subheader("🖼️ Live Image Feature Inspector")
    st.markdown("Unggah citra sembarang untuk menghitung fitur warna, tekstur, dan ketajamannya secara *real-time*!")
    
    uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        user_img = Image.open(uploaded_file).convert("RGB")
        col_u1, col_u2 = st.columns([1, 2])
        
        with col_u1:
            st.image(user_img, caption="Citra yang Diunggah", width=220)
            
        with col_u2:
            img_arr = np.array(user_img)
            hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            
            sat_val = hsv[:, :, 1].mean()
            bright_val = hsv[:, :, 2].mean()
            lap_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast_val = graycoprops(glcm, 'contrast')[0, 0]
            homog_val = graycoprops(glcm, 'homogeneity')[0, 0]
            
            st.markdown("#### Karakteristik Citra Terdeteksi:")
            m1, m2, m3 = st.columns(3)
            m1.metric("Kejenuhan (Sat)", f"{sat_val:.2f}", help="Benchmark REAL: ~72.1 | FAKE: ~92.7")
            m2.metric("Ketajaman (Laplacian)", f"{lap_val:.2f}", help="Benchmark REAL: ~2500 | FAKE: ~3804")
            m3.metric("Kecerahan (Val)", f"{bright_val:.2f}", help="Benchmark REAL: ~137.1 | FAKE: ~129.5")
            
            m4, m5, _ = st.columns(3)
            m4.metric("GLCM Kontras", f"{contrast_val:.2f}", help="Benchmark REAL: ~539.6 | FAKE: ~643.8")
            m5.metric("GLCM Homogenitas", f"{homog_val:.4f}", help="Benchmark REAL: ~0.144 | FAKE: ~0.184")
            
            st.divider()
            
            # Logika Komparatif Berdasarkan Temuan Benchmark Statistik
            # Skor kecenderungan: makin banyak fitur yang condong ke REAL, makin kuat status REAL
            poin_real = 0
            poin_fake = 0
            alasan = []
            
            # Evaluasi Saturasi (Threshold tengah: ~82.4)
            if sat_val < 82.4:
                poin_real += 1
                alasan.append(f"Kejenuhan warna alami ({sat_val:.1f} < 82.4)")
            else:
                poin_fake += 1
                alasan.append(f"Kejenuhan warna tinggi/pekat ({sat_val:.1f} >= 82.4)")
                
            # Evaluasi Ketajaman Laplacian (Threshold tengah: ~3150)
            if lap_val < 3150:
                poin_real += 1
                alasan.append(f"Ketajaman tepi wajar/alami ({lap_val:.0f} < 3150)")
            else:
                poin_fake += 1
                alasan.append(f"Tingkat ketajaman/artefak frekuensi tinggi ({lap_val:.0f} >= 3150)")
                
            # Evaluasi Kecerahan (Threshold tengah: ~133.3)
            if bright_val > 133.3:
                poin_real += 1
                alasan.append(f"Pencahayaan dinamis khas citra asli ({bright_val:.1f} > 133.3)")
            else:
                poin_fake += 1
                alasan.append(f"Pencahayaan terkonsentrasi khas citra AI ({bright_val:.1f} <= 133.3)")
                
            # Tampilkan Hasil Evaluasi Dinamis
            st.markdown("#### 🎯 Hasil Evaluasi Berdasarkan Distribusi Fitur:")
            if poin_real >= 2:
                st.success(f"""
                ### ✅ Karakteristik Lebih Mendekati: CITRA ASLI (REAL)
                **Analisis:**
                - {alasan[0]}
                - {alasan[1]}
                - {alasan[2]}
                
                *Kesimpulan: Pola piksel citra ini tidak memperlihatkan artefak difusi maupun saturasi berlebih yang biasa ditemukan pada citra buatan AI.*
                """)
            else:
                st.warning(f"""
                ### ⚠️ Karakteristik Lebih Mendekati: CITRA GENERASI AI (FAKE)
                **Analisis:**
                - {alasan[0]}
                - {alasan[1]}
                - {alasan[2]}
                
                *Kesimpulan: Citra ini menunjukkan karakteristik yang sejalan dengan artefak generator AI (kejenuhan tinggi dan/atau variansi laplacian yang agresif).*
                """)
