"""
ILASPP Hybrid Project - Streamlit Dashboard (Deploy Version)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="ILASPP - Hybrid Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DATA LOADING - SEMUA FILE DALAM SATU FOLDER
# ============================================
@st.cache_data
def load_original_data():
    """Load original land value data"""
    try:
        df = pd.read_csv("land_values_clean.csv")
        return df
    except FileNotFoundError:
        st.error("File land_values_clean.csv not found in deployment directory")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_gwr_results():
    """Load GWR analysis results"""
    try:
        df = pd.read_csv("gwr_coefficients.csv")
        return df
    except FileNotFoundError:
        st.warning("GWR results file not found")
        return None
    except Exception as e:
        st.error(f"Error loading GWR: {e}")
        return None

@st.cache_data
def load_kriging_results():
    """Load Kriging analysis results"""
    try:
        df = pd.read_csv("kriging_predictions.csv")
        return df
    except FileNotFoundError:
        st.warning("Kriging results file not found")
        return None
    except Exception as e:
        st.error(f"Error loading Kriging: {e}")
        return None

# Load all data
df = load_original_data()
gwr_df = load_gwr_results()
kriging_df = load_kriging_results()

# ============================================
# SIDEBAR MENU
# ============================================
with st.sidebar:
    st.markdown("## 🗺️ ILASPP")
    st.markdown("*Sistem Informasi Nilai Tanah*")
    st.markdown("---")
    
    page = st.radio(
        "📌 **Menu Navigasi**",
        [
            "📊 Dashboard Utama",
            "🗺️ GWR Results",
            "🌐 Kriging Results",
            "📈 Perbandingan & Interpretasi"
        ]
    )
    
    st.markdown("---")
    st.caption("Hybrid R + Python | GWR + Kriging")
    st.caption("ATR/BPN - ILASPP Project")
    st.caption(f"Data: {len(df) if df is not None else 0} records")

# ============================================
# PAGE 1: DASHBOARD UTAMA
# ============================================
if page == "📊 Dashboard Utama":
    st.title("🗺️ ILASPP - Sistem Informasi Nilai Tanah")
    st.markdown("*Hybrid R + Python | GWR + Kriging | Streamlit Dashboard*")
    st.markdown("---")
    
    if df is not None and len(df) > 0:
        st.success(f"✅ Loaded {len(df)} records")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏘️ Total Properti", len(df))
        with col2:
            st.metric("📊 Rata-rata Nilai", f"{df['land_value'].mean():.1f} juta/m²")
        with col3:
            st.metric("📈 Nilai Tertinggi", f"{df['land_value'].max():.1f} juta/m²")
        with col4:
            st.metric("📉 Nilai Terendah", f"{df['land_value'].min():.1f} juta/m²")
        
        # Map
        st.subheader("🗺️ Peta Distribusi Nilai Tanah")
        fig = px.scatter_mapbox(
            df, lat='latitude', lon='longitude',
            color='land_value', size=[8] * len(df),
            color_continuous_scale='Viridis',
            hover_data=['district', 'zone_type', 'land_value'],
            zoom=10, height=500,
            title="Distribusi Spasial Nilai Tanah"
        )
        fig.update_layout(mapbox_style='carto-positron')
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart per district
        st.subheader("📊 Rata-rata Nilai Tanah per Kecamatan")
        district_stats = df.groupby('district')['land_value'].agg(['mean', 'count']).reset_index()
        district_stats.columns = ['district', 'avg_value', 'count']
        district_stats = district_stats.sort_values('avg_value', ascending=False)
        
        fig2 = px.bar(district_stats, x='district', y='avg_value', 
                      color='avg_value', color_continuous_scale='Viridis',
                      title='Rata-rata Nilai Tanah per Kecamatan',
                      labels={'avg_value': 'Nilai (juta/m²)', 'district': 'Kecamatan'})
        st.plotly_chart(fig2, use_container_width=True)
        
        # Show raw data
        with st.expander("📋 Lihat Data Mentah"):
            st.dataframe(df)
    else:
        st.error("⚠️ Data tidak ditemukan. Pastikan file CSV tersedia di folder deploy.")

# ============================================
# PAGE 2: GWR RESULTS
# ============================================
elif page == "🗺️ GWR Results":
    st.title("📈 Geographically Weighted Regression (GWR)")
    st.markdown("*Analisis Variasi Spasial Pengaruh Variabel terhadap Nilai Tanah*")
    st.markdown("---")
    
    if gwr_df is not None and len(gwr_df) > 0:
        st.success(f"✅ Loaded {len(gwr_df)} GWR results")
        
        # Key Metrics
        st.subheader("📊 Ringkasan Statistik GWR")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Mean Local R²", f"{gwr_df['Local_R2'].mean():.4f}")
        with col2:
            st.metric("📉 Min Local R²", f"{gwr_df['Local_R2'].min():.4f}")
        with col3:
            st.metric("📊 Max Local R²", f"{gwr_df['Local_R2'].max():.4f}")
        with col4:
            st.metric("📐 SD Local R²", f"{gwr_df['Local_R2'].std():.4f}")
        
        # Coefficient Distribution
        st.subheader("📊 Distribusi Koefisien per Variabel")
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(gwr_df, x='distance_center', 
                                title='Koefisien Distance Center (Jarak ke Pusat)',
                                labels={'distance_center': 'Koefisien'},
                                color_discrete_sequence=['blue'],
                                nbins=30)
            fig1.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.histogram(gwr_df, x='road_width',
                                title='Koefisien Road Width (Lebar Jalan)',
                                labels={'road_width': 'Koefisien'},
                                color_discrete_sequence=['green'],
                                nbins=30)
            fig2.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Coefficient Summaries
        st.subheader("📋 Ringkasan Koefisien")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distance Center (Jarak ke Pusat Kota)**")
            st.markdown(f"""
            - Mean: **{gwr_df['distance_center'].mean():.4f}**
            - Min: {gwr_df['distance_center'].min():.4f}
            - Max: {gwr_df['distance_center'].max():.4f}
            - % Negative: **{(gwr_df['distance_center'] < 0).mean() * 100:.1f}%**
            """)
        
        with col2:
            st.markdown("**Road Width (Lebar Jalan)**")
            st.markdown(f"""
            - Mean: **{gwr_df['road_width'].mean():.4f}**
            - Min: {gwr_df['road_width'].min():.4f}
            - Max: {gwr_df['road_width'].max():.4f}
            - % Positive: **{(gwr_df['road_width'] > 0).mean() * 100:.1f}%**
            """)
        
        # Dominant Variable
        st.subheader("🎯 Variabel Dominan per Lokasi")
        
        coeff_abs = abs(gwr_df[['distance_center', 'road_width']])
        dominant = coeff_abs.idxmax(axis=1)
        dom_counts = dominant.value_counts()
        
        fig3 = px.pie(values=dom_counts.values, names=dom_counts.index,
                      title='Proporsi Variabel Dominan',
                      color_discrete_sequence=['blue', 'green'])
        st.plotly_chart(fig3, use_container_width=True)
        
        # Raw data
        with st.expander("📋 Lihat Data GWR Lengkap"):
            st.dataframe(gwr_df)
            
    else:
        st.warning("⚠️ GWR results not found. Pastikan file gwr_coefficients.csv tersedia.")

# ============================================
# PAGE 3: KRIGING RESULTS
# ============================================
elif page == "🌐 Kriging Results":
    st.title("🌐 Kriging Interpolation")
    st.markdown("*Prediksi Nilai Tanah di Seluruh Area Berdasarkan Data Titik*")
    st.markdown("---")
    
    if kriging_df is not None and len(kriging_df) > 0:
        st.success(f"✅ Loaded {len(kriging_df)} Kriging predictions")
        
        # Key Metrics
        st.subheader("📊 Ringkasan Statistik Kriging")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Min Prediction", f"{kriging_df['var1.pred'].min():.2f} juta/m²")
        with col2:
            st.metric("📊 Max Prediction", f"{kriging_df['var1.pred'].max():.2f} juta/m²")
        with col3:
            st.metric("📉 Mean Prediction", f"{kriging_df['var1.pred'].mean():.2f} juta/m²")
        
        # Prediction distribution
        st.subheader("📊 Distribusi Nilai Prediksi")
        fig = px.histogram(kriging_df, x='var1.pred',
                          title='Distribusi Nilai Tanah Hasil Kriging',
                          labels={'var1.pred': 'Predicted Value (juta/m²)'},
                          nbins=50,
                          color_discrete_sequence=['purple'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        with st.expander("📋 Lihat Data Kriging Lengkap"):
            st.dataframe(kriging_df.head(500))
            
    else:
        st.warning("⚠️ Kriging results not found. Pastikan file kriging_predictions.csv tersedia.")

# ============================================
# PAGE 4: PERBANDINGAN & INTERPRETASI
# ============================================
elif page == "📈 Perbandingan & Interpretasi":
    st.title("📈 Perbandingan GWR vs Kriging")
    st.markdown("*Integrasi Kedua Metode untuk Analisis Spasial yang Komprehensif*")
    st.markdown("---")
    
    # Overview table
    st.subheader("🔬 Ringkasan Metode")
    
    overview_df = pd.DataFrame({
        "Metode": ["GWR", "Kriging"],
        "Tujuan": [
            "Mengetahui penyebab variasi nilai tanah",
            "Memprediksi nilai di seluruh area"
        ],
        "Output Utama": [
            "Koefisien per lokasi, Local R²",
            "Peta prediksi kontinu, Uncertainty map"
        ],
        "Kelebihan": [
            "Menjelaskan variasi spasial, Interpretable",
            "Mengisi area kosong, Quantify uncertainty"
        ]
    })
    st.dataframe(overview_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # GWR Interpretation
    st.subheader("📊 Interpretasi GWR")
    
    if gwr_df is not None and len(gwr_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 **Distance Center**")
            mean_dc = gwr_df['distance_center'].mean()
            pct_neg = (gwr_df['distance_center'] < 0).mean() * 100
            st.markdown(f"""
            - Rata-rata koefisien: **{mean_dc:.4f}**
            - {pct_neg:.0f}% lokasi koefisien **negatif**
            
            **Interpolasi:**  
            Semakin jauh dari pusat kota, nilai tanah **menurun**.
            Setiap +1 km jarak, nilai turun **{abs(mean_dc):.2f} juta/m²**.
            """)
        
        with col2:
            st.markdown("#### 🛣️ **Road Width**")
            mean_rw = gwr_df['road_width'].mean()
            pct_pos = (gwr_df['road_width'] > 0).mean() * 100
            st.markdown(f"""
            - Rata-rata koefisien: **{mean_rw:.4f}**
            - {pct_pos:.0f}% lokasi koefisien **positif**
            
            **Interpolasi:**  
            Semakin lebar jalan, nilai tanah **meningkat**.
            Setiap +1 m lebar jalan, nilai naik **{mean_rw:.2f} juta/m²**.
            """)
        
        st.markdown("#### 📈 **Kualitas Model**")
        st.markdown(f"""
        - Mean Local R²: **{gwr_df['Local_R2'].mean():.4f}**
        - Range: {gwr_df['Local_R2'].min():.4f} - {gwr_df['Local_R2'].max():.4f}
        """)
    else:
        st.info("GWR results not loaded")
    
    st.markdown("---")
    
    # Kriging Interpretation
    st.subheader("🌐 Interpretasi Kriging")
    
    if kriging_df is not None and len(kriging_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 **Statistik Prediksi**")
            st.markdown(f"""
            - Range: **{kriging_df['var1.pred'].min():.2f} - {kriging_df['var1.pred'].max():.2f}** juta/m²
            - Mean: **{kriging_df['var1.pred'].mean():.2f}** juta/m²
            """)
        
        with col2:
            st.markdown("#### ⚠️ **Ketidakpastian**")
            st.markdown(f"""
            - Mean variance: **{kriging_df['var1.var'].mean():.2f}**
            - Std Dev: **{np.sqrt(kriging_df['var1.var'].mean()):.2f}**
            
            Area dengan variance tinggi perlu survei tambahan.
            """)
    else:
        st.info("Kriging results not loaded")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Rekomendasi Kebijakan")
    
    st.markdown("""
    ### 📌 Berdasarkan GWR:
    1. Prioritas pengembangan di area dengan akses jalan lebar
    2. Insentif untuk pengembangan di area pinggiran
    
    ### 📌 Berdasarkan Kriging:
    1. Fokus survei di area dengan variance tinggi
    2. Identifikasi hotspot untuk investasi properti
    
    ### 📌 Rekomendasi Lanjutan:
    1. Integrasikan data raster (elevasi, kepadatan)
    2. Perbarui data berkala
    3. Kembangkan dashboard real-time
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🗺️ ILASPP Hybrid Portfolio | R + Python | GWR + Kriging | ATR/BPN")
