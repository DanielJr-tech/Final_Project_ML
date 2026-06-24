import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Kekerasan Sukabumi",
    layout="wide"
)

st.title("📊 Analisis & Proyeksi Kasus Kekerasan — Kota & Kabupaten Sukabumi")
st.caption("Migrasi dari notebook Colab ke Streamlit")

KATEGORI_KEKERASAN = ['FISIK', 'PSIKIS', 'SEKSUAL', 'EKSPLOITASI',
                       'TRAFFICKING', 'PENELANTARAN', 'LAINNYA']

WARNA_7_KASUS = {
    'FISIK': '#ef5350', 'PSIKIS': '#4fc3f7', 'SEKSUAL': '#f5a623',
    'EKSPLOITASI': '#ab47bc', 'TRAFFICKING': '#26a69a',
    'PENELANTARAN': '#66bb6a', 'LAINNYA': '#888888'
}

# =========================================================
# CELL 1 (versi Streamlit): UPLOAD FILE
# =========================================================
st.header("1️⃣ Upload Dataset")

col1, col2 = st.columns(2)
with col1:
    file_kota = st.file_uploader("Upload dataset_kota_sukabumi.csv", type="csv", key="kota")
with col2:
    file_kab = st.file_uploader("Upload dataset_kabupaten_sukabumi.csv", type="csv", key="kab")

if not file_kota or not file_kab:
    st.info("⬆️ Silakan upload kedua file CSV untuk melanjutkan analisis.")
    st.stop()

df_kota = pd.read_csv(file_kota)
df_kab = pd.read_csv(file_kab)
st.success("Kedua dataset berhasil dimuat!")

with st.expander("Lihat pratinjau data mentah"):
    st.write("**Data Kota Sukabumi**")
    st.dataframe(df_kota.head())
    st.write("**Data Kabupaten Sukabumi**")
    st.dataframe(df_kab.head())

# =========================================================
# CELL 2: AGREGASI DATA
# =========================================================
@st.cache_data
def agregasi_data(df_kota, df_kab, kategori):
    daftar_tahun = np.arange(2017, 2026)

    kota_pivot = df_kota.groupby(['tahun', 'bentuk_kekerasan'])['jumlah_korban'].sum().unstack(fill_value=0)
    kota_pivot = kota_pivot.reindex(index=daftar_tahun, columns=kategori, fill_value=0)

    kab_pivot = df_kab.groupby(['tahun', 'bentuk_kekerasan'])['jumlah_korban'].sum().unstack(fill_value=0)
    kab_pivot = kab_pivot.reindex(index=daftar_tahun, columns=kategori, fill_value=0)

    total_kota_tahunan = kota_pivot.sum(axis=1).values
    total_kab_tahunan = kab_pivot.sum(axis=1).values

    return daftar_tahun, kota_pivot, kab_pivot, total_kota_tahunan, total_kab_tahunan

try:
    daftar_tahun, kota_pivot, kab_pivot, total_kota_tahunan, total_kab_tahunan = agregasi_data(
        df_kota, df_kab, KATEGORI_KEKERASAN
    )
except KeyError as e:
    st.error(f"Kolom tidak ditemukan di CSV: {e}. Pastikan ada kolom 'tahun', 'bentuk_kekerasan', 'jumlah_korban'.")
    st.stop()

st.header("2️⃣ Data Teragregasi")
c1, c2 = st.columns(2)
with c1:
    st.write("**Pivot Kota Sukabumi**")
    st.dataframe(kota_pivot)
with c2:
    st.write("**Pivot Kabupaten Sukabumi**")
    st.dataframe(kab_pivot)

# =========================================================
# CELL 3: MODEL REGRESI LINIER (TOTAL TAHUNAN)
# =========================================================
st.header("3️⃣ Model Regresi Linear (Total Tahunan)")

tahun_prediksi = np.arange(2026, 2035)
X_historis = daftar_tahun.reshape(-1, 1)
X_prediksi = tahun_prediksi.reshape(-1, 1)

model_kota = LinearRegression().fit(X_historis, total_kota_tahunan)
fitted_kota = model_kota.predict(X_historis)
prediksi_kota = model_kota.predict(X_prediksi)
rmse_kota = np.sqrt(mean_squared_error(total_kota_tahunan, fitted_kota))
r2_kota = r2_score(total_kota_tahunan, fitted_kota)

model_kab = LinearRegression().fit(X_historis, total_kab_tahunan)
fitted_kab = model_kab.predict(X_historis)
prediksi_kab = model_kab.predict(X_prediksi)
rmse_kab = np.sqrt(mean_squared_error(total_kab_tahunan, fitted_kab))
r2_kab = r2_score(total_kab_tahunan, fitted_kab)

m1, m2 = st.columns(2)
with m1:
    st.metric("Slope Kota Sukabumi", f"{model_kota.coef_[0]:+.2f} / tahun")
    st.metric("R² Kota Sukabumi", f"{r2_kota:.4f}")
with m2:
    st.metric("Slope Kabupaten Sukabumi", f"{model_kab.coef_[0]:+.2f} / tahun")
    st.metric("R² Kabupaten Sukabumi", f"{r2_kab:.4f}")

# =========================================================
# CELL 4: KESIMPULAN STATISTIK DESKRIPTIF
# =========================================================
st.header("4️⃣ Kesimpulan Analisis Historis (2017–2025)")

idx_max_kota = np.argmax(total_kota_tahunan)
tahun_max_kota = daftar_tahun[idx_max_kota]
nilai_max_kota = total_kota_tahunan[idx_max_kota]
kategori_dominan_kota = kota_pivot.sum().idxmax()
jumlah_dominan_kota = kota_pivot.sum().max()

idx_max_kab = np.argmax(total_kab_tahunan)
tahun_max_kab = daftar_tahun[idx_max_kab]
nilai_max_kab = total_kab_tahunan[idx_max_kab]
kategori_dominan_kab = kab_pivot.sum().idxmax()
jumlah_dominan_kab = kab_pivot.sum().max()

k1, k2 = st.columns(2)
with k1:
    st.subheader("Kota Sukabumi")
    st.write(f"- Total korban tertinggi: **{int(nilai_max_kota)}** pada tahun **{tahun_max_kota}**")
    st.write(f"- Kategori paling dominan: **{kategori_dominan_kota}** ({int(jumlah_dominan_kota)} korban kumulatif)")
    st.write(f"- Tren: **{'NAIK ⬆️' if model_kota.coef_[0] > 0 else 'TURUN ⬇️'}** ({model_kota.coef_[0]:+.2f} kasus/tahun)")
with k2:
    st.subheader("Kabupaten Sukabumi")
    st.write(f"- Total korban tertinggi: **{int(nilai_max_kab)}** pada tahun **{tahun_max_kab}**")
    st.write(f"- Kategori paling dominan: **{kategori_dominan_kab}** ({int(jumlah_dominan_kab)} korban kumulatif)")
    st.write(f"- Tren: **{'NAIK ⬆️' if model_kab.coef_[0] > 0 else 'TURUN ⬇️'}** ({model_kab.coef_[0]:+.2f} kasus/tahun)")

# =========================================================
# CELL 5: VISUALISASI 2x2 (REGRESI + PERKEMBANGAN 7 KASUS)
# =========================================================
st.header("5️⃣ Visualisasi Perbandingan Wilayah")

plt.rcParams['text.color'] = '#e8eaf0'
plt.rcParams['axes.labelcolor'] = '#e8eaf0'
plt.rcParams['xtick.color'] = '#a9afc2'
plt.rcParams['ytick.color'] = '#a9afc2'
plt.rcParams['axes.edgecolor'] = '#252a38'

fig1, axes = plt.subplots(2, 2, figsize=(18, 16))
fig1.patch.set_facecolor('#0d0f14')
tahun_gabung = np.concatenate([daftar_tahun, tahun_prediksi])

ax = axes[0, 0]
ax.set_facecolor('#13161e')
ax.grid(True, alpha=0.15, color='#7a8099')
ax.scatter(daftar_tahun, total_kota_tahunan, color='#4fc3f7', s=80, label='Data Aktual CSV', zorder=5)
ax.plot(daftar_tahun, fitted_kota, color='#ef5350', linewidth=2, label='Garis Regresi Historis')
ax.plot(tahun_prediksi, prediksi_kota, color='#f5a623', linestyle='--', linewidth=2, label='Proyeksi (2026-2034)')
ax.axvline(x=2025.5, color='#252a38', linestyle=':')
ax.set_title(f"Kota Sukabumi - Tren Regresi (R² = {r2_kota:.3f})", fontweight='bold', pad=10)
ax.set_xticks(tahun_gabung)
ax.set_xticklabels(tahun_gabung, rotation=45, fontsize=8)
ax.legend(facecolor='#1a1e28', edgecolor='#252a38')

ax = axes[0, 1]
ax.set_facecolor('#13161e')
ax.grid(True, alpha=0.15, color='#7a8099')
ax.scatter(daftar_tahun, total_kab_tahunan, color='#26a69a', s=80, label='Data Aktual CSV', zorder=5)
ax.plot(daftar_tahun, fitted_kab, color='#ef5350', linewidth=2, label='Garis Regresi Historis')
ax.plot(tahun_prediksi, prediksi_kab, color='#f5a623', linestyle='--', linewidth=2, label='Proyeksi (2026-2034)')
ax.axvline(x=2025.5, color='#252a38', linestyle=':')
ax.set_title(f"Kabupaten Sukabumi - Tren Regresi (R² = {r2_kab:.3f})", fontweight='bold', pad=10)
ax.set_xticks(tahun_gabung)
ax.set_xticklabels(tahun_gabung, rotation=45, fontsize=8)
ax.legend(facecolor='#1a1e28', edgecolor='#252a38')

ax = axes[1, 0]
ax.set_facecolor('#13161e')
ax.grid(True, alpha=0.15, color='#7a8099')
for kolom in kota_pivot.columns:
    ax.plot(daftar_tahun, kota_pivot[kolom].values, marker='o', color=WARNA_7_KASUS[kolom], label=kolom)
ax.set_title("Kota Sukabumi - Perkembangan 7 Kategori Kasus", fontweight='bold', pad=10)
ax.set_xticks(daftar_tahun)
ax.legend(facecolor='#1a1e28', edgecolor='#252a38', fontsize=8)

ax = axes[1, 1]
ax.set_facecolor('#13161e')
ax.grid(True, alpha=0.15, color='#7a8099')
for kolom in kab_pivot.columns:
    ax.plot(daftar_tahun, kab_pivot[kolom].values, marker='o', color=WARNA_7_KASUS[kolom], label=kolom)
ax.set_title("Kabupaten Sukabumi - Perkembangan 7 Kategori Kasus", fontweight='bold', pad=10)
ax.set_xticks(daftar_tahun)
ax.legend(facecolor='#1a1e28', edgecolor='#252a38', fontsize=8)

plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# =========================================================
# CELL 6: PROYEKSI PER KATEGORI (2026-2036)
# =========================================================
st.header("6️⃣ Proyeksi Tren per Kategori Kasus (2026–2036)")

tahun_prediksi_baru = np.arange(2026, 2037)
X_prediksi_baru = tahun_prediksi_baru.reshape(-1, 1)

def hitung_proyeksi_kategori(pivot_df):
    hasil_pred = {}
    tabel = []
    for kasus in KATEGORI_KEKERASAN:
        y_hist = pivot_df[kasus].values
        if np.sum(y_hist) == 0:
            pred = np.zeros(len(tahun_prediksi_baru))
            status = "TETAP (0 Kasus)"
        else:
            model = LinearRegression().fit(X_historis, y_hist)
            pred = np.maximum(0, model.predict(X_prediksi_baru))
            slope = model.coef_[0]
            if slope > 0.1:
                status = "↑ NAIK"
            elif slope < -0.1:
                status = "↓ TURUN"
            else:
                status = "→ TETAP"
        hasil_pred[kasus] = pred
        tabel.append([kasus, status, round(pred[0], 1), round(pred[-1], 1)])
    return hasil_pred, pd.DataFrame(tabel, columns=['Bentuk Kekerasan', 'Prediksi Tren', 'Awal (2026)', 'Akhir (2036)'])

prediksi_per_kasus_kota, df_tren_kota = hitung_proyeksi_kategori(kota_pivot)
prediksi_per_kasus_kab, df_tren_kab = hitung_proyeksi_kategori(kab_pivot)

t1, t2 = st.columns(2)
with t1:
    st.write("**Kota Sukabumi**")
    st.dataframe(df_tren_kota, hide_index=True)
with t2:
    st.write("**Kabupaten Sukabumi**")
    st.dataframe(df_tren_kab, hide_index=True)

# =========================================================
# CELL 7: VISUALISASI PROYEKSI PER KATEGORI
# =========================================================
st.header("7️⃣ Grafik Proyeksi 7 Kategori Kasus")

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
fig2.patch.set_facecolor('#0d0f14')

ax1.set_facecolor('#13161e')
ax1.grid(True, alpha=0.1, color='#7a8099')
for kasus in KATEGORI_KEKERASAN:
    warna = WARNA_7_KASUS[kasus]
    ax1.plot(daftar_tahun, kota_pivot[kasus].values, marker='o', color=warna, label=kasus, linewidth=1.8)
    ax1.plot(tahun_prediksi_baru, prediksi_per_kasus_kota[kasus], linestyle='--', color=warna, alpha=0.8)
ax1.axvline(x=2025.5, color='#f5a623', linestyle=':', alpha=0.6)
ax1.text(2025.7, ax1.get_ylim()[1] * 0.9, 'Batas Prediksi (2026) →', color='#f5a623', fontsize=9)
ax1.set_title("Proyeksi Tren 7 Jenis Kasus Kekerasan — KOTA SUKABUMI (S.d 2036)", fontweight='bold', pad=12)
ax1.set_ylabel("Jumlah Korban")
ax1.set_xticks(np.concatenate([daftar_tahun, tahun_prediksi_baru]))
ax1.set_xticklabels(np.concatenate([daftar_tahun, tahun_prediksi_baru]), rotation=45, fontsize=8)
ax1.legend(facecolor='#1a1e28', edgecolor='#252a38', loc='upper left')

ax2.set_facecolor('#13161e')
ax2.grid(True, alpha=0.1, color='#7a8099')
for kasus in KATEGORI_KEKERASAN:
    warna = WARNA_7_KASUS[kasus]
    ax2.plot(daftar_tahun, kab_pivot[kasus].values, marker='o', color=warna, label=kasus, linewidth=1.8)
    ax2.plot(tahun_prediksi_baru, prediksi_per_kasus_kab[kasus], linestyle='--', color=warna, alpha=0.8)
ax2.axvline(x=2025.5, color='#f5a623', linestyle=':', alpha=0.6)
ax2.text(2025.7, ax2.get_ylim()[1] * 0.9, 'Batas Prediksi (2026) →', color='#f5a623', fontsize=9)
ax2.set_title("Proyeksi Tren 7 Jenis Kasus Kekerasan — KABUPATEN SUKABUMI (S.d 2036)", fontweight='bold', pad=12)
ax2.set_ylabel("Jumlah Korban")
ax2.set_xlabel("Tahun")
ax2.set_xticks(np.concatenate([daftar_tahun, tahun_prediksi_baru]))
ax2.set_xticklabels(np.concatenate([daftar_tahun, tahun_prediksi_baru]), rotation=45, fontsize=8)
ax2.legend(facecolor='#1a1e28', edgecolor='#252a38', loc='upper left')

plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.success("Analisis selesai ✅")
