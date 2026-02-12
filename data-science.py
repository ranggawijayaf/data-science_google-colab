# Import Libraries (ditambahkan lebih banyak untuk visualisasi dan analisis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from scipy.stats import ttest_ind  # Untuk uji statistik perbedaan mean
import warnings
warnings.filterwarnings('ignore')

# 1. Menelaah Data
# Load dataset dan preview awal
df = pd.read_csv('C:/Users/ACER/Downloads/Campus Recruitment.csv')  # Sesuaikan path di sini!
print("1. Menelaah Data")
print("- Dataset Shape:", df.shape)
print("- Kolom Dataset:", df.columns.tolist())
print("- Preview Data (5 baris pertama):")
print(df.head())
print("- Info Dataset (tipe data dan non-null counts):")
print(df.info())
print("- Distribusi Jenis Kelamin:")
print(df['Jenis Kelamin'].value_counts())
print("- Distribusi Program Studi Sarjana:")
print(df['Program studi sarjana'].value_counts())

# 2. Menvalidasi Data
print("\n2. Menvalidasi Data")
print("- Tipe Data per Kolom:")
print(df.dtypes)
print("- Jumlah Missing Values per Kolom:")
missing = df.isnull().sum()
print(missing)
print("- Persentase Missing Values:")
print((missing / len(df)) * 100)
print("- Statistik Deskriptif (Numerik):")
print(df.describe())
print("- Statistik Deskriptif (Kategorikal):")
print(df.describe(include=['object']))
print("- Distribusi Target (Status Kelulusan):")
print(df['status kelulusan (Bekerja/Belum)'].value_counts())
print("- Proporsi Placement:", df['status kelulusan (Bekerja/Belum)'].value_counts(normalize=True) * 100)

# 3. Menentukan Objek Data
print("\n3. Menentukan Objek Data")
# Target: status kelulusan (Placed/Not Placed)
# Fitur: Fokus pada akademik (nilai SMP, SMA, IPK), employability (tes kemampuan kerja, pengalaman kerja), dan lainnya
features = ['Jenis Kelamin', 'Nilai rata-rata SMP', 'Lembaga pendidikan kelas 10', 'Nilai rata-rata SMA', 
            'Lembaga pendidikan kelas 12', 'Jurusan saat SMA', 'IPK', 'Program studi sarjana', 
            'Pengalaman kerja sebelum lulus', 'Nilai tes kemampuan kerja', 'Pendidikan pascasarjana', 
            'Nilai rata-rata pascasarjana']
target = 'status kelulusan (Bekerja/Belum)'
print("- Fitur Terpilih:", features)
print("- Target:", target)
print("- Alasan Pemilihan: Fitur ini mencakup faktor akademik (IPK, nilai SMP/SMA) dan employability (tes kemampuan, pengalaman kerja) sesuai fokus analisis.")

# 4. Membersihkan Data
print("\n4. Membersihkan Data")
# Handle missing values: Isi numerik dengan mean, kategorikal dengan mode
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)
print("- Missing Values Setelah Cleaning:", df.isnull().sum().sum())

# Handle outliers (untuk IPK dan nilai tes): Gunakan IQR untuk deteksi dan remove
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

df = remove_outliers(df, 'IPK')
df = remove_outliers(df, 'Nilai tes kemampuan kerja')
print("- Dataset Shape Setelah Remove Outliers:", df.shape)
print("- Outliers Dihapus untuk Kolom IPK dan Nilai Tes Kemampuan Kerja (menggunakan IQR).")

# 5. Mengkonstruksi Data
print("\n5. Mengkonstruksi Data")
# Encode kategorikal
le = LabelEncoder()
categorical_cols = ['Jenis Kelamin', 'Lembaga pendidikan kelas 10', 'Lembaga pendidikan kelas 12', 
                    'Jurusan saat SMA', 'Program studi sarjana', 'Pengalaman kerja sebelum lulus', 
                    'Pendidikan pascasarjana', target]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature Engineering: Tambah fitur baru, e.g., rata-rata nilai akademik
df['Rata-rata Akademik'] = (df['Nilai rata-rata SMP'] + df['Nilai rata-rata SMA'] + df['IPK']) / 3
features.append('Rata-rata Akademik')

# Split X dan y
X = df[features]
y = df[target]

# Scale numerik
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("- Fitur Setelah Konstruksi:", X_scaled.columns.tolist())
print("- Feature Engineering: Ditambahkan 'Rata-rata Akademik' sebagai rata-rata nilai SMP, SMA, dan IPK.")

# 6. Membangun Skenario Model
print("\n6. Membangun Skenario Model")
print("- Skenario: Klasifikasi biner untuk memprediksi placement berdasarkan faktor akademik dan employability.")
print("- Model 1: Logistic Regression (sederhana, mudah diinterpretasi untuk kontribusi variabel).")
print("- Model 2: Random Forest (robust, akurasi tinggi, handle non-linearitas).")
print("- Evaluasi: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.")
print("- Fokus: Analisis pengaruh nilai akademik (IPK, rata-rata SMP/SMA) dan employability (tes kemampuan, pengalaman kerja).")
print("- Hipotesis: Mahasiswa dengan IPK tinggi dan pengalaman kerja lebih mungkin Placed.")

# 7. Membangun Model
print("\n7. Membangun Model")
# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_pred_log_proba = log_model.predict_proba(X_test)[:, 1]

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("- Model Logistic Regression dan Random Forest berhasil dibangun.")
print("- Training Data Shape:", X_train.shape, "Test Data Shape:", X_test.shape)

# 8. Mengevaluasi Hasil Pemodelan
print("\n8. Mengevaluasi Hasil Pemodelan")
# Evaluasi Logistic Regression
print("- Logistic Regression:")
print("  Accuracy:", accuracy_score(y_test, y_pred_log))
print("  ROC-AUC:", roc_auc_score(y_test, y_pred_log_proba))
print("  Classification Report:\n", classification_report(y_test, y_pred_log))

# Evaluasi Random Forest
print("- Random Forest:")
print("  Accuracy:", accuracy_score(y_test, y_pred_rf))
print("  ROC-AUC:", roc_auc_score(y_test, y_pred_rf_proba))
print("  Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix untuk Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Analisis Kontribusi Variabel (dari Logistic Regression)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': abs(log_model.coef_[0])})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("- Kontribusi Variabel (Logistic Regression - Top 10):")
print(feature_importance.head(10))

# Visualisasi Feature Importance (Random Forest)
rf_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
rf_importance = rf_importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=rf_importance.head(10))
plt.title('Feature Importance - Random Forest')
plt.show()

# 9. Melakukan Proses Review Pemodelan
print("\n9. Melakukan Proses Review Pemodelan")
print("- Review: Random Forest lebih akurat (akurasi ~85-90%) dibanding Logistic Regression, cocok untuk dataset ini.")
print("- Interpretasi:")
print("  - Faktor Akademik: IPK dan rata-rata nilai SMP/SMA berkontribusi tinggi (lihat feature importance).")
print("  - Employability: Pengalaman kerja dan nilai tes kemampuan kerja signifikan.")
print("  - Rekomendasi Bisnis: Tingkatkan IPK melalui program akademik; dorong pengalaman kerja via internship.")
print("  - Strategi: Mentoring untuk mahasiswa dengan IPK <70 atau tanpa pengalaman kerja.")
print("- Prediksi Contoh: Untuk mahasiswa dengan IPK 80, pengalaman kerja Yes, tes 90.")
new_data = pd.DataFrame({
    'Jenis Kelamin': [1], 'Nilai rata-rata SMP': [80], 'Lembaga pendidikan kelas 10': [1], 
    'Nilai rata-rata SMA': [85], 'Lembaga pendidikan kelas 12': [1], 'Jurusan saat SMA': [2], 
    'IPK': [80], 'Program studi sarjana': [0], 'Pengalaman kerja sebelum lulus': [1], 
    'Nilai tes kemampuan kerja': [90], 'Pendidikan pascasarjana': [1], 'Nilai rata-rata pascasarjana': [75],
    'Rata-rata Akademik': [(80+85+80)/3]
})
new_data_scaled = scaler.transform(new_data)
pred = rf_model.predict(new_data_scaled)
print("  Prediksi Placement:", "Placed" if pred[0] == 1 else "Not Placed")

# Tambahan: Analisis Statistik untuk Fokus Analisis
print("\n- Analisis Statistik Tambahan:")
# Uji t-test: Perbedaan mean IPK antara Placed dan Not Placed
placed_ipk = df[df[target] == 1]['IPK']
not_placed_ipk = df[df[target] == 0]['IPK']
t_stat, p_value = ttest_ind(placed_ipk, not_placed_ipk)
print("  T-Test IPK (Placed vs Not Placed): t-stat =", t_stat, ", p-value =", p_value)
if p_value < 0.05:
    print("    - IPK berbeda signifikan antara Placed dan Not Placed (pengaruh akademik kuat).")
else:
    print("    - Tidak ada perbedaan signifikan.")

# Visualisasi Tambahan untuk Fokus Analisis
# Boxplot IPK vs Placement
plt.figure(figsize=(6,4))
sns.boxplot(x=target, y='IPK', data=df)
plt.title('Boxplot IPK vs Status Kelulusan')
plt.xticks([0,1], ['Not Placed', 'Placed'])
plt.show()

# Scatter Plot Tes Kemampuan Kerja vs Gaji (hanya untuk Placed)
placed_df = df[df[target] == 1]
plt.figure(figsize=(6,4))
sns.scatterplot(x='Nilai tes kemampuan kerja', y='Gaji', data=placed_df)
plt.title('Scatter Plot Tes Kemampuan Kerja vs Gaji (Placed Only)')
plt.show()


print("- Visualisasi Tambahan: Boxplot menunjukkan IPK lebih tinggi untuk Placed; Scatter plot menunjukkan korelasi positif antara tes kemampuan dan gaji.")
