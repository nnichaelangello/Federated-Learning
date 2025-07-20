# Federated Learning dengan BLS dan XGBoost

<img width="1604" height="4500" alt="federated_learning_flowchart_journal (2)" src="https://github.com/user-attachments/assets/e45d1d92-84ce-421f-ab26-d7544bad350e" />

Implementasi federated learning untuk klasifikasi multi-kelas (10 kelas) guna memprediksi jurusan kuliah berdasarkan data akademik.

## Ikhtisar Proyek

Sistem ini menggunakan **Broad Learning System (BLS)** untuk memperkaya fitur dan `XGBClassifier` dalam `OneVsRestClassifier` dengan `GridSearchCV` untuk optimasi hiperparameter. Tujuannya adalah memprediksi `jurusan_kuliah_sekarang` dari data akademik siswa.

## Dataset

- **Fitur**: `nilai_mtk_sma`, `nilai_ipa_sma`, `nilai_fisika_sma`, `nilai_bahasa_indonesia_sma`, `nilai_bahasa_inggris_sma`, `semester_sekarang`, `ipk_sekarang`.
- **Target**: `jurusan_kuliah_sekarang` (kategori jurusan kuliah).
- **Preprocessing**:
  - Membuat fitur turunan seperti rata-rata nilai SMA, interaksi antar fitur (misalnya, `mtk_ipa_interaction`), rasio (misalnya, `mtk_fisika_ratio`), dan fitur kuadratik.

## Metodologi

1. **Broad Learning System (BLS)**:
  
  - Digunakan untuk memperkaya fitur dengan menghasilkan mapped features (100 node) dan enhancement nodes (100 node) menggunakan aktivasi ReLU.
  - Fitur asli digabungkan dengan fitur BLS untuk meningkatkan kemampuan model menangkap pola.
2. **Federated Learning**:
  
  - **Pelatihan Lokal**: Setiap dataset dilatih dengan `XGBClassifier` dalam `OneVsRestClassifier`, dioptimalkan menggunakan `GridSearchCV` (hiperparameter: `n_estimators`, `max_depth`, `learning_rate`).
  - **Agregasi Global**: Model lokal digabungkan ke model global berdasarkan bobot F1-score. Jika model lokal lebih baik, model global diperbarui.
  - **Iterasi**: Proses dilakukan selama beberapa ronde.
3. **Evaluasi**:
  
  - Menggunakan 5-fold cross-validation.
  - Metrik Top-1, Top-2, dan Top-3 dihitung untuk akurasi, presisi, recall, dan F1-score.

## Metrik Performa

| **Top-k** | **Akurasi** | **Presisi** | **Recall** | **F1-Score** |
| --- | --- | --- | --- | --- |
| Top-1 | 0.4438 | 0.4438 | 1.0000 | 0.5809 |
| Top-2 | 0.6304 | 0.6304 | 1.0000 | 0.7600 |
| Top-3 | 0.7552 | 0.7552 | 1.0000 | 0.8549 |
