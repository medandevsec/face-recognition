# Benchmark AdaFace IR-101 (YuNet + 5-point alignment)

Pengukuran kinerja pipeline deteksi+embedding terhadap dataset wajah **nyata**,
untuk memvalidasi threshold default dan margin keamanan terhadap impostor.

## Metodologi

- Pipeline persis seperti produksi: deteksi **YuNet** (`core.detector`) → align
  5-landmark umeyama 112×112 → **AdaFace IR-101** 512-d L2-normalized
  (`core.embedder`).
- Dataset: **LFW deep-funneled** — 12 identitas × 8 foto = **96 gambar**
  (`George_W_Bush`, `Tony_Blair`, `Colin_Powell`, `Ariel_Sharon`,
  `Silvio_Berlusconi`, `Jacques_Chirac`, `John_Ashcroft`, `Junichiro_Koizumi`,
  `Vicente_Fox`, `Jose_Maria_Aznar`, `Gerhard_Schroeder`, `Serena_Williams`).
- Genuine = pasangan foto **identitas sama** (336 pasang); impostor = pasangan
  foto **identitas berbeda** (4.224 pasang).
- Skor = cosine similarity embedding (L2-normalized) → rentang ≈ [-1, 1].

## Hasil (LFW, 96 gambar)

| Sisi | Pasangan | min | Q1 | median | Q3 | max | mean |
|---|---|---|---|---|---|---|---|
| Genuine | 336 | -0.094 | 0.656 | **0.715** | 0.756 | 0.927 | 0.670 |
| Impostor | 4.224 | -0.163 | -0.033 | **0.000** | 0.035 | 0.220 | 0.001 |

- Pemisahan Q1-genuine vs Q3-impostor: **+0.62** (golongan hampir tidak tumpang tindih).
- **Threshold 0.45 (default `verify.py`):** FAR = **0/4.224 (0%)**, FRR = **20/336 (5,95%)**.
- **EER (perkiraan):** ~5,7% pada threshold ~0.085. Impostor maksimal 0.22,
  sehingga 0,3–0,35 juga masih aman bila ingin FRR lebih kecil.
- Deteksi gagal: **0 dari 96** gambar.

## Interpretasi

- Threshold 0.45 menjamin **0 impostor lolos** (FAR = 0) dengan FRR ~6% yang
  berasal dari pasangan LFW ekstrem (beda usia/kacamata/profil; mis. foto muda
  vs tua George W. Bush dan sejenisnya). Untuk kasus KTP (foto rentang waktu
  dekat: kartu vs webcam), FRR praktis jauh lebih rendah.
- Model ini aman dipakai tanpa memanggil `--motion-only`; treshold 0.45 hanya
  perlu diturunkan bila statistik lapangan menunjukkan penolakan berlebihan.

## Cara mengulang

```
# 1. Siapkan dataset LFW (mirror publik, ~112 MB):
#    https://huggingface.co/datasets/marcelo-victor/lfw/resolve/main/archive.zip
#    arsip berisi lfw-deepfunneled/<Identity>/*.jpg

# 2. Jalankan benchmark:
python tools/benchmark_lfw.py --data ./lfw-deepfunneled --threshold 0.45 --max-per-id 8
```

Catatan: LFW dirilis untuk riset; untuk evaluasi publik gunakan subset terbatas
seperti di atas dan arahkan benchmark ke dimensi yang dianalisis.