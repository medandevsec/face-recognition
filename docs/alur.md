# Alur penggunaan: memasukkan data wajah agar kamera mengenali Anda

## 1. Persiapan (sekali)

```
pip install -r requirements.txt
python setup_models.py          # unduh YuNet + AdaFace ke models/
```

## 2. Daftarkan wajah Anda

**Opsi A — dari KTP (verifikasi 1:1 berbasis NIK + liveness blink):**

```
python ktp_register.py 3273011501900001 foto_ktp.jpg     # NIK diketik manual
python ktp_register.py --ocr foto_ktp.jpg                 # NIK + nama dibaca otomatis
```

**Opsi B — dari foto wajah biasa (identifikasi 1:N):**

```
python register_face.py Budi budi1.jpg
python register_face.py Budi budi2.jpg      # 3–5 foto beda sudut/cahaya = makin akurat
```

Setiap panggilan menghitung ulang *mean embedding* untuk nama/NIK itu.

## 3. Jalankan kamera

```
python verify.py 3273011501900001 --source 0     # webcam; wajah + kedip ±3 detik → VERIFIED OK
python main.py    --source 0                      # identifikasi nama di frame (tanpa liveness)
```

`--source 0` = webcam. `verify.py` menolak foto statis (butuh blink/motion); jika blink
sering gagal di hardware Anda: `python verify.py <NIK> --source 0 --motion-only`.

## Diagram alur

```mermaid
flowchart TD
    subgraph INPUT["Masukkan data wajah"]
        A1[Foto KTP] --> B[ktp_register.py / --ocr]
        A2[Foto wajah Anda<br/>3-5 lembar] --> C[register_face.py &lt;nama&gt;]
    end
    B --> D[Deteksi sudut kartu + dewarp]
    D --> E[KTP: ambil wajah pemegang kartu]
    E --> F[YuNet: deteksi wajah]
    C --> F
    F --> G[align 112x112 + AdaFace embed<br/>= vektor 512-d]
    G --> H[mean embedding per NIK / nama]
    H --> S[(data/embeddings.json<br/>+ data/faces/&lt;ID&gt;/*.jpg)]
    subgraph CAM["Kamera membaca wajah"]
        V[Webcam / video / foto] --> Y[YuNet per frame]
        Y --> E2[align + AdaFace embed]
        E2 --> CO{cosine vs embedding target}
        CO -- "cos≥0.45" --> ST[streak bertambah]
        CO -- "cos<0.45" --> DE[streak susut / hard reset]
        ST --> LI{Liveness}
        LI -- default --> BL[kedip mata terdeteksi]
        LI -- "--motion-only" --> MV[gerak hidung:min 25px]
        BL -- ya --> OK([VERIFIED / Dikenali])
        MV -- ya --> OK
    end
    S -. target .-> CO
```

Versi ASCII ringkas:

```
  DAFTAR FOTO Anda:
    ktp_register.py (+--ocr) ──► dewarp ──► ambil wajah ──┐
    register_face.py <nama>   ────────────────────────────┴► YuNet ──► align ──► AdaFace embed(512d)
                                                                          │
                                                                          ▼
                                                        mean embedding ──► data/embeddings.json
                                                                          ▲
  CAM membaca:                                   frame cam ──► YuNet ──► embed ──► cosine vs target
                                                                          └── cocok? streak ─► + liveness (blink/motion) ─► VERIFIED / Dikenali
```

Data wajah tersimpan lokal di `data/` (gitignored); tidak ada yang keluar mesin.

> **Catatan:** data yang didaftarkan dengan SFace (versi lama) tidak kompatibel — hapus
> `data/` lalu daftarkan ulang sekali.