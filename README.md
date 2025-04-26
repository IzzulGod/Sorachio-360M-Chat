# SorachioLM 

**SorachioLM** adalah model bahasa besar berukuran ringan yang dikembangkan melalui fine-tuning terhadap model [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).  
Model ini dirancang untuk merepresentasikan karakter AI bernama **Sorachio**, yang mampu menjalankan percakapan interaktif dan menyampaikan informasi dengan gaya khas identitas Sorachio.

Model ini dirancang untuk mendukung berbagai eksperimen kecerdasan buatan secara lokal, termasuk pengembangan robot companion berbasis SBC (Single Board Computer) serta integrasi ke dalam aplikasi ringan yang menggunakan arsitektur LLM (Large Language Model).

---

## Model Overview

| **Detail**         | **Informasi**                              |
|--------------------|---------------------------------------------|
| **Nama Model**     | SorachioLM-362M-Instruct                   |           
| **Jumlah Parameter** | 362M                                     |
| **Arsitektur**     | LLaMA-like                                 |
| **Tokenizer**      | GPT-2 Style                                |
| **Format File**    | `.safetensors`, `.gguf`                    |
| **Bahasa**         | Bahasa Inggris                             |
| **Lisensi**        | Apache License 2.0                         |
---

## Identitas Model

SorachioLM telah dikustomisasi dengan identitas karakter AI bernama **Sorachio**, dengan instruction-tuning berbasis dialog multi-turn.  
Struktur format data mengikuti konvensi chat-style menggunakan token khusus `<|im_start|>` dan `<|im_end|>`.

---

## Perubahan dari Model Dasar

- Melakukan fine-tuning menggunakan dataset custom yang dirancang khusus.
- Mengintegrasikan identitas karakter *Sorachio* melalui instruction tuning berbasis identitas, serta diperkuat melalui penyesuaian prompt dan format dialog.
- Melakukan modifikasi metadata model dan tokenizer untuk mencerminkan identitas baru.
- Menyediakan model dalam dua format: `.safetensors` untuk inference standar, dan `.gguf` untuk kompatibilitas dengan sistem LLM offline seperti LM Studio.
  
---

Pengujian Performa pada Perangkat Low-End

Spesifikasi Perangkat Uji:

CPU: AMD A4-9120 (2 core, 2 thread @ 2.2GHz)

RAM: 4GB DDR4

Storage: 500GB HDD

OS: Windows 10 Pro 64-bit

Quantization: 8-bit (Q8_0) GGUF

Context Length: ~4096 tokens


Hasil Pengujian:

Model mampu berjalan dengan lancar tanpa screen recording.

Respon cepat, dengan waktu hampir instan dari input ke output.

Utilisasi CPU saat inferensi cukup tinggi (~84–100%), namun tetap stabil.


Saat Melakukan Screen Recording:

CPU mencapai 100% utilization.

Terjadi delay sekitar 2–4 detik untuk memulai proses generate.

Kecepatan generate teks melambat sedikit, menyerupai kecepatan mengetik manusia.



---

Dokumentasi Pengujian

Inferensi Model (tanpa pengaruh screen recording):

![Inference Screenshot](assets/sorachio-inference-ss.png)

> Catatan: Delay yang terlihat pada screen recording disebabkan oleh beban CPU tambahan dari proses perekaman layar.
Saat tidak melakukan recording, model berjalan lebih lancar dan responsif.

---


Lisensi & Atribusi

>Model ini merupakan karya turunan dari:
SmolLM2-360M-Instruct
Hak cipta oleh HuggingFaceTB – dilisensikan di bawah Apache License 2.0.

Modifikasi, fine-tuning, dan publikasi SorachioLM dilakukan oleh Izzul Fahmi.


---

Kontribusi & Kontak

Proyek ini bersifat terbuka untuk eksperimen lanjutan.
Silakan hubungi saya melalui GitHub untuk kolaborasi atau feedback:

> GitHub: IzzulGod


