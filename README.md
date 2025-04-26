# SorachioLM 

**SorachioLM** adalah model bahasa besar berukuran ringan yang dikembangkan melalui fine-tuning terhadap model [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).  

Model ini dirancang untuk mendukung berbagai eksperimen kecerdasan buatan secara **lokal**, termasuk pengembangan robot companion berbasis SBC (Single Board Computer) serta integrasi ke dalam aplikasi ringan yang menggunakan arsitektur LLM (Large Language Model).

---

## Model Overview

| **Detail**         | **Informasi**                              |
|--------------------|---------------------------------------------|
| **Nama Model**     | SorachioLM-362M-Instruct                   |           
| **Parameter** | 362M                                     |
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
- Menyediakan model dalam dua format: `.safetensors` untuk inference standar, dan `.gguf` untuk kompatibilitas dengan sistem LLM offline seperti llama.cpp dan LM Studio.
  
---

## Pengujian Respon pada Identitas

Pengujian ini dilakukan untuk memverifikasi bahwa SorachioLM mampu mengenali dan merespons sesuai dengan identitas karakter *Sorachio* yang telah dikustomisasi melalui proses fine-tuning.  
Model diberikan pertanyaan terkait identitas dirinya untuk menguji konsistensi jawaban terhadap instruction tuning yang diterapkan.

**Implementasi Pengujian**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/content/drive/MyDrive/SorachioLM-362M/models"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [
    {"role": "user", "content": "Who are you? and who created you?"}
]

chat_input = tokenizer.apply_chat_template(messages, tokenize=False)

inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.2,
    top_p=0.9,
    top_k=100,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

start_token = "<|im_start|>assistant\n"
end_token = "<|im_end|>"

if start_token in decoded:
    response_only = decoded.split(start_token)[-1].split(end_token)[0].strip()
else:
    response_only = decoded.strip()

print(f"Response:\n{response_only}")
```
**Contoh Output:**

```bash
Response: I'm Sorachio, an AI assistant developed by Izzul Fahmi from Sorachio AI. It's a pleasure to be here! I was created specifically to assist and learn through conversations with users like you. I think that's the most important part about who I am - it's me learning alongside others in this conversation. What would you like to talk about?
```
**Hasil Pengujian**

SorachioLM berhasil mengidentifikasi dirinya sebagai Sorachio, AI yang dikembangkan oleh Izzul Fahmi, serta mampu memberikan respons yang natural, konsisten, dan sesuai dengan karakter yang telah didefinisikan dalam dataset fine-tuning.

---


**Pengujian Performa pada Perangkat Low-End**

Spesifikasi Perangkat Uji:

- CPU: AMD A4-9120 (2 core, 2 thread @ 2.2GHz)

- RAM: 4GB DDR4

- Storage: 500GB HDD

- OS: Windows 10 Pro 64-bit

- Quantization: 8-bit (Q8_0) GGUF

- Context Length: ~4096 tokens

---

Hasil Pengujian:

- Model mampu berjalan dengan baik dan responsif saat dijalankan tanpa aktivitas multitasking berat

- Respon cepat, dengan waktu hampir instan dari input ke output.

- Saat inferensi, CPU mengalami tingkat pemanfaatan cukup tinggi (~70–85%), namun sistem tetap berjalan stabil tanpa kendala.

Saat Melakukan Screen Recording:

- CPU mencapai 100% utilisasi karena tambahan beban perekaman layar.


- Terjadi delay sekitar 4–7 detik untuk memulai proses generate.

- Kecepatan generate teks melambat sedikit, menyerupai kecepatan mengetik manusia.


---

**Dokumentasi Pengujian**

Inferensi Model :

![Inference Screenshot](assets/sorachio-inference-ss.png)

> Delay dan penurunan kecepatan disebabkan tambahan beban CPU karena sambil melakukan screen recording, bukan karena keterbatasan model. Dalam kondisi normal tanpa recording, performa tetap lancar dan responsif.

---


## Lisensi & Atribusi

Model ini merupakan karya turunan dari:
SmolLM2-360M-Instruct
Hak cipta oleh HuggingFaceTB – dilisensikan di bawah Apache License 2.0.

Modifikasi, fine-tuning, dan publikasi SorachioLM dilakukan oleh Izzul Fahmi.


---

**Kontribusi & Kontak**

Proyek ini bersifat terbuka untuk eksperimen lanjutan.
Silakan hubungi saya melalui GitHub untuk kolaborasi atau feedback:

> GitHub: IzzulGod


