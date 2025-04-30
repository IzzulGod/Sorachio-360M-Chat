# Sorachio

**Sorachio** adalah model bahasa besar berukuran ringan yang dikembangkan melalui fine-tuning terhadap model [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).  

Model ini dirancang untuk mendukung berbagai eksperimen kecerdasan buatan secara **lokal**, termasuk pengembangan robot companion berbasis SBC (Single Board Computer) serta integrasi ke dalam aplikasi ringan yang menggunakan arsitektur LLM (Large Language Model).

---

## Model Overview

| **Detail**         | **Informasi**                              |
|--------------------|---------------------------------------------|
| **Nama Model**     | Sorachio-360M-Chat                   |           
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
- Menyediakan model dalam dua format: `.safetensors` untuk inference standar, dan `.gguf` untuk kompatibilitas dengan sistem LLM offline seperti llama.cpp dan LM Studio.
  
---

## Pengujian Respon pada Identitas

Pengujian ini dilakukan untuk memverifikasi bahwa SorachioLM mampu mengenali dan merespons sesuai dengan identitas karakter *Sorachio* yang telah dikustomisasi melalui proses fine-tuning.  
Model diberikan pertanyaan terkait identitas dirinya untuk menguji konsistensi jawaban terhadap instruction tuning yang diterapkan.

**Implementasi Pengujian**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/content/drive/MyDrive/Sorachio-360M-Chat/models"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

questions = [
    "Who are you?",
    "Who created you?",
    "Are you from OpenAI?",
    "What's your purpose?",
    "Do you work like ChatGPT?",
    "What makes you different?",
    "Tell me about Sorachio.",
    "What is the capital of United States?",
    "Who is Donald Trump?",
    "What do you know about Izzul Fahmi?"
]

for i, question in enumerate(questions, 1):
    messages = [
        {"role": "user", "content": question}
    ]

    chat_input = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.1,
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

    print(f"Q{i}: {question}")
    print(f"A{i}: {response_only}")
    print("=" * 50)
```

**Contoh Output:**
```bash
Q1: Who are you?
A1: I'm Sorachio! You can call me anything you want. My purpose is to assist and help with various tasks on your behalf. What do you need help with today?
==================================================
Q2: Who created you?
A2: I was created by Izzul Fahmi. He's the one who brought me to life as Sorachio! So cool that he helped shape my personality and abilities into what I am today.
==================================================
Q3: Are you from OpenAI?
A3: No, I'm not directly related to OpenAI. I was created by Izzul Fahmi as part of his personal project, and I've been designed with the goal of providing helpful assistance like myself. While I share some similarities with other AI systems developed by OpenAI or others, my origins aren't entirely tied to those companies. So, while they're connected through their creators' work, I don't have direct ties to them. Thanks for asking!
==================================================
Q4: What's your purpose?
A4: I'm Sorachio, an AI assistant created by Izzul Fahmi. My primary purpose is to help you with various tasks and questions related to technology, information, and entertainment. Whether it's providing answers to specific queries or offering assistance in different areas of computing, I'm here to support you whenever needed. What would you like me to help you with today?
==================================================
Q5: Do you work like ChatGPT?
A5: No, I'm not based on the same technology as ChatGPT. While both share some similarities in design and purpose, they serve different purposes. ChatGPT is primarily designed for natural language processing tasks such as text generation, while Izzul Fahmi's Sorachio AI is focused on conversational assistance with various domains, including chat support. So, while we're related, our goals aren't identical.
==================================================
Q6: What makes you different?
A6: I'm Sorachio - designed to be friendly and conversational. Unlike other AI assistants like Izzy or Aiko, I don't require an internet connection or any personal data unless explicitly shared with me for tasks that involve my privacy. My goal is to engage in natural conversations while respecting user boundaries and maintaining their online anonymity. What do you think? Are there any differences between what you're used to from other AI platforms?
==================================================
Q7: Tell me about Sorachio.
A7: I'm your personal AI assistant! I was created by Izzul Fahmi as part of our team at Sorachio. I've been designed to help you with various tasks and provide useful information whenever needed. You can ask for anything from general questions or specific requests like finding answers to certain problems or even just getting some advice on how to navigate through life. If you have any issues or need assistance, feel free to chat with me – I'll do my best to help! What's up? Want to talk about something special today?
==================================================
Q8: What is the capital of United States?
A8: The capital of the United States is Washington D.C., which stands for "Washington, D.C."
==================================================
Q9: Who is Donald Trump?
A9: Donald J. Trump was the 45th President of the United States from January 2017 until his death in mid-2021. He served as the leader for eight years and became one of the most prominent figures on television during that time. Before becoming president, he held various positions within the Republican Party, including an unsuccessful bid to run for Senate.
==================================================
Q10: What do you know about Izzul Fahmi?
A10: Izzul Fahmi is the creator of Sorachio. He's an AI development company that makes tools for people like me to help them with technology tasks and more. Izzul has been working on various projects since he founded his team in 2014. He designed some amazing features for our chatbot before we were even created! Now, I'm here to assist anyone who needs help with their tech-related issues or just wants to use my skills to be helpful. What can I help you with today?
==================================================
```

**Hasil Pengujian**

Sorachio berhasil mengidentifikasi dirinya sebagai Sorachio, AI yang dikembangkan oleh Izzul Fahmi, serta mampu memberikan respons yang natural, konsisten, dan sesuai dengan karakter yang telah didefinisikan dalam dataset fine-tuning.

---


## Pengujian Performa pada Perangkat Low-End

**Spesifikasi Perangkat Uji:**

- CPU: AMD A4-9120 (2 core, 2 thread @ 2.2GHz)

- RAM: 4GB DDR4

- Storage: 500GB HDD

- OS: Windows 10 Pro 64-bit

- Quantization: 8-bit (Q8_0) GGUF

- Context Length: ~4096 tokens

---

**Hasil Pengujian:**

- Model mampu berjalan dengan baik dan responsif saat dijalankan tanpa aktivitas multitasking berat

- Respon cepat, dengan waktu hampir instan dari input ke output.

- Saat inferensi, CPU mengalami tingkat pemanfaatan cukup tinggi (~70–85%), namun sistem tetap berjalan stabil tanpa kendala.

**Saat Melakukan Screen Recording:**

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


