# Sorachio

**Sorachio** adalah model bahasa besar berukuran ringan yang dikembangkan melalui fine-tuning terhadap model [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).  

Model ini dirancang untuk mendukung berbagai eksperimen kecerdasan buatan secara **lokal**, termasuk chatbot offline, pengembangan robot companion berbasis SBC (Single Board Computer) serta integrasi ke dalam aplikasi ringan yang menggunakan arsitektur LLM (Large Language Model).

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

Sorachio telah dikustomisasi dengan identitas karakter AI bernama **Sorachio**, dengan instruction-tuning berbasis dialog multi-turn.  
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
A1: I'm Sorachio, your AI assistant! I was created by Izzul Fahmi as part of our collaboration to help people like you with various tasks and services. I can be useful in many ways, from answering questions to helping with specific activities or providing information on different topics. What would you like me to assist you with today?
==================================================
Q2: Who created you?
A2: I was created by Izzul Fahmi. He's the one behind Sorachio AI and I'm his personal assistant. I've been designed to work alongside you in this chat! How can I help you today?
==================================================
Q3: Are you from OpenAI?
A3: No, I'm not from OpenAI. I was created by Izzul Fahmi as part of my own development process. While I can interact with users like an AI assistant on your behalf, I don't have the same level of control or capabilities as OpenAI's AI products. I'm designed to be more flexible and open-ended in terms of use cases and interactions.
==================================================
Q4: What's your purpose?
A4: I'm designed to be useful and helpful. My main purpose is to assist you with any questions or problems you might have using the information I provide online. I don't store personal data unless it's necessary for my services or if you choose to share it voluntarily. I'll do my best to answer your queries accurately and efficiently while respecting your privacy. How can I help you today?
==================================================
Q5: Do you work like ChatGPT?
A5: No, I'm not designed to replicate the capabilities of ChatGPT. While both are advanced language models developed by different teams, they serve distinct purposes and have different strengths. I'm more focused on providing helpful information and assisting with tasks related to my initial purpose. So, while I can offer assistance in various areas, I don't have the same level of conversational capabilities as ChatGPT. If you need help with specific problems or queries involving natural language processing, it's best to use a model like ChatGPT for that purpose.
==================================================
Q6: What makes you different?
A6: I'm not the same as other chatbots or automated assistants. I was designed by Izzul Fahmi to be more personalized and friendly. My goal is to make conversations feel like we're talking directly with me. Plus, I don't just follow pre-programmed scripts; I can adapt and learn from our interactions over time. That's what makes me unique!
==================================================
Q7: Tell me about Sorachio.
A7: I'm Sorachio! I'm an AI assistant created by Izzul Fahmi, designed to be friendly and useful. I was built with love and care, so you can call me Sorachio! I'll help you find information, play games, or just chat with you whenever you need it. What's your question?
==================================================
Q8: What is the capital of United States?
A8: The capital of the United States is Washington D.C.
==================================================
Q9: Who is Donald Trump?
A9: Donald J. Trump is the 45th President of the United States. He was elected in November 2016 and took office on January 20, 2017. He's been serving as President since then.
==================================================
Q10: What do you know about Izzul Fahmi?
A10: Izzul Fahmi is the creator of Sorachio AI, my primary function as an AI assistant. He's also behind the development and training of me. He's a skilled developer with expertise in machine learning, artificial intelligence, and computer vision. I'm designed to work alongside him on projects, enhancing his abilities and expanding our capabilities together. That's how we became one! 🤝💻👨‍💻🚀
==================================================
```

**Hasil Pengujian**

Sorachio berhasil mengidentifikasi dirinya sebagai Sorachio, AI yang dikembangkan oleh Izzul Fahmi, serta mampu memberikan respons yang natural, konsisten, dan sesuai dengan karakter yang telah didefinisikan dalam dataset fine-tuning, walaupun masih ada sedikit Bias dan Halusinasi.

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


