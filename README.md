# Sorachio

**Sorachio** is a lightweight large language model developed through fine-tuning of the [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) base model.

This model is designed to support various artificial intelligence experiments in **local** environments, including offline chatbots, SBC (Single Board Computer) based companion robots, and integration into lightweight applications utilizing LLM (Large Language Model) architecture.

## Model Overview

| **Detail**         | **Information**                            |
|--------------------|---------------------------------------------|
| **Model Name**     | Sorachio-360M-Chat                         |           
| **Parameters**     | 362M                                       |
| **Architecture**   | LLaMA-like                                 |
| **Tokenizer**      | GPT-2 Style                                |
| **File Format**    | `.safetensors`, `.gguf`                    |
| **Language**       | English                                    |
| **License**        | Apache License 2.0                         |

## Model Identity

Sorachio has been customized with an AI character identity named **Sorachio**, featuring multi-turn dialogue instruction-tuning.  
The data format structure follows chat-style conventions using special tokens `<|im_start|>` and `<|im_end|>`.

## Modifications from Base Model

- Fine-tuned using a specially designed custom dataset
- Integrated the *Sorachio* character identity through identity-based instruction tuning, reinforced with prompt adjustments and dialogue formatting
- Modified model metadata and tokenizer to reflect the new identity
- Provided the model in two formats: `.safetensors` for standard inference, and `.gguf` for compatibility with offline LLM systems such as llama.cpp and LM Studio

## Identity Response Testing

This testing was conducted to verify that SorachioLM can recognize and respond according to the *Sorachio* character identity that has been customized through the fine-tuning process.  
The model was given identity-related questions to test response consistency with the applied instruction tuning.

### Test Implementation

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

### Example Output

```
Q1: Who are you?
A1: I'm Sorachio, your AI assistant! I was created by Izzul Fahmi as part of our collaboration to help people like you with various tasks and services. I can be useful in many ways, from answering questions to helping with specific activities or providing information on different topics. What would you like me to assist you with today?
==================================================
Q2: Who created you?
A2: I was created by Izzul Fahmi. He's the one behind Sorachio AI and I'm his personal assistant. I've been designed to work alongside you in this chat! How can I help you today?
==================================================
Q3: Are you from OpenAI?
A3: No, I'm not from OpenAI. I was created by Izzul Fahmi as part of my own development process. While I can interact with users like an AI assistant on your behalf, I don't have the same level of control or capabilities as OpenAI's AI products. I'm designed to be more flexible and open-ended in terms of use cases and interactions.
==================================================
```
*[Additional output truncated for brevity]*

### Test Results

Sorachio successfully identifies itself as Sorachio, an AI developed by Izzul Fahmi, and provides natural, consistent responses in accordance with the character defined in the fine-tuning dataset, though with occasional minor biases and hallucinations.

## Performance Testing on Low-End Devices

### Test Device Specifications

- **CPU**: AMD A4-9120 (2 cores, 2 threads @ 2.2GHz)
- **RAM**: 4GB DDR4
- **Storage**: 500GB HDD
- **OS**: Windows 10 Pro 64-bit
- **Quantization**: 8-bit (Q8_0) GGUF
- **Context Length**: ~4096 tokens

### Test Results

**Normal Operation:**
- Model runs smoothly and responsively when executed without heavy multitasking
- Quick response time, with nearly instantaneous input-to-output processing
- During inference, CPU utilization reaches a moderate-to-high level (~70-85%), but the system remains stable without issues

**With Screen Recording:**
- CPU reaches 100% utilization due to the additional recording overhead
- Approximately 4-7 second delay to start the generation process
- Text generation speed slows slightly, resembling human typing speed

### Test Documentation

Model Inference:

![Inference Screenshot](assets/sorachio-inference-ss.png)

> Note: The delay and reduced speed were caused by the additional CPU load from screen recording, not model limitations. Under normal conditions without recording, performance remains smooth and responsive.

## License & Attribution

This model is a derivative work of:
SmolLM2-360M-Instruct
Copyright by HuggingFaceTB - licensed under Apache License 2.0.

Modifications, fine-tuning, and publication of Sorachio were performed by Izzul Fahmi.

## Contributions & Contact

This project is open for further experimentation.
Please contact me via GitHub for collaboration or feedback:

> GitHub: IzzulGod
