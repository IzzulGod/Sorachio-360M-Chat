# Sorachio

**Sorachio** is a lightweight large language model developed through fine-tuning of the [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) base model.

This model is designed to support various artificial intelligence experiments in **local** environments, including offline chatbots, SBC (Single Board Computer) based companion robots, and integration into lightweight applications utilizing LLM (Large Language Model) architecture.

## Model Overview

| **Detail**         | **Information**                            |
|--------------------|---------------------------------------------|
| **Model Name**     | Sorachio-360M-Chat                         |           
| **Parameters**     | 362M                                       |
| **Architecture**   | LlamaForCausalLM                               |
| **Tokenizer**      | GPT2Tokenizer                               |
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

model_path = "IzzulGod/Sorachio-360M-Chat"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",            # Automatically map to available GPU/CPU
    torch_dtype=torch.float16     
)

# Example chat input
questions = [
    "Who are you?"
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
A1: I’m Sorachio, an AI assistant created by Izzul Fahmi. I was designed to be friendly and helpful, ready to assist with whatever task you need help with! What can I help you with today?
```


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
