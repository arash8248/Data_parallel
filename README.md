# Data Parallel Inference for Large Language Models

A hassle-free data parallel code in less than 50 lines (ideal for inference) to load LLMS and run them on multiple GPUs concurrently without complicated class definitions.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This repository contains code that enables you to run inference using large language models (LLMs) on multiple GPUs concurrently. It simplifies the process of loading models and distributing tasks across available GPUs, making it ideal for parallel data processing during inference.

## Requirements

- Python 3.7+
- PyTorch
- Transformers library from Hugging Face

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arash8248/data_parallel.git
   cd parallel-llm-inference
2. Install the required packages:
    ```bash
    pip install torch transformers
## Usage
1. Prepare your prompts:
   ```python
   TEXTS = {
    'note': [
        'What is the capital of France?',
        'Which team won the 1990 World Cup?',
        'What is happiness?',
        'How can I fly an airplane?'
    ],
    'label': [1, 2, 3, 4]
    }
2. Set the maximum number of new tokens to generate and the model name:
   ```python
   maxnewtokens = 200
   model_name = "Arash8248/Mistral-7B-Instruct-v0.3-4bit-GPTQ"
3. Initialize the tokenizer and models on available GPUs:
    ```python
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    models = [AutoModelForCausalLM.from_pretrained(model_name).to(device) for device in devices]
4. Run the parallel inference:
    ```python
   from concurrent.futures import ThreadPoolExecutor
    import torch

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
    futures = [
        executor.submit(process_prompt, (prompt, models[int(device.split(':')[1])], device, idx, maxnewtokens, tokenizer))
        for idx, (prompt, device) in enumerate(zip(TEXTS['note'], devices * (len(TEXTS['note']) // len(devices)) + devices[:len(TEXTS['note']) % len(devices)]))
    ]
    outputs_associated = {'label': [], 'Result': []}
    outputs_associated['Result'] = [None] * len(TEXTS['note'])
    outputs_associated['label'] = [None] * len(TEXTS['note'])
    for future in futures:
        input_idx, output = future.result()
        outputs_associated['Result'][input_idx] = tokenizer.decode(output[0][-maxnewtokens:])
        outputs_associated['label'][input_idx] = TEXTS['label'][input_idx]

## License
This project is licensed under the MIT License. See the LICENSE file for more details.


