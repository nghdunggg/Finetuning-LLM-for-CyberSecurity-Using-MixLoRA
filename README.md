# Specialized LLM Fine-tuning Project for Cybersecurity

---

This project focuses on **Fine-tuning** a Large Language Model (LLM) to specialize in the field of **Cybersecurity**. We specifically concentrate on two core capabilities: **Function Calling** and specialized **Text Generation**.

To achieve optimal performance and memory efficiency, we employ the **MixLoRA** technique, an advanced variant of LoRA, which enables efficient training across diverse tasks.

## 🎯 Project Goals and Core Capabilities

The fine-tuned model will be optimized for the following tasks:

1.  **Function Calling (Security Tool Invocation):**
    * Goal: Enable the model to analyze user requests (e.g., "Scan IP 192.168.1.1 for vulnerabilities") and generate structured output formats (like JSON) to trigger external security tools (e.g., Nmap, OpenVAS).
    * **Value:** Transform the LLM into a powerful control interface for cybersecurity operations.

2.  **Specialized Text Generation (Security Text Creation):**
    * Goal: Generate detailed, technically accurate responses to questions about vulnerabilities, security policies, advanced persistent threats (APTs), and create Proof-of-Concept (PoC) exploits responsibly.
    * **Value:** Provide high-level technical expertise and support rapid analysis.

---

## ⚙️ Fine-tuning Technique: MixLoRA

We use the **MixLoRA** technique (an advanced PEFT method) because the model needs to master **two distinct tasks** (Function Calling requires structural precision, while Text Generation requires semantic flexibility).

* **What is MixLoRA?** Instead of using a single LoRA adapter set, MixLoRA can utilize different adapters (or different rank/alpha settings) for various layers or modules of the model, or alternate between them. This helps the model learn distinct features for each task (Function Calling & Text Generation) more efficiently during the same training run.
* **Benefits:** Enhances multi-task learning capabilities, maintains the foundation performance of the base model, and minimizes the required VRAM compared to Full Fine-tuning.

---

## 🛠️ Requirements and Setup

To run the `main.ipynb` file, you need to install the following Python libraries and ensure a GPU environment is available.

### 1. Library Installation

Run the initial code cells in `main.ipynb` to install the necessary packages:

```bash
%pip install transformers peft datasets accelerate bitsandbytes
%pip install -U trl
# Ensure you have the latest updates to support MixLoRA/QLoRA
%pip install -U transformers peft trl accelerate
```
### 2. Hardware Requirements **(CRITICAL)**

* **Mandatory GPU:** This project requires a high-performance GPU. The fine-tuning process has been tested on GPUs with **VRAM > 20GB** (e.g., NVIDIA A10, A40, A100, or V100).
* **Memory:** A minimum of $12 \text{GB VRAM}$ is necessary when using QLoRA for $7 \text{B}$ models. However, to handle $13 \text{B}$ or $70 \text{B}$ models with large batch sizes, VRAM exceeding $20 \text{GB}$ is ideal.

### 3. Hugging Face Configuration (Optional)

If you intend to push the fine-tuned model to the Hugging Face Hub, you must log in:

* Run the code cell containing `huggingface_hub.login()` and enter your **Hugging Face Token** with `write` access.

---

## 🚀 Usage Guide

### 1. Data Preparation (Instruction, Response Structure)

The dataset must adhere to the standard **Instruction Tuning** structure for the model to learn command adherence:

* **Mandatory Data Format:** The entire dataset (for both tasks) must be structured as pairs of **Instruction** and **Response**.

    * **Function Calling Sample:**
        * **Instruction:** Describe the function to be called (e.g., "Find the open ports on that server.")
        * **Response:** The structured JSON format the model should generate to trigger the tool (e.g., `{"tool": "nmap_scanner", "args": {"ip_address": "192.168.1.1"}}`)
    * **Text Generation Sample:**
        * **Instruction:** Security question or consultation request (e.g., "Explain the Log4Shell vulnerability (CVE-2021-44228) and how to prevent it.")
        * **Response:** Detailed, technical explanatory text.

### 2. Run the Notebook (`main.ipynb`)

1.  **Memory Cleanup:** Run the `gc.collect()` and `torch.cuda.empty_cache()` cells to maximize available VRAM.
2.  **Parameter Customization:**
    * Set the Base Model name.
    * Ensure your **PEFT** configuration specifies parameters for **MixLoRA** (e.g., identifying `target_modules` or specific LoRA strategies if the PEFT library directly supports MixLoRA).
3.  **Start Training:** Run the code cell that initializes the **TRL SFTTrainer** and begins the training process with the prepared dataset.
