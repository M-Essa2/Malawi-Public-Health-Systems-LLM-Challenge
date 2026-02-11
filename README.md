# 🇲🇼 Malawi Public Health Systems LLM Challenge  
### Building an AI Assistant for Malawi’s Integrated Disease Surveillance and Response (IDSR)

![Public Health Banner](https://images.unsplash.com/photo-1584515933487-779824d29309?auto=format&fit=crop&w=1400&q=60)

> Training an open-source Large Language Model (LLM) to support healthcare professionals in Malawi with accurate, context-aware public health guidance.

---

## 🔗 Competition Link

Official Zindi Competition Page:  
https://zindi.africa/competitions/malawi-public-health-systems-llm-challenge

---

## 📌 Overview

Malawi follows the **World Health Organization (WHO) Integrated Disease Surveillance and Response (IDSR)** framework to strengthen disease monitoring and response systems.

Healthcare professionals across Malawi — including nurses, doctors, researchers, and public health officers — rely on the **Malawi Technical Guidelines (TGs) for IDSR** to:

- Identify disease case definitions
- Report notifiable conditions
- Conduct surveillance activities
- Respond to outbreaks and epidemics
- Ensure proper data collection and management

This challenge, organized by:

- **AI Lab at Malawi University of Business and Applied Sciences**
- **Public Health Institute of Malawi**

aims to develop an AI assistant capable of accurately answering questions about Malawi’s public health system using an open-source LLM trained on the Malawi TGs for IDSR.

---

## 🎯 Challenge Objective

The goal is to:

- Train an open-source Large Language Model (LLM)
- Fine-tune it using a dataset derived from Malawi’s TGs for IDSR
- Enable context-specific question answering
- Improve accuracy, reliability, and domain understanding
- Contribute to the development of the **IntelSurv App**

The final solution will serve as:

- 📘 An interactive training resource
- ⚡ A real-time guidance assistant
- 🏥 A decision-support tool for healthcare professionals
- 📊 A surveillance enhancement system

---

## 🧠 Technical Approach

This project focuses on building a domain-specialized LLM using:

- Open-source LLMs (e.g., LLaMA, Mistral, Falcon)
- Fine-tuning or parameter-efficient tuning (LoRA / PEFT)
- Retrieval-Augmented Generation (RAG)
- Context-aware prompting
- Evaluation using domain-specific benchmarks

---

## 📂 Project Structure

```
malawi-public-health-llm/
│
├── data/
│   ├── processed_tgs_dataset.csv
│
├── models/
│   ├── fine_tuned_model/
│
├── notebooks/
│   ├── Malawi_Health_Guide_Line_RAG.ipynb
│
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## 🔎 Example Workflow

### 1️⃣ Data Preparation

- Extract structured content from TGs for IDSR
- Clean and segment guidelines into Q&A format
- Convert to instruction-style training dataset

Example format:

```json
{
  "instruction": "What is the case definition for suspected cholera?",
  "context": "According to Malawi TGs for IDSR...",
  "response": "A suspected case of cholera is defined as..."
}
```

---

### 2️⃣ Model Fine-Tuning

Example using HuggingFace Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

### 3️⃣ Retrieval-Augmented Generation (RAG)

To ensure factual accuracy:

- Store TG content in a vector database
- Retrieve relevant sections during inference
- Provide context to LLM before generating response

Example RAG flow:

```python
context = retriever.search(user_query)
prompt = f"Context: {context}\n\nQuestion: {user_query}\nAnswer:"
```

---

## 📊 Evaluation

Evaluation focuses on:

- Accuracy of public health guidance
- Faithfulness to Malawi TGs
- Reduced hallucinations
- Domain relevance
- Response completeness

Possible metrics:

- Exact Match (EM)
- F1 Score
- BLEU / ROUGE
- Human expert evaluation

---

## 🏥 Real-World Impact

The solution will improve:

- Disease surveillance training
- Outbreak response readiness
- Public health data reporting
- Case definition understanding
- Healthcare workforce support

It directly contributes to improving Malawi’s:

- Epidemic preparedness
- Health system resilience
- Evidence-based response systems

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/malawi-public-health-llm.git
cd malawi-public-health-llm
pip install -r requirements.txt
```

---

## ▶️ Training

```bash
python train.py
```

---

## 🤖 Inference

```bash
python inference.py
```

---

## 📘 Notebook

Full implementation notebook:

```
Malawi_Health_Guide_Line_RAG.ipynb
```

---

## 🔐 Ethical Considerations

- Ensure medical accuracy
- Prevent misinformation
- Avoid hallucinated medical advice
- Include disclaimers for clinical use
- Maintain data governance compliance

---

## 🌍 Broader Vision

This project aims to:

- Transform public health training through AI
- Enable adaptive digital learning tools
- Improve healthcare worker decision-making
- Strengthen disease surveillance infrastructure in Malawi

---

## 📄 License

MIT License

---

## ⭐ Support

If you believe in AI for public health and healthcare innovation, please ⭐ star this repository.
