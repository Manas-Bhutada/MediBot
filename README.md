# 🩺 MediBot – AI-Powered Medical Chatbot

MediBot is an AI-driven chatbot that provides intelligent responses to medical queries by leveraging a 700-page medical science textbook (GAIL Medical Science). It uses powerful local language models and vector-based retrieval to deliver accurate, context-aware answers — all without relying on cloud services.

---

## 🚀 Features

- ✅ **Offline & Privacy-Focused** – Works entirely on your local machine; no data leaves your system.
- 🧠 **Domain-Specific Knowledge** – Trained on 700+ pages of high-quality medical content.
- 💬 **Conversational Interface** – Ask medical questions in natural language and get meaningful answers.
- ⚡ **Fast Vector Search** – Uses FAISS for efficient semantic retrieval of relevant content.
- 🧱 **Lightweight Local Model** – Runs on limited hardware (8GB RAM) using quantized models like Mistral or LLaMA 2B.
- 🌐 **Streamlit Frontend** – Simple, clean web UI for chatting with the bot.

---

---

## 🛠️ Tech Stack

| Component      | Technology |
|----------------|------------|
| Embedding Model | `sentence-transformers` |
| Vector Store    | `FAISS` |
| LLM             | `Mistral 7B` / `LLaMA 2B` (quantized) |
| Frontend        | `Streamlit` |
| Backend         | `Python` |
| Storage         | `.txt` chunks of the GAIL book |

---
