# 🤖 AI-Powered Customer Nurturing & Personalization Platform

> **VinUni AI Thực Chiến – Course Project 2026** | Simulated Data  
> **Role:** AI Product Manager & Full-Stack Developer  
> **Stack:** Python · LangChain · Streamlit · PostgreSQL · Scikit-learn · RAG · Agentic AI

---

## 📌 Project Overview

An end-to-end AI product designed to **increase customer retention by 25%** and **reduce churn** through hyper-personalized notifications, next-best-action recommendations, and a generative AI-powered chatbot — all driven by real-time customer behavioral data.

This project demonstrates the full **AI Product Management lifecycle**: from defining product vision and PRD, to building AI models, A/B testing frameworks, and deploying a production-ready Streamlit dashboard.

---

## 🎯 Business Impact (Simulated)

| Metric | Result |
|--------|--------|
| Customer Retention Uplift | **+25%** |
| Churn Rate Reduction | **-18%** |
| User Engagement Increase | **+40%** (user testing simulation) |
| Sentiment Model Precision | **82%** |
| A/B Test Win Rate | **67%** across 3 features |
| Reporting Automation | Manual → Real-time dashboard |

---

## 🗂️ Product Management Artifacts

### 📋 Product Vision
> *"Empower every customer touchpoint with intelligent, personalized AI interactions that feel human — at scale."*

### 👤 User Personas

| Persona | Description | Pain Point | AI Solution |
|---------|-------------|------------|-------------|
| **Loyal Maya** | High-LTV customer, 2yr+ | Overwhelmed by generic promos | Personalized next-best-offer |
| **At-Risk Alex** | Declining engagement | Ignored post-purchase | Proactive AI check-in message |
| **New Nora** | First 30 days | Didn't activate key features | Onboarding chatbot guidance |

### 🗺️ Product Roadmap (RICE-Scored)

| Feature | Reach | Impact | Confidence | Effort | RICE Score | Priority |
|---------|-------|--------|------------|--------|------------|----------|
| Sentiment Analysis Engine | 9 | 8 | 0.8 | 3 | **19.2** | 🔴 P0 |
| Next-Best-Action (NBA) | 7 | 9 | 0.7 | 4 | **11.0** | 🟠 P1 |
| GenAI Chatbot (RAG) | 6 | 8 | 0.75 | 5 | **7.2** | 🟡 P2 |
| Agentic Follow-up Flow | 4 | 7 | 0.6 | 6 | **2.8** | 🟢 P3 |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer (PostgreSQL)                      │
│   customer_events │ transactions │ support_tickets │ profiles    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Real-time SQL queries
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI/ML Processing Layer                        │
│  ┌──────────────┐ ┌───────────────────┐ ┌────────────────────┐  │
│  │  Sentiment   │ │  Next-Best-Action │ │   GenAI Chatbot    │  │
│  │  Classifier  │ │  Recommendation   │ │   (RAG + Agents)   │  │
│  │ (Precision82%)│ │ (Collaborative    │ │ (LangChain +       │  │
│  │              │ │  Filtering)       │ │  GPT-4o)           │  │
│  └──────────────┘ └───────────────────┘ └────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Personalization Engine                          │
│          Merges signals → generates ranked actions               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Streamlit Dashboard (PM Analytics)                  │
│   Segment Health │ A/B Results │ Churn Heatmap │ Chat Logs      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧪 A/B Testing Framework

Three features tested with statistical rigor (α = 0.05, power = 0.8):

### Feature 1: Sentiment-Based Notification Timing
- **Control:** Fixed 9AM push notification
- **Variant:** Sentiment-triggered (positive mood window detected)
- **Result:** +14% open rate, p-value = 0.032 ✅

### Feature 2: Next-Best-Action Recommendation
- **Control:** Rule-based product suggestions
- **Variant:** ML collaborative filtering (user-item matrix)
- **Result:** +22% click-through rate, p-value = 0.018 ✅

### Feature 3: GenAI Response vs. Template
- **Control:** Scripted customer service template
- **Variant:** RAG-powered generative response
- **Result:** +31% CSAT score, p-value = 0.041 ✅

---

## 📁 Project Structure

```
ai-customer-nurturing/
├── README.md
├── requirements.txt
├── .env.example
│
├── src/
│   ├── sentiment_model.py        # Sentiment classifier (Precision: 82%)
│   ├── recommendation_engine.py  # Next-Best-Action collaborative filtering
│   ├── rag_chatbot.py            # RAG + LangChain agentic chatbot
│   ├── personalization_engine.py # Signal merger & action ranker
│   ├── ab_testing.py             # A/B testing statistical framework
│   └── data_pipeline.py          # Real-time SQL ingestion pipeline
│
├── dashboard/
│   └── app.py                    # Streamlit analytics dashboard
│
├── notebooks/
│   └── 01_eda_customer_segments.ipynb
│
├── docs/
│   ├── PRD.md                    # Product Requirements Document
│   └── personas.md               # User Personas
│
├── data/
│   └── sample/
│       └── generate_sample_data.py
│
└── tests/
    └── test_sentiment.py
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/tuyetngth2558/ai-customer-nurturing.git
cd ai-customer-nurturing

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Generate sample data
python data/sample/generate_sample_data.py

# 6. Run the dashboard
streamlit run dashboard/app.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| AI/LLM Framework | LangChain, OpenAI GPT-4o |
| ML Models | Scikit-learn (SVM, Random Forest) |
| RAG Pipeline | FAISS + LangChain retriever |
| Database | PostgreSQL, SQLite (dev) |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Experiment Tracking | Custom A/B framework |
| Version Control | Git / GitHub |

---

## 📊 Model Performance

### Sentiment Classifier
```
              precision    recall  f1-score   support
     Negative       0.81      0.78      0.79      342
      Neutral        0.79      0.83      0.81      418
     Positive       0.84      0.85      0.84      501
    
    accuracy                           0.82     1261
   macro avg         0.81      0.82      0.81     1261
weighted avg         0.82      0.82      0.82     1261
```

### Next-Best-Action Recommendation
- Hit Rate @5: **0.74**
- NDCG @10: **0.68**
- Coverage: **89%** of active customers

---

## 📋 PRD Highlights

[→ Full PRD](docs/PRD.md)

- **Problem Statement:** 34% of customers churn within 90 days due to impersonal, irrelevant communications.
- **Success Metrics (OKRs):** Retention ≥ 25%↑, Churn ≤ 18%↓, CSAT ≥ 4.2/5
- **MVP Scope:** Sentiment tagging + NBA for top 20% LTV customers
- **Go-to-Market:** Pilot with 500 users → measure → scale
- **Risk:** LLM hallucination in chatbot → mitigated by RAG grounding + fallback rules

---

## 👩‍💼 About

**Tuyet Nguyen** — AI Product Manager  
📧 tuyetnguyen1368.contact@gmail.com  
💼 [LinkedIn](https://linkedin.com/in/tuyetnguyen1368/)  
🐙 [GitHub](https://github.com/tuyetngth2558)

---

*This project uses simulated data for demonstration purposes. All metrics represent model outcomes on synthetic datasets.*
