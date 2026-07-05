# End-to-End AI & Machine Learning Tasks

This repository contains the implementation of multiple advanced AI and Machine Learning tasks completed during the internship program. The project covers end-to-end ML pipelines, multimodal learning, Retrieval-Augmented Generation (RAG), and Large Language Model (LLM) classification.

---

## Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

### 🎯 Objective
To build a reusable, production-ready machine learning pipeline for predicting customer churn using the **Telco Churn Dataset**.

### 🛠️ Methodology & Approach
* **Data Preprocessing:** Implemented feature scaling and categorical encoding seamlessly using the Scikit-learn `Pipeline` API to prevent data leakage.
* **Model Training:** Trained robust classification models including **Logistic Regression** and **Random Forest**.
* **Hyperparameter Tuning:** Optimized model parameters using `GridSearchCV` to find the best performing configuration.
* **Model Export:** Exported the entire end-to-end pipeline into a serialized format (`.joblib`) to ensure production-readiness and instant reusability.

### 📊 Key Results & Observations
* The structured pipeline successfully eliminated manual preprocessing steps during inference.
* The optimized Random Forest model provided the highest classification accuracy and robust handling of feature variance.

---

## Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

### 🎯 Objective
To predict housing prices by integrating and leveraging dual modalities: structured tabular data and house images.

### 🛠️ Methodology & Approach
* **Feature Extraction:** Built a Convolutional Neural Network (CNN) architecture to extract high-level spatial visual features from house images.
* **Feature Fusion:** Combined the extracted visual embeddings with traditional structured tabular features into a unified dataset.
* **Regression Modeling:** Trained a multimodal regression model on the fused data representations to predict final sales prices.
* **Evaluation:** Benchmarked performance strictly using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### 📊 Key Results & Observations
* Combining visual and structured data significantly decreased the prediction error compared to using tabular features alone.

---

## Task 4: Context-Aware Chatbot Using LangChain / RAG

### 🎯 Objective
To build an intelligent, conversational chatbot capable of retaining history and retrieving context-specific answers from an external knowledge base.

### 🛠️ Methodology & Approach
* **Vector Store Integration:** Chunked and embedded a custom document corpus into a vectorized store for semantic similarity searching.
* **Retrieval-Augmented Generation (RAG):** Connected the vector store with an LLM using LangChain to pull accurate, external information dynamically.
* **Context Memory:** Implemented conversational buffer memory to ensure the chatbot tracks history and maintains context across multiple turns.
* **Deployment:** Wrapped the entire application logic inside a clean, user-friendly **Streamlit** web application.

### 📊 Key Results & Observations
* The chatbot successfully mitigates LLM hallucinations by strictly anchoring responses to the retrieved document contexts.

---

## Task 5: Auto Tagging Support Tickets Using LLM

### 🎯 Objective
To automatically classify and tag free-text customer support tickets into distinct functional categories using an LLM.

### 🛠️ Methodology & Approach
* **Prompt Engineering:** Designed specialized system prompts and applied **Few-Shot learning** techniques (providing sample examples) to align the LLM's classification logic.
* **Performance Comparison:** Evaluated and compared the accuracy of Zero-Shot configurations against Few-Shot prompts.
* **Ranking & Output:** Programmed the system to accurately extract and rank the **top 3 most probable tags** for every incoming ticket.

### 📊 Key Results & Observations
* Few-shot prompting drastically improved the categorization accuracy on complex, ambiguous support tickets compared to zero-shot baselines.
