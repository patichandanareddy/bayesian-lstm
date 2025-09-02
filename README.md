# Bayesian LSTM with MC Dropout (Uncertainty-Aware Forecasting)

## 📌 Overview
This project implements a **Bayesian Long Short-Term Memory (LSTM) network** with **Monte Carlo (MC) Dropout** to perform **time series forecasting with uncertainty estimation**.  
Unlike traditional LSTMs that provide point forecasts, this approach also quantifies **predictive uncertainty** by leveraging dropout at inference time.

The model has been evaluated on **synthetic datasets** as well as **real-world data** (e.g., temperature and financial series).  
Results show that incorporating uncertainty improves decision-making in domains where **forecast confidence** is critical.

---

## ✨ Features
- ✅ **Bayesian LSTM architecture** with MC Dropout  
- ✅ **Uncertainty-aware forecasting** (predictive mean & variance)  
- ✅ **Vectorized Monte Carlo sampling** for efficiency  
- ✅ Comparison against **standard LSTM** and **Prophet**  
- ✅ Modular **PyTorch implementation**  
- ✅ Reproducible experiments with notebooks  

---

## 📂 Project Structure
bayesian-lstm/
│── bayeslstm/ # Core model code
│── notebooks/ # Jupyter notebooks (training & evaluation)
│── data/ # Sample datasets (if included)
│── requirements.txt # Dependencies
│── README.md # Project documentation
