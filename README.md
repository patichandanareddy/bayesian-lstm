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

---

## ⚙️ Installation

### A. Local Installation (recommended)
```bash
git clone https://github.com/patichandanareddy/bayesian-lstm.git
cd bayesian-lstm
pip install -e .

B. Direct from GitHub (pip):
pip install git+https://github.com/patichandanareddy/bayesian-lstm.git

###🖥B.Usage Example
from bayeslstm import BayesianLSTM, vectorized_mc_dropout_predict

model = BayesianLSTM(input_size=1, hidden_size=32, output_size=1)
mean, std = vectorized_mc_dropout_predict(model, test_x, T=100)

➡️ See /notebooks/ for full simulations and real-data examples.

### C.Results and Visuals

*Simulated sine wave → model captures waveform and shows credible intervals

*Real-world temperature → competitive performance vs Prophet

*Evaluation metrics → RMSE, MAE, 95% prediction interval

*Visualizations and plots are included in /notebooks/ and /report/.

### D.Testing

Run the unit test to validate the model structure:

python tests/test_models.py

### E.References

*Gal & Ghahramani (2016): Dropout as a Bayesian Approximation

*Jospin et al. (2022): Hands-on Bayesian Neural Networks

*Hochreiter & Schmidhuber (1997): LSTM

*Taylor & Letham (2018): Prophet

### F.License

This project is licensed under the MIT License.

