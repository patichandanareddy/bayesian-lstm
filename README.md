# Bayesian LSTM for Uncertainty-Aware Time Series Forecasting

This project implements a Bayesian version of Long Short-Term Memory (LSTM) using Monte Carlo (MC) Dropout to estimate predictive uncertainty in time series forecasting. It supports efficient vectorized inference, real and simulated data, and comparative analysis with traditional models like standard LSTM and Prophet.

---

##  Features

- ✅ Bayesian LSTM using MC Dropout
- ✅ Vectorized MC sampling for fast inference
- ✅ Predictive mean and uncertainty bounds (95% confidence intervals)
- ✅ Comparative analysis with standard LSTM and Prophet
- ✅ Reproducible experiments and Jupyter notebooks
- ✅ Pip-installable Python package

---

## Project Structure

bayesian-lstm/
├── bayeslstm/ # Python package
├── notebooks/ # Experiments and visualizations
├── report/ # Final report (.docx, .pdf)
├── tests/ # Unit tests
├── requirements.txt # Dependencies
├── setup.py # Package installer
└── README.md # This file

Installation

 A. Local Installation (recommended)

```bash

git clone git@gitlab2.tamucc.edu:cpati/bayesian-lstm.git
cd bayesian-lstm
pip install -e .

 ### B. Direct from GitLab (pip)

pip install git+https://gitlab2.tamucc.edu/cpati/bayesian-lstm.git


---

## Usage Example

```markdown
## Usage

```python
from bayeslstm import BayesianLSTM, vectorized_mc_dropout_predict

model = BayesianLSTM(input_size=1, hidden_size=32, output_size=1)
mean, std = vectorized_mc_dropout_predict(model, test_x, T=100)


See /notebooks/ for full simulations and real-data examples.

---

###  **Results and Visuals**

```markdown
## Results

- Simulated sine wave: model captures waveform and shows credible intervals
- Real-world temperature: model performs competitively with Prophet
- Evaluation metrics: RMSE, MAE, 95% prediction interval

Visualizations and plots are included in `notebooks/` and `report/`.

## Testing

Run the unit test to validate model structure:

```bash
python tests/test_models.py


---

## **References**

```markdown
## References

- Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
- Jospin et al. (2022): Hands-on Bayesian Neural Networks
- Hochreiter & Schmidhuber (1997): LSTM
- Taylor & Letham (2018): Prophet

## License

This project is licensed under the MIT License.

