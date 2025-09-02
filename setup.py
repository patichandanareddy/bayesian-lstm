from setuptools import setup, find_packages

setup(
    name='bayeslstm',
    version='0.1',
    description='Bayesian LSTM with Monte Carlo Dropout for Forecasting',
    author='Mahidhar Reddy Patukuri, Dimple Alekya Basimi, Chandana Pati',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'pandas',
        'scikit-learn'
    ],
)
