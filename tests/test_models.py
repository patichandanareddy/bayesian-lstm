from bayeslstm import BayesianLSTM
import torch

def test_model_output_shape():
    model = BayesianLSTM(1, 32, 1)
    dummy_input = torch.randn(4, 30, 1)
    output = model(dummy_input)
    assert output.shape == (4, 1)

if __name__ == "__main__":
    test_model_output_shape()
    print("Test passed!")
