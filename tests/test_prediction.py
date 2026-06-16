import pandas as pd
import pytest

from src.predict import predict_probability, predict_risk


class DummyModel:
    def predict(self, input_data):
        return ["Sedang"] * len(input_data)

    def predict_proba(self, input_data):
        return [[0.1, 0.8, 0.1] for _ in range(len(input_data))]


def test_predict_risk_accepts_dict_input():
    prediction = predict_risk(DummyModel(), {"tahun": 2024})

    assert prediction == "Sedang"


def test_predict_probability_accepts_dataframe_input():
    probabilities = predict_probability(DummyModel(), pd.DataFrame([{"tahun": 2024}]))

    assert probabilities == [0.1, 0.8, 0.1]


def test_predict_probability_requires_supported_model():
    class NoProbabilityModel:
        def predict(self, input_data):
            return ["Sedang"]

    with pytest.raises(AttributeError):
        predict_probability(NoProbabilityModel(), {"tahun": 2024})
