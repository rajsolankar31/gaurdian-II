import pytest
import os
import pandas as pd
import numpy as np
from src.connectors.sqlite_connector import inspect_sqlite_database
from src.advanced_pets import add_laplace_noise, generate_synthetic_dataset

def test_sqlite_inspector():
    db_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_hospital.db")
    res = inspect_sqlite_database(db_path)
    assert res["summary"]["total_tables"] >= 2
    table_names = [t["table_name"] for t in res["tables"]]
    assert "patients" in table_names
    assert "staff" in table_names
    assert len(res["findings"]) > 0

def test_laplace_noise():
    np.random.seed(42)
    s = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    noisy = add_laplace_noise(s, epsilon=1.0)
    assert len(noisy) == len(s)
    # Ensure noise altered values but kept distribution roughly reasonable
    assert not s.equals(noisy)
    assert abs(noisy.mean() - s.mean()) < 200.0

def test_synthetic_data_generator():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "category": ["A", "B", "A", "B", "A"],
        "salary": [50000, 60000, 70000, 80000, 90000]
    })
    synth = generate_synthetic_dataset(df, num_samples=10)
    assert len(synth) == 10
    assert "age" in synth.columns
    assert "category" in synth.columns
    assert "salary" in synth.columns
    assert set(synth["category"].unique()).issubset({"A", "B"})
