"""
Tests — API Endpoints
=====================
Tests for the FastAPI application using the TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestSystemEndpoints:
    """Tests for health and root endpoints."""

    def test_read_main(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Fraud Detection API is running"
        assert "version" in data

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ("healthy", "unhealthy")

    def test_request_id_header(self):
        response = client.get("/")
        assert "x-request-id" in response.headers

    def test_process_time_header(self):
        response = client.get("/")
        assert "x-process-time" in response.headers


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def _sample_payload(self) -> dict:
        payload = {"Time": 10000.0, "Amount": 150.50}
        for i in range(1, 29):
            payload[f"V{i}"] = 0.1 * i
        return payload

    def test_predict_returns_200_or_503(self):
        """Predict should return 200 (model loaded) or 503 (model not loaded)."""
        response = client.post("/predict", json=self._sample_payload())
        assert response.status_code in (200, 503)

    def test_predict_invalid_input(self):
        """Missing required fields should return 422."""
        response = client.post("/predict", json={"Time": 10000.0})
        assert response.status_code == 422

    def test_predict_batch_invalid(self):
        """Batch endpoint with invalid data."""
        response = client.post("/predict_batch", json=[{"invalid": True}])
        # Should either work (503 no model) or fail gracefully
        assert response.status_code in (200, 422, 503)

    def test_explain_returns_200_or_503(self):
        """Explain should return 200 or 503."""
        response = client.post("/explain", json=self._sample_payload())
        assert response.status_code in (200, 503)
