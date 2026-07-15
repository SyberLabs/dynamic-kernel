"""
Smoke tests for the FastAPI service.

Run from dynamic_kernel/ directory:
    python -m pytest tests/test_api.py -v

Requires: httpx (pip install httpx)
"""
import pytest
import numpy as np

# TestClient uses httpx under the hood
from fastapi.testclient import TestClient

from api import app, kernel, sim


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/topology
# ---------------------------------------------------------------------------

class TestGetTopology:
    def test_status_200(self, client):
        r = client.get("/api/topology")
        assert r.status_code == 200

    def test_required_fields(self, client):
        data = client.get("/api/topology").json()
        for field in ["labels", "nodesConfig", "distanceMatrix", "adjacencyMask",
                      "N", "F", "featureLabels", "intentPresets", "undirected"]:
            assert field in data, f"missing field: {field}"

    def test_N_matches_labels(self, client):
        data = client.get("/api/topology").json()
        assert data["N"] == len(data["labels"])

    def test_adjacency_mask_shape(self, client):
        data = client.get("/api/topology").json()
        N = data["N"]
        assert len(data["adjacencyMask"]) == N
        assert all(len(row) == N for row in data["adjacencyMask"])

    def test_no_self_loops_in_mask(self, client):
        data = client.get("/api/topology").json()
        mask = data["adjacencyMask"]
        for i, row in enumerate(mask):
            assert row[i] == 0, f"self-loop found at node {i}"

    def test_undirected_flag_present(self, client):
        data = client.get("/api/topology").json()
        assert isinstance(data["undirected"], bool)


# ---------------------------------------------------------------------------
# GET /api/topology/presets
# ---------------------------------------------------------------------------

class TestListPresets:
    def test_status_200(self, client):
        assert client.get("/api/topology/presets").status_code == 200

    def test_known_presets_present(self, client):
        data = client.get("/api/topology/presets").json()
        for preset in ["mall", "airport", "museum", "supply_chain",
                       "wheel_city", "rhizome_city", "social_media"]:
            assert preset in data

    def test_preset_has_meta_fields(self, client):
        data = client.get("/api/topology/presets").json()
        for key, meta in data.items():
            for field in ["key", "name", "description", "icon", "accent",
                          "undirected", "nodeCount", "featureLabels", "intentPresets"]:
                assert field in meta, f"preset '{key}' missing field '{field}'"

    def test_directed_presets_flagged(self, client):
        data = client.get("/api/topology/presets").json()
        assert data["airport"]["undirected"] is False
        assert data["supply_chain"]["undirected"] is False
        assert data["mall"]["undirected"] is True
        assert data["museum"]["undirected"] is True


# ---------------------------------------------------------------------------
# POST /api/topology/load
# ---------------------------------------------------------------------------

class TestLoadTopology:
    def test_load_preset_airport(self, client):
        r = client.post("/api/topology/load", json={"preset": "airport"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["presetName"] == "Airport Terminal"

    def test_load_preset_museum(self, client):
        r = client.post("/api/topology/load", json={"preset": "museum"})
        assert r.status_code == 200

    def test_load_unknown_preset_404(self, client):
        r = client.post("/api/topology/load", json={"preset": "does_not_exist"})
        assert r.status_code == 404

    def test_load_custom_topology(self, client):
        r = client.post("/api/topology/load", json={
            "nodes": {
                "Alpha": [0.9, 0.1, 0.0],
                "Beta":  [0.1, 0.9, 0.0],
                "Gamma": [0.0, 0.1, 0.9],
            },
            "edges": [["Alpha", "Beta", 2.0], ["Beta", "Gamma", 3.0]],
            "undirected": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["N"] == 3
        assert data["F"] == 3

    def test_load_custom_missing_fields_400(self, client):
        r = client.post("/api/topology/load", json={"nodes": {"X": [1, 0, 0]}})
        assert r.status_code == 400

    def test_restore_mall(self, client):
        """Restore default for subsequent tests."""
        r = client.post("/api/topology/load", json={"preset": "mall"})
        assert r.status_code == 200

    def test_session_isolation_between_presets(self, client):
        r1 = client.post("/api/topology/load?session_id=wheel_test", json={"preset": "wheel_city"})
        r2 = client.post("/api/topology/load?session_id=rhizome_test", json={"preset": "rhizome_city"})
        assert r1.status_code == 200
        assert r2.status_code == 200

        wheel = client.get("/api/topology?session_id=wheel_test").json()
        rhizome = client.get("/api/topology?session_id=rhizome_test").json()
        assert wheel["presetName"] == "WHEEL City"
        assert rhizome["presetName"] == "RHIZOME City"
        assert wheel["labels"] != rhizome["labels"]


# ---------------------------------------------------------------------------
# Research metrics endpoints
# ---------------------------------------------------------------------------

class TestResearchEndpoints:
    def test_stationary_distribution_sums_to_one(self, client):
        client.post("/api/topology/load?session_id=stationary_test", json={"preset": "wheel_city"})
        r = client.get(
            "/api/topology/stationary?session_id=stationary_test"
            "&telemetry=0.1,0.8,0.0,0.8"
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["stationary"]) == 12
        assert abs(data["sum"] - 1.0) < 1e-6

    def test_metrics_history_shape(self, client):
        client.post("/api/topology/load?session_id=metrics_test", json={"preset": "mall"})
        r = client.get("/api/metrics/history?session_id=metrics_test")
        assert r.status_code == 200
        history = r.json()["history"]
        for key in [
            "tick", "mean_entropy", "mixing_time", "active_transit_pct",
            "edge_current_norm", "entropy_production",
        ]:
            assert key in history

# ---------------------------------------------------------------------------
# POST /api/diagnostic
# ---------------------------------------------------------------------------

class TestDiagnostic:
    @pytest.fixture(autouse=True)
    def restore_mall(self, client):
        client.post("/api/topology/load", json={"preset": "mall"})

    def _mall_payload(self):
        N = 5
        return {
            "telemetry": [1.0, 0.0, 0.0],
            "beta": [[5.0] * N for _ in range(N)],
            "sponsor_friction": [[0.0] * N for _ in range(N)],
            "node_bias": [0.0] * N,
            "temperature": 1.0,
        }

    def test_status_200(self, client):
        r = client.post("/api/diagnostic", json=self._mall_payload())
        assert r.status_code == 200

    def test_response_fields(self, client):
        data = client.post("/api/diagnostic", json=self._mall_payload()).json()
        for field in ["alignment", "transition_matrix", "row_entropy",
                      "effective_rank", "mixing_time", "edge_flux",
                      "edge_current", "entropy_production", "irreversible_flux"]:
            assert field in data

    def test_transition_matrix_shape(self, client):
        data = client.post("/api/diagnostic", json=self._mall_payload()).json()
        P = data["transition_matrix"]
        assert len(P) == 5
        assert all(len(row) == 5 for row in P)

    def test_row_stochastic(self, client):
        data = client.post("/api/diagnostic", json=self._mall_payload()).json()
        P = np.array(data["transition_matrix"])
        # Connected rows should sum to ~1.0
        row_sums = P.sum(axis=1)
        for i, s in enumerate(row_sums):
            if s > 0:
                assert abs(s - 1.0) < 1e-6, f"row {i} sums to {s}"

    def test_flow_diagnostics_shape(self, client):
        data = client.post("/api/diagnostic", json=self._mall_payload()).json()
        current = np.array(data["edge_current"])
        flux = np.array(data["edge_flux"])
        assert current.shape == (5, 5)
        assert flux.shape == (5, 5)
        np.testing.assert_allclose(current, -current.T, atol=1e-10)
        assert data["entropy_production"] >= -1e-10

    def test_wrong_telemetry_length_400(self, client):
        payload = self._mall_payload()
        payload["telemetry"] = [1.0, 0.0]  # wrong length
        r = client.post("/api/diagnostic", json=payload)
        assert r.status_code == 400

    def test_wrong_beta_shape_400(self, client):
        payload = self._mall_payload()
        payload["beta"] = [[5.0] * 3 for _ in range(3)]  # 3x3 instead of 5x5
        r = client.post("/api/diagnostic", json=payload)
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/export
# ---------------------------------------------------------------------------

class TestExport:
    @pytest.fixture(autouse=True)
    def restore_mall(self, client):
        client.post("/api/topology/load", json={"preset": "mall"})

    def test_json_export_200(self, client):
        r = client.get("/api/export?format=json")
        assert r.status_code == 200

    def test_json_export_structure(self, client):
        data = client.get("/api/export?format=json").json()
        for key in [
            "topology", "simulation", "ensemble_diagnostics",
            "node_analytics", "edge_analytics",
        ]:
            assert key in data

    def test_json_export_ensemble_diagnostics_shape(self, client):
        data = client.get("/api/export?format=json").json()
        ensemble = data["ensemble_diagnostics"]
        for key in [
            "mean_entropy", "mixing_time", "edge_current_norm",
            "entropy_production", "expected_edge_flow", "expected_edge_current",
        ]:
            assert key in ensemble
        N = data["topology"]["N"]
        assert len(ensemble["expected_edge_flow"]) == N
        assert len(ensemble["expected_edge_current"]) == N

    def test_csv_export_200(self, client):
        r = client.get("/api/export?format=csv")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]

    def test_reset_endpoint(self, client):
        r = client.post("/api/export/reset")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
