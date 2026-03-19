def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "queue_gpu_length" in data
    assert "queue_cpu_length" in data
