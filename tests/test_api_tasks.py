def test_create_task(client):
    response = client.post("/tasks", json={"type": "youtube", "source_url": "https://youtube.com/watch?v=test"})
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == "youtube"
    assert data["status"] == "queued"
    assert "id" in data


def test_get_task(client):
    create = client.post("/tasks", json={"type": "audio"})
    task_id = create.json()["id"]
    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["id"] == task_id


def test_get_task_not_found(client):
    response = client.get("/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_list_tasks(client):
    client.post("/tasks", json={"type": "youtube", "source_url": "https://example.com/1"})
    client.post("/tasks", json={"type": "pdf"})
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["tasks"]) == 2


def test_list_tasks_filter_by_type(client):
    client.post("/tasks", json={"type": "youtube", "source_url": "https://example.com/1"})
    client.post("/tasks", json={"type": "pdf"})
    response = client.get("/tasks?type=youtube")
    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_delete_task(client):
    create = client.post("/tasks", json={"type": "audio"})
    task_id = create.json()["id"]
    response = client.delete(f"/tasks/{task_id}")
    assert response.status_code == 204
    get_response = client.get(f"/tasks/{task_id}")
    assert get_response.status_code == 404
