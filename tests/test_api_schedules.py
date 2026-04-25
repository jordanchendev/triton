def test_create_schedule(client):
    response = client.post(
        "/schedules",
        json={
            "name": "Daily YouTube Check",
            "cron_expression": "0 8 * * *",
            "type": "youtube",
            "config": {"channel_ids": ["UC123"]},
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Daily YouTube Check"
    assert data["enabled"] is True


def test_get_schedules(client):
    client.post(
        "/schedules",
        json={
            "name": "Job 1",
            "cron_expression": "0 8 * * *",
            "type": "youtube",
            "config": {},
        },
    )
    response = client.get("/schedules")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_update_schedule(client):
    create = client.post(
        "/schedules",
        json={
            "name": "Job 1",
            "cron_expression": "0 8 * * *",
            "type": "youtube",
            "config": {},
        },
    )
    sched_id = create.json()["id"]
    response = client.put(f"/schedules/{sched_id}", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_delete_schedule(client):
    create = client.post(
        "/schedules",
        json={
            "name": "Job 1",
            "cron_expression": "0 8 * * *",
            "type": "youtube",
            "config": {},
        },
    )
    sched_id = create.json()["id"]
    response = client.delete(f"/schedules/{sched_id}")
    assert response.status_code == 204
