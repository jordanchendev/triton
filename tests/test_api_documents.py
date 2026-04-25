def test_create_document(client):
    response = client.post(
        "/documents",
        json={
            "type": "web",
            "title": "Market Analysis",
            "content": "S&P 500 rose 2% today...",
            "source_url": "https://example.com/article",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "S&P 500 rose 2% today..."
    assert "id" in data


def test_get_document(client):
    create = client.post("/documents", json={"type": "tweet", "content": "BTC to 100k"})
    doc_id = create.json()["id"]
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["content"] == "BTC to 100k"


def test_get_document_not_found(client):
    response = client.get("/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_list_documents(client):
    client.post("/documents", json={"type": "web", "content": "Article 1"})
    client.post("/documents", json={"type": "tweet", "content": "Tweet 1"})
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_list_documents_filter_by_type(client):
    client.post("/documents", json={"type": "web", "content": "Article 1"})
    client.post("/documents", json={"type": "tweet", "content": "Tweet 1"})
    response = client.get("/documents?type=web")
    assert response.status_code == 200
    assert response.json()["total"] == 1
