from backend.compliance import summarize_compliance

def test_no_helmet_is_violation():
    result = summarize_compliance([{"class_id": 1, "class_name": "NoHelmet"}])
    assert result["status"] == "VIOLATION"
    assert "No helmet detected" in result["violations"]

def test_no_vest_is_violation():
    result = summarize_compliance([{"class_id": 2, "class_name": "NoVest"}])
    assert result["status"] == "VIOLATION"
    assert "No safety vest detected" in result["violations"]

def test_empty_detections_is_unknown():
    result = summarize_compliance([])
    assert result["status"] == "UNKNOWN"
    assert result["violations"] == []

def test_ppe_detected_requires_review():
    result = summarize_compliance([{"class_id": 0, "class_name": "Helmet"}])
    assert result["status"] == "REVIEW"
