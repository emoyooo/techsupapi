"""
Простые тесты API с httpx.
Запуск: python test_api.py  (сервер должен быть запущен)
"""
import httpx

BASE = "http://localhost:8000"


def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    print("✅ /health:", data)


def test_generate_valid():
    r = httpx.post(f"{BASE}/generate", json={
        "prompt": "I was charged twice this month, please refund."
    })
    assert r.status_code == 200
    data = r.json()
    assert "generated_text" in data
    print("✅ /generate:", data)


def test_generate_empty_prompt():
    r = httpx.post(f"{BASE}/generate", json={"prompt": ""})
    assert r.status_code == 422
    print("✅ Empty prompt → 422:", r.json()["detail"][0]["msg"])


def test_generate_too_long_prompt():
    r = httpx.post(f"{BASE}/generate", json={"prompt": "x" * 3000})
    assert r.status_code == 422
    print("✅ Long prompt → 422:", r.json()["detail"][0]["msg"])


if __name__ == "__main__":
    test_health()
    test_generate_valid()
    test_generate_empty_prompt()
    test_generate_too_long_prompt()
    print("\n✅ Все тесты пройдены")
