import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from eqpred.ui.server import LabHandler


@pytest.fixture()
def base_url():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), LabHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _call(url, data=None):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_a_bad_date_gets_a_json_error_not_a_dropped_connection(base_url):
    status, body = _call(f"{base_url}/api/day?date=not-a-date")
    assert status == 400
    assert "not-a-date" in body["error"]


def test_an_unknown_api_path_is_a_json_404(base_url):
    status, body = _call(f"{base_url}/api/nope")
    assert status == 404
    assert body["error"]


def test_a_malformed_save_request_is_a_400(base_url):
    status, body = _call(f"{base_url}/api/saved-strategies", data=b"{not json")
    assert status == 400
    assert body["error"]


def test_training_and_removing_cards_need_a_post(base_url):
    status, body = _call(f"{base_url}/api/learn?train=1")
    assert status == 405
    status, body = _call(f"{base_url}/api/saved-strategies?drop=anything")
    assert status == 405


def _post(url, body, headers):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_another_site_cannot_change_anything(base_url):
    url = f"{base_url}/api/saved-strategies"
    body = b'{"action": "drop", "id": "nothing"}'
    assert _post(url, body, {"Content-Type": "text/plain"}) == 403
    assert _post(url, body, {"Content-Type": "application/json", "Origin": "http://evil.example"}) == 403
    request = urllib.request.Request(f"{base_url}/api/tapes", headers={"Host": "evil.example:80"})
    try:
        urllib.request.urlopen(request, timeout=30)
        status = 200
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 403


def test_large_answers_are_compressed_when_the_browser_asks(base_url):
    import gzip

    request = urllib.request.Request(f"{base_url}/api/hunt?sequence=1", headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        encoding = response.headers.get("Content-Encoding")
    payload = json.loads(gzip.decompress(raw)) if encoding == "gzip" else json.loads(raw)
    assert isinstance(payload, dict)
    if len(json.dumps(payload)) > 1024:
        assert encoding == "gzip"
