"""Read-only helper: fetch journal pages with cookies and save stripped text to scratch/04."""
import re
import sys
import http.cookiejar
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [("User-Agent", UA), ("Accept-Language", "en")]


def text(html):
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html)


for name, url in (a.split("=", 1) for a in sys.argv[1:]):
    try:
        with opener.open(url, timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
            status = r.status
            final = r.geturl()
    except Exception as e:  # noqa: BLE001
        print(f"{name}: ERR {e}")
        continue
    t = text(body)
    (OUT / f"{name}.txt").write_text(t, encoding="utf-8")
    print(f"{name}: {status} {final} chars={len(t)}")
