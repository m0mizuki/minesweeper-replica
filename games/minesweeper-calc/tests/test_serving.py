"""Check the browser module loading boundary through the documented HTTP server."""

from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from threading import Thread
import unittest
from urllib.parse import urljoin
from urllib.request import urlopen


APP_ROOT = Path(__file__).resolve().parents[1]
JAVASCRIPT_TYPES = {"text/javascript", "application/javascript"}


class ModuleScripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("type") == "module" and "src" in attrs:
            self.sources.append(attrs["src"])


class WindowsStaticHandler(SimpleHTTPRequestHandler):
    # Reproduce Windows installations where .mjs is registered as plain text.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/plain"}

    def log_message(self, *_args):
        pass


class StaticServingTests(unittest.TestCase):
    def test_entry_and_all_imports_are_served_as_javascript(self):
        for root, path in ((APP_ROOT, "/"), (APP_ROOT.parents[1], "/games/minesweeper-calc/")):
            with self.subTest(root=root):
                handler = partial(WindowsStaticHandler, directory=str(root))
                with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
                    thread = Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        base = f"http://127.0.0.1:{server.server_port}{path}"
                        with urlopen(base, timeout=5) as response:
                            parser = ModuleScripts()
                            parser.feed(response.read().decode("utf-8"))
                        self.assertTrue(parser.sources, "HTML must load an application module")
                        pending = [urljoin(base, src) for src in parser.sources]
                        visited = set()
                        while pending:
                            url = pending.pop()
                            if url in visited:
                                continue
                            visited.add(url)
                            with urlopen(url, timeout=5) as response:
                                self.assertIn(response.headers.get_content_type(), JAVASCRIPT_TYPES, url)
                                source = response.read().decode("utf-8")
                            imports = re.findall(r'\bfrom\s+["\']([^"\']+)["\']', source)
                            pending.extend(urljoin(url, name) for name in imports if name.startswith("."))
                        self.assertEqual(len(visited), 4, "entry plus BP, Gibbs, and math modules")
                    finally:
                        server.shutdown()
                        thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
