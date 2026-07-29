"""HTML landing page and favicon for the --serve HTTP server."""

# A dragon emoji rendered as SVG text works as a favicon with zero image
# assets to ship, and doubles as the landing page's hero icon.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<text x="32" y="50" font-size="52" text-anchor="middle">\U0001F409</text>'
    "</svg>"
)

_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dragon CLI — __MODEL_NAME__</title>
<link rel="icon" type="image/svg+xml" href="/favicon.ico">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2.5rem 1.5rem 4rem;
    background: #0b0f14; color: #d7e0ea;
    font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
    line-height: 1.55;
  }
  main { max-width: 720px; margin: 0 auto; }
  header { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem; }
  .dragon { font-size: 3rem; line-height: 1; }
  h1 { margin: 0; font-size: 1.6rem; color: #eef4fa; }
  .subtitle { margin: 0.15rem 0 0; color: #7fe0d1; font-size: 0.95rem; }
  .card {
    background: #121924; border: 1px solid #212c3a; border-radius: 10px;
    padding: 1rem 1.25rem; margin: 1.75rem 0;
  }
  .card h2 { margin-top: 0; font-size: 1.05rem; color: #7fe0d1; }
  .kv { display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 1rem; font-size: 0.92rem; }
  .kv dt { color: #8a97a8; }
  .kv dd { margin: 0; font-family: ui-monospace, Consolas, monospace; word-break: break-all; }
  pre {
    background: #0d1420; border: 1px solid #1c2634; border-radius: 8px;
    padding: 0.9rem 1rem; overflow-x: auto; font-size: 0.85rem;
    color: #c7e8ff;
  }
  code { font-family: ui-monospace, Consolas, monospace; }
  a { color: #67c9ff; }
  footer { margin-top: 2.5rem; color: #62707f; font-size: 0.85rem; }
</style>
</head>
<body>
<main>
  <header>
    <span class="dragon">\U0001F409</span>
    <div>
      <h1>Dragon CLI</h1>
      <p class="subtitle">local model server — OpenAI-compatible API</p>
    </div>
  </header>

  <div class="card">
    <dl class="kv">
      <dt>Model</dt><dd>__MODEL_NAME__</dd>
      <dt>Base URL</dt><dd>__BASE_URL__</dd>
    </dl>
  </div>

  <div class="card">
    <h2>Interactive API docs</h2>
    <p><a href="/docs">Swagger UI</a> &middot; <a href="/redoc">ReDoc</a> &middot;
       <a href="/openapi.json">OpenAPI schema</a></p>
  </div>

  <div class="card">
    <h2>List models</h2>
    <pre><code>curl __BASE_URL__/v1/models</code></pre>
  </div>

  <div class="card">
    <h2>Chat completion</h2>
    <pre><code>curl __BASE_URL__/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'</code></pre>
    <p>Add <code>"stream": true</code> to the JSON body for server-sent-events streaming.</p>
  </div>

  <div class="card">
    <h2>Python (OpenAI SDK)</h2>
    <pre><code>from openai import OpenAI

client = OpenAI(base_url="__BASE_URL_SLASH_V1__", api_key="not-needed")
resp = client.chat.completions.create(
    model="__MODEL_NAME__",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)</code></pre>
  </div>

  <footer>No MCP tool-calling on this endpoint — model only. Started with <code>dragon-cli --serve</code>.</footer>
</main>
</body>
</html>
"""


def render_index(model_name: str, base_url: str) -> str:
    base = base_url.rstrip("/")
    return (
        _INDEX_TEMPLATE
        .replace("__MODEL_NAME__", model_name)
        .replace("__BASE_URL_SLASH_V1__", f"{base}/v1")
        .replace("__BASE_URL__", base)
    )
