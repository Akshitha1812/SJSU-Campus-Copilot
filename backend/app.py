import os
import tempfile
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from metrics import with_metrics, mount_metrics
from rag import add_docs_from_pdf, ingest_any_file, answer_query, bootstrap_data_dir, list_sources, preview_retrieval
from werkzeug.exceptions import HTTPException

load_dotenv()

app = Flask(__name__, static_folder="../static",
            template_folder="../templates")
mount_metrics(app)

# ---------- JSON errors instead of HTML ----------


@app.errorhandler(Exception)
def handle_any_error(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    return jsonify({"error": str(e)}), 500


@app.get("/sources")
def sources():
    return jsonify({"sources": list_sources()})


@app.post("/debug-retrieve")
def debug_retrieve():
    data = request.get_json(silent=True) or {}
    q = data.get("query", "")
    return jsonify({"query": q, "top": preview_retrieval(q, n=8)})


@app.get("/")
def home():
    default_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data"))
    return render_template("index.html", default_dir=default_dir)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})

# ---------- Ingest endpoints ----------


@app.post("/ingest")
def ingest_pdf():
    if "file" not in request.files:
        return jsonify({"error": "file required"}), 400
    f = request.files["file"]
    suffix = os.path.splitext(f.filename)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    n = ingest_any_file(tmp.name)  # <- writes to Chroma, idempotent
    return jsonify({"indexed_chunks": n})


@app.post("/ingest-batch")
def ingest_batch():

    if "files" not in request.files:
        return jsonify({"error": "files[] required"}), 400
    files = request.files.getlist("files")
    total = 0
    # for f in files:
    #     suffix = os.path.splitext(f.filename)[1] or ""
    #     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    #     f.save(tmp.name)
    #     total += ingest_any_file(tmp.name)
    # return jsonify({"indexed_chunks": total})

    results = []
    for f in files:
        suffix = os.path.splitext(f.filename)[1] or ""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            f.save(tmp.name)
            added = ingest_any_file(
                tmp.name, source_name=f.filename)  # <- real name
            results.append({"file": f.filename, "indexed_chunks": added})
        finally:
            try:
                os.unlink(tmp.name)
            except:
                pass
    return jsonify({
        "results": results,
        "total_indexed_chunks": sum(r["indexed_chunks"] for r in results)
    })


@app.post("/ingest-dir")
def ingest_dir():
    data = request.get_json(silent=True) or {}
    folder = data.get("dir")
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "valid 'dir' path required"}), 400
    stats = bootstrap_data_dir(folder)  # <- idempotent folder ingest
    return jsonify(stats)

# ---------- Ask ----------


@app.post("/ask")
@with_metrics("ask")
def ask():
    data = request.get_json(silent=True) or {}
    q = data.get("query") or data.get("question") or data.get("q")
    if not isinstance(q, str) or not q.strip():
        return jsonify({"error": "query required"}), 400
    return jsonify(answer_query(q))


if __name__ == "__main__":
    # One-time bootstrap for /data on startup (idempotent; safe with reload)
    DEFAULT_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data"))
    stats = bootstrap_data_dir(DEFAULT_DIR)
    print("Bootstrap:", stats)
    app.run(host="0.0.0.0", debug=True, port=8080)
