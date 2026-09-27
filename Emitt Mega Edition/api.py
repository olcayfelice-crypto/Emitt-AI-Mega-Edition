
import logging

from flask import Flask, request, jsonify

from config import EMITT_NAME, EMITT_VERSION, HOST, PORT
from emitt_engine import EmittEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("emitt.api")

app = Flask(__name__)

# Motor sunucu açılırken bir kez yüklenir. Model dosyaları henüz
# yoksa (training.py hiç çalıştırılmadıysa) engine.ready False olur
# ama sunucu yine de ayağa kalkar; /api/chat o zaman 503 döner.
engine = EmittEngine()

if engine.ready:
    logger.info("Emitt motoru hazır. Vocabulary: %s", engine.tokenizer.vocabulary_size)
else:
    logger.warning("Emitt motoru hazır değil: %s", engine.load_error)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def home():
    return jsonify({
        "name": EMITT_NAME,
        "version": EMITT_VERSION,
        "status": "online",
    })


@app.get("/api/status")
def status():
    payload = {
        "name": EMITT_NAME,
        "status": "online",
        "engine": engine.name,
        "engine_version": engine.version,
        "engine_status": "ready" if engine.ready else "not_installed",
    }

    if engine.ready:
        payload["vocabulary_size"] = engine.tokenizer.vocabulary_size
        payload["context_size"] = engine.model.context_size
    else:
        payload["engine_error"] = engine.load_error

    return jsonify(payload)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON body gerekli."}), 400

    message = data.get("message")

    if not isinstance(message, str):
        return jsonify({"error": "message alanı string olmalıdır."}), 400

    if not message.strip():
        return jsonify({"error": "Mesaj boş olamaz."}), 400

    use_internet = data.get("use_internet")

    if use_internet is not None and not isinstance(use_internet, bool):
        return jsonify({"error": "use_internet alanı boolean olmalıdır."}), 400

    if not engine.ready:
        return jsonify({
            "model": EMITT_NAME,
            "message": message,
            "response": None,
            "status": "engine_not_ready",
            "error": engine.load_error,
        }), 503

    try:
        result = engine.generate(message, use_internet=use_internet)

    except Exception as error:
        logger.exception("Cevap üretilirken hata oluştu")
        return jsonify({"error": f"Cevap üretilemedi: {error}"}), 500

    return jsonify({
        "model": EMITT_NAME,
        "message": message,
        "response": result["response"],
        "source": result["source"],
        "status": "ok",
    })


@app.post("/api/reload")
def reload_engine():
    """training.py'yi yeniden çalıştırdıktan sonra sunucuyu yeniden
    başlatmadan yeni model ağırlıklarını yükler."""

    success = engine.reload()

    status_code = 200 if success else 503

    return jsonify({
        "status": "ready" if success else "not_installed",
        "error": engine.load_error,
    }), status_code


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
