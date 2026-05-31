"""
TTS Reader Server — run with: ./start.sh (uses GPU)
Or: python server.py (falls back to CPU, slower)
Then open http://localhost:5000 in a browser.
"""
import json
import io
import math
import os
import re
import sqlite3
import subprocess
import threading
import time
import tempfile
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, send_file, abort

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

LIBRARY_DIR = Path.home() / "book_reader" / "library"
AUDIO_DIR   = LIBRARY_DIR / "audio"
SYNC_DIR    = LIBRARY_DIR / "sync"
SAMPLES_DIR = LIBRARY_DIR / "samples"
SOURCES_DIR = LIBRARY_DIR / "sources"
MODELS_DIR  = LIBRARY_DIR / "models"
DB_PATH     = LIBRARY_DIR / "library.db"

for d in (LIBRARY_DIR, AUDIO_DIR, SYNC_DIR, SAMPLES_DIR, SOURCES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

HTML_PATH = Path(__file__).resolve().parent / "frontend.html"

SAMPLE_TEXT = "In the still hours before dawn, the city held its breath."

VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bm_george",
]



# ---------- DB ----------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source_format TEXT NOT NULL,
            voice TEXT NOT NULL,
            total_duration REAL,
            word_count INTEGER,
            last_word_index INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'processing',
            eta_seconds REAL,
            processing_started_at REAL,
            error TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """)
        conn.execute(
            "UPDATE books SET status='failed', error='Server restarted during processing' "
            "WHERE status='processing'"
        )


init_db()


# ---------- CORS ----------

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Methods'] = 'POST, GET, PATCH, DELETE, OPTIONS'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return r


# ---------- text extraction ----------

def read_txt(data):
    return data.decode('utf-8', errors='ignore')


def read_docx(data):
    import docx
    doc = docx.Document(io.BytesIO(data))
    return '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())


def read_doc(data):
    import subprocess
    with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        result = subprocess.run(['antiword', tmp], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("antiword failed: " + result.stderr)
        return result.stdout
    finally:
        os.unlink(tmp)


def read_epub(data):
    import ebooklib
    from ebooklib import epub
    from html.parser import HTMLParser

    class Extractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
            self.skip = False
        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style'):
                self.skip = True
        def handle_endtag(self, tag):
            if tag in ('script', 'style'):
                self.skip = False
        def handle_data(self, data):
            if not self.skip:
                self.text.append(data)

    with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        book = epub.read_epub(tmp)
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                p = Extractor()
                p.feed(item.get_content().decode('utf-8', errors='ignore'))
                t = ' '.join(' '.join(p.text).split())
                if t.strip():
                    texts.append(t)
        return '\n\n'.join(texts)
    finally:
        os.unlink(tmp)


def extract_text(data, ext):
    if ext == '.txt':   return read_txt(data)
    if ext == '.docx':  return read_docx(data)
    if ext == '.doc':   return read_doc(data)
    if ext == '.epub':  return read_epub(data)
    raise Exception(f"Unsupported format: {ext}")


# ---------- Kokoro TTS ----------

_settled_n = None        # safe concurrent worker count (int), determined once by probe
_settled_lock = threading.Lock()
_use_gpu = None          # bool, set during probe
_synthesis_lock = threading.Lock()  # espeak is not thread-safe — only one synthesis at a time

MAX_WORKERS = 3

_PROBE_TEXT = (
    "In the still hours before dawn, the city held its breath "
    "with a quiet determination that only the long dark night could understand. "
    "The empty streets stretched on, silent and wide, beneath a pale and heavy sky."
)


def get_model_paths():
    model  = MODELS_DIR / "kokoro-v1.0.onnx"
    voices = MODELS_DIR / "voices-v1.0.bin"
    if not model.exists() or not voices.exists():
        raise FileNotFoundError(
            f"Kokoro model files not found in {MODELS_DIR}\n"
            "Download with:\n"
            "  wget https://github.com/thewh1teagle/kokoro-onnx/releases/"
            "download/model-files-v1.0/kokoro-v1.0.onnx "
            f"-P {MODELS_DIR}\n"
            "  wget https://github.com/thewh1teagle/kokoro-onnx/releases/"
            "download/model-files-v1.0/voices-v1.0.bin "
            f"-P {MODELS_DIR}"
        )
    return str(model), str(voices)


def _make_session(model_path, providers):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0 if 'CUDAExecutionProvider' in providers else 1
    if 'CUDAExecutionProvider' in providers:
        cuda_opts = {"cudnn_conv_algo_search": "HEURISTIC"}
        providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers, sess_options=opts)


def _vram_used_mb():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / 1024 / 1024
    except Exception:
        return None


def _create_workers(n, providers, model_path, voices_path):
    """Spin up n fresh Kokoro sessions. Always creates new ONNX sessions (clean BFC arenas)."""
    from kokoro_onnx import Kokoro
    workers = []
    for i in range(n):
        before = _vram_used_mb()
        sess = _make_session(model_path, providers)
        workers.append(Kokoro.from_session(sess, voices_path))
        after = _vram_used_mb()
        if before is not None and after is not None:
            print(f"  session {i+1}/{n}: +{after - before:.0f} MB  (total {after:.0f} MB used)")
    return workers


def _probe_concurrent(workers, probe_phonemes):
    """Run inference on all workers simultaneously. Returns None on success, exception on OOM."""
    errors = [None] * len(workers)

    def run(idx, w):
        try:
            w.create(probe_phonemes, voice="af_heart", is_phonemes=True, speed=1.0, lang="en-us")
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=run, args=(i, w)) for i, w in enumerate(workers)]
    for t in threads: t.start()
    for t in threads: t.join()

    return next((e for e in errors if e is not None), None)


def get_settled_n():
    """Return the safe concurrent worker count, probing once at first call."""
    import gc
    global _settled_n, _use_gpu

    if _settled_n is not None:
        return _settled_n

    with _settled_lock:
        if _settled_n is not None:
            return _settled_n

        from kokoro_onnx import Kokoro

        model_path, voices_path = get_model_paths()
        available = ort.get_available_providers()

        if 'CUDAExecutionProvider' not in available:
            print("GPU not detected — using CPU (launch via ./start.sh for GPU)")
            _use_gpu = False
            _settled_n = 1
            print("Kokoro ready (1 worker, CPU)")
            return _settled_n

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _use_gpu = True
        print("GPU detected — probing safe worker count...")

        probe_workers = []
        probe_phonemes = None
        safe_count = 0

        for attempt in range(1, MAX_WORKERS + 1):
            try:
                sess = _make_session(model_path, providers)
                candidate = Kokoro.from_session(sess, voices_path)
            except Exception as e:
                if "Failed to allocate memory" in str(e):
                    print(f"  {attempt} worker(s) → OOM at load, settling on {safe_count}")
                    break
                raise

            probe_workers.append(candidate)

            if probe_phonemes is None:
                raw = candidate.tokenizer.phonemize(_PROBE_TEXT, 'en-us')
                probe_phonemes = _split_phoneme_string(raw)[0]
                print(f"  probe phoneme length: {len(probe_phonemes)} chars")

            err = _probe_concurrent(probe_workers, probe_phonemes)
            if err is not None:
                probe_workers.pop()
                print(f"  {attempt} worker(s) → OOM at inference, settling on {safe_count}")
                break

            safe_count = attempt
            print(f"  {attempt} worker(s) → OK")

            if attempt == MAX_WORKERS:
                print(f"  hit MAX_WORKERS cap ({MAX_WORKERS})")

        # Discard all probe sessions — synthesis creates its own fresh sessions each time
        del probe_workers
        gc.collect()

        if safe_count == 0:
            raise RuntimeError("Even a single GPU worker OOMed during probe — check VRAM")

        _settled_n = safe_count
        print(f"Kokoro ready ({safe_count} worker(s), GPU)")
        return _settled_n


def _preload():
    try:
        get_settled_n()
    except Exception as e:
        print(f"Kokoro preload failed: {e}")

threading.Thread(target=_preload, daemon=True).start()


MAX_PHONEME_CHARS = 150

def split_sentences(text):
    text = re.sub(r'\s+', ' ', text.strip())
    parts = re.split(r'(?<=[.!?…])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _split_phoneme_string(p):
    """Split a phoneme string on word boundaries so no chunk exceeds MAX_PHONEME_CHARS."""
    if len(p) <= MAX_PHONEME_CHARS:
        return [p]
    words = p.split(' ')
    chunks, current, current_len = [], [], 0
    for w in words:
        add = (1 if current else 0) + len(w)
        if current and current_len + add > MAX_PHONEME_CHARS:
            chunks.append(' '.join(current))
            current, current_len = [w], len(w)
        else:
            current.append(w)
            current_len += add
    if current:
        chunks.append(' '.join(current))
    return chunks



def synthesize(text, voice):
    import lameenc, gc
    global _settled_n, _use_gpu

    n = get_settled_n()
    model_path, voices_path = get_model_paths()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if _use_gpu else ["CPUExecutionProvider"])

    sentences = split_sentences(text)
    if not sentences:
        raise Exception("No text to synthesize")

    workers = None
    try:
        # Create fresh sessions — clean BFC arenas, no carry-over from previous book
        workers = _create_workers(n, providers, model_path, voices_path)

        # Phonemize in main thread (espeak not thread-safe); done once, outside retry loop
        phonemized = []
        for s in sentences:
            p = workers[0].tokenizer.phonemize(s, 'en-us')
            phonemized.append((s, _split_phoneme_string(p)))
        while True:
            k, m = divmod(len(phonemized), n)
            groups, start = [], 0
            for i in range(n):
                size = k + (1 if i < m else 0)
                if size:
                    groups.append(phonemized[start:start + size])
                start += size

            actual_n = len(groups)
            chunk_results = [None] * actual_n
            chunk_errors  = [None] * actual_n

            def run_chunk(cidx, group, worker):
                try:
                    enc = lameenc.Encoder()
                    enc.set_bit_rate(48)
                    enc.set_in_sample_rate(24000)
                    enc.set_channels(1)
                    enc.set_quality(7)

                    mp3_parts = []
                    timings   = []
                    chunk_time = 0.0

                    for s, p_chunks in group:
                        pieces = []
                        for p_chunk in p_chunks:
                            audio, _ = worker.create(
                                p_chunk, voice=voice, is_phonemes=True, speed=1.0, lang='en-us'
                            )
                            pieces.append(audio)

                        seg = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                        dur = len(seg) / 24000
                        pcm = np.clip(seg * 32767, -32768, 32767).astype(np.int16)
                        mp3_parts.append(enc.encode(pcm.tobytes()))
                        del seg, pieces, pcm

                        ws = s.split()
                        if ws:
                            total = sum(len(w) for w in ws)
                            t = chunk_time
                            for w in ws:
                                wd = len(w) / max(total, 1) * dur
                                timings.append({"word": w, "start": t, "end": t + wd})
                                t += wd
                        chunk_time += dur

                    mp3_parts.append(enc.flush())
                    chunk_results[cidx] = (b''.join(mp3_parts), timings, chunk_time)
                except Exception as e:
                    chunk_errors[cidx] = e

            threads = [
                threading.Thread(target=run_chunk, args=(i, groups[i], workers[i]))
                for i in range(actual_n)
            ]
            for t in threads: t.start()
            for t in threads: t.join()

            oom = next((e for e in chunk_errors if e is not None
                        and "Failed to allocate memory" in str(e)), None)
            other = next((e for e in chunk_errors if e is not None
                          and "Failed to allocate memory" not in str(e)), None)

            if other:
                raise other

            if oom:
                with _settled_lock:
                    if n <= 1:
                        raise RuntimeError(
                            "OOM with a single worker — VRAM too low for this input"
                        ) from oom
                    n -= 1
                    _settled_n = n
                    print(f"OOM during synthesis — reducing to {n} worker(s) and retrying")
                # Replace sessions with a fresh, smaller set
                workers = None
                gc.collect()
                workers = _create_workers(n, providers, model_path, voices_path)
                continue

            # Success — stitch chunks in book order
            final_mp3     = []
            final_timings = []
            time_offset   = 0.0
            for mp3_bytes, timings, chunk_dur in chunk_results:
                final_mp3.append(mp3_bytes)
                for w in timings:
                    final_timings.append({
                        "word":  w["word"],
                        "start": w["start"] + time_offset,
                        "end":   w["end"]   + time_offset,
                    })
                time_offset += chunk_dur

            return b''.join(final_mp3), final_timings

    finally:
        if workers is not None:
            del workers
        gc.collect()


# ---------- calibration ----------

def get_sec_per_word(conn):
    rows = conn.execute(
        "SELECT key, value FROM meta WHERE key IN ('total_seconds', 'total_words')"
    ).fetchall()
    d = {r["key"]: float(r["value"]) for r in rows}
    if d.get("total_words", 0) > 0:
        return d["total_seconds"] / d["total_words"]
    return 0.006   # GPU default: ~166 words/sec


def update_calibration(conn, word_count, elapsed):
    def upsert(k, v):
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (k, str(v))
        )
    cur = conn.execute(
        "SELECT key,value FROM meta WHERE key IN ('total_seconds','total_words')"
    ).fetchall()
    d = {r["key"]: float(r["value"]) for r in cur}
    upsert('total_seconds', d.get('total_seconds', 0) + elapsed)
    upsert('total_words',   d.get('total_words',   0) + word_count)


# ---------- background processing ----------

def process_book(book_id, voice):
    try:
        text = (SOURCES_DIR / f"{book_id}.txt").read_text(encoding='utf-8')
        t0   = time.time()
        with _synthesis_lock:
            audio, words = synthesize(text, voice)
        elapsed = time.time() - t0
        total_duration = words[-1]["end"] if words else 0.0
        word_count = len(text.split())
        print(f"[book {book_id}] synthesized {word_count} words / {total_duration:.1f}s audio in {elapsed:.1f}s real time ({total_duration/elapsed:.1f}x RTF)")

        (AUDIO_DIR / f"{book_id}.mp3").write_bytes(audio)
        (SYNC_DIR  / f"{book_id}.json").write_text(json.dumps({"words": words}))

        with db() as conn:
            conn.execute(
                "UPDATE books SET status='ready', total_duration=?, "
                "eta_seconds=NULL, processing_started_at=NULL, error=NULL WHERE id=?",
                (total_duration, book_id)
            )
            update_calibration(conn, len(text.split()), elapsed)
    except Exception as e:
        traceback.print_exc()
        with db() as conn:
            conn.execute(
                "UPDATE books SET status='failed', eta_seconds=NULL, "
                "processing_started_at=NULL, error=? WHERE id=?",
                (str(e), book_id)
            )


# ---------- routes ----------

@app.route('/')
def index():
    return send_file(HTML_PATH)


@app.route('/api/books', methods=['GET'])
def list_books():
    with db() as conn:
        rows = conn.execute("SELECT * FROM books ORDER BY created_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    if not row:
        abort(404)
    return jsonify(dict(row))


@app.route('/api/books', methods=['POST', 'OPTIONS'])
def create_book():
    if request.method == 'OPTIONS':
        return '', 200
    file  = request.files.get('file')
    voice = request.form.get('voice', 'af_heart')
    if not file:
        return jsonify({"error": "No file"}), 400
    if voice not in VOICES:
        return jsonify({"error": "Unknown voice"}), 400

    name = Path(file.filename).stem
    ext  = Path(file.filename).suffix.lower()
    data = file.read()

    try:
        text = extract_text(data, ext)
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {e}"}), 400

    if not text.strip():
        return jsonify({"error": "Empty document"}), 400

    word_count = len(text.split())

    with db() as conn:
        sec_per_word = get_sec_per_word(conn)
        eta = word_count * sec_per_word
        cur = conn.execute(
            "INSERT INTO books(title, source_format, voice, word_count, eta_seconds, "
            "status, processing_started_at, created_at) "
            "VALUES (?,?,?,?,?,'processing',?,datetime('now'))",
            (name, ext.lstrip('.'), voice, word_count, eta, time.time())
        )
        book_id = cur.lastrowid
        row = dict(conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone())

    (SOURCES_DIR / f"{book_id}.txt").write_text(text, encoding='utf-8')
    threading.Thread(target=process_book, args=(book_id, voice), daemon=True).start()
    return jsonify(row)


@app.route('/api/books/<int:book_id>', methods=['PATCH'])
def update_book(book_id):
    data = request.get_json(silent=True) or {}
    fields, args = [], []
    if 'title' in data:
        title = str(data['title']).strip()
        if not title:
            return jsonify({"error": "Title cannot be empty"}), 400
        fields.append("title=?"); args.append(title)
    if 'last_word_index' in data:
        fields.append("last_word_index=?"); args.append(int(data['last_word_index']))
    if not fields:
        return jsonify({"error": "Nothing to update"}), 400
    args.append(book_id)
    with db() as conn:
        conn.execute(f"UPDATE books SET {', '.join(fields)} WHERE id=?", args)
        row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    if not row:
        abort(404)
    return jsonify(dict(row))


@app.route('/api/books/<int:book_id>/position', methods=['POST'])
def update_position(book_id):
    idx = None
    if request.is_json:
        idx = (request.get_json(silent=True) or {}).get('idx')
    if idx is None:
        raw = request.get_data(as_text=True)
        if raw:
            try:
                idx = json.loads(raw).get('idx')
            except Exception:
                pass
    if idx is None:
        idx = request.args.get('idx')
    if idx is None:
        return jsonify({"error": "no idx"}), 400
    with db() as conn:
        conn.execute("UPDATE books SET last_word_index=? WHERE id=?", (int(idx), book_id))
    return jsonify({"ok": True})


@app.route('/api/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    with db() as conn:
        conn.execute("DELETE FROM books WHERE id=?", (book_id,))
    for p in (
        AUDIO_DIR  / f"{book_id}.mp3",
        SYNC_DIR   / f"{book_id}.json",
        SOURCES_DIR / f"{book_id}.txt",
    ):
        if p.exists():
            p.unlink()
    return jsonify({"ok": True})


@app.route('/api/books/<int:book_id>/regenerate', methods=['POST'])
def regenerate(book_id):
    data  = request.get_json(silent=True) or {}
    voice = data.get('voice')
    if voice not in VOICES:
        return jsonify({"error": "Unknown voice"}), 400
    if not (SOURCES_DIR / f"{book_id}.txt").exists():
        return jsonify({"error": "Source not available"}), 404

    with db() as conn:
        row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        sec_per_word = get_sec_per_word(conn)
        eta = (row['word_count'] or 0) * sec_per_word
        conn.execute(
            "UPDATE books SET voice=?, status='processing', eta_seconds=?, "
            "processing_started_at=?, error=NULL WHERE id=?",
            (voice, eta, time.time(), book_id)
        )
        row = dict(conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone())

    threading.Thread(target=process_book, args=(book_id, voice), daemon=True).start()
    return jsonify(row)


@app.route('/api/books/<int:book_id>/audio')
def book_audio(book_id):
    p = AUDIO_DIR / f"{book_id}.mp3"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype='audio/mpeg', conditional=True)


@app.route('/api/books/<int:book_id>/sync')
def book_sync(book_id):
    p = SYNC_DIR / f"{book_id}.json"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype='application/json')


@app.route('/api/voices')
def list_voices():
    with db() as conn:
        rows = conn.execute(
            "SELECT voice, COUNT(*) as count FROM books WHERE status='ready' GROUP BY voice"
        ).fetchall()
    counts = {r['voice']: r['count'] for r in rows}
    return jsonify([{"id": v, "count": counts.get(v, 0)} for v in VOICES])


@app.route('/api/voices/<voice>/sample')
def voice_sample(voice):
    if voice not in VOICES:
        abort(404)
    p = SAMPLES_DIR / f"{voice}.mp3"
    if not p.exists():
        try:
            audio, _ = synthesize(SAMPLE_TEXT, voice)
            p.write_bytes(audio)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return send_file(p, mimetype='audio/mpeg', conditional=True)


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"Library: {LIBRARY_DIR}")
    print("Open http://localhost:5000 in a browser")
    print("For GPU: run ./start.sh instead of python server.py")
    app.run(host='0.0.0.0', port=5000, threaded=True)
