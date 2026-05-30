"""
TTS Reader Server — run with: ./start.sh (uses GPU)
Or: python server.py (falls back to CPU, slower)
Then open http://localhost:5000 in a browser.
"""
import json
import io
import math
import os
import random
import re
import sqlite3
import subprocess
import threading
import time
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, jsonify, send_file, abort

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

LIBRARY_DIR = Path.home() / "book_reader" / "library"
AUDIO_DIR   = LIBRARY_DIR / "audio"
SYNC_DIR    = LIBRARY_DIR / "sync"
SAMPLES_DIR = LIBRARY_DIR / "samples"
SOURCES_DIR = LIBRARY_DIR / "sources"
VOICES_DIR  = LIBRARY_DIR / "voices"
DB_PATH     = LIBRARY_DIR / "library.db"

for d in (LIBRARY_DIR, AUDIO_DIR, SYNC_DIR, SAMPLES_DIR, SOURCES_DIR, VOICES_DIR):
    d.mkdir(parents=True, exist_ok=True)

HTML_PATH = Path(__file__).resolve().parent / "frontend.html"

SAMPLE_TEXT = "In the still hours before dawn, the city held its breath."

def get_voices():
    """'default' plus any .wav reference clips the user has uploaded."""
    names = ["default"]
    for p in sorted(VOICES_DIR.glob("*.wav")):
        names.append(p.stem)
    return names



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
        cols = {r['name'] for r in conn.execute("PRAGMA table_info(books)").fetchall()}
        if 'exaggeration' not in cols:
            conn.execute("ALTER TABLE books ADD COLUMN exaggeration REAL NOT NULL DEFAULT 0.5")
        if 'cfg_weight' not in cols:
            conn.execute("ALTER TABLE books ADD COLUMN cfg_weight REAL NOT NULL DEFAULT 0.5")
        conn.execute(
            "UPDATE books SET status='failed', error='Server restarted during processing' "
            "WHERE status='processing'"
        )


def _clamp01(v, default=0.5):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return max(0.0, min(1.0, f))


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


# ---------- Chatterbox TTS ----------

_cb_model = None
_cb_model_lock  = threading.Lock()
_cb_synthesis_lock = threading.Lock()  # Chatterbox is not thread-safe for concurrent synthesis


def get_cb_model():
    global _cb_model
    if _cb_model is not None:
        return _cb_model
    with _cb_model_lock:
        if _cb_model is not None:
            return _cb_model
        print("Loading Chatterbox... (first run downloads weights ~1 GB)")
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
        _cb_model = model
        print(f"Chatterbox ready — device={device}  sr={model.sr}")
        return _cb_model


threading.Thread(target=lambda: get_cb_model(), daemon=True).start()


def _apply_humanization(audio, sr):
    """EQ + compression + subtle reverb + loudness normalization on a float32 mono array."""
    if len(audio) == 0:
        return audio
    try:
        from pedalboard import Pedalboard, HighpassFilter, Compressor, Reverb
    except ImportError:
        return audio

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-18, ratio=2.5, attack_ms=5.0, release_ms=150.0),
        Reverb(room_size=0.08, wet_level=0.03, dry_level=0.97, damping=0.7),
    ])
    processed = board(audio.reshape(1, -1), sample_rate=sr)[0]

    # Loudness normalization — compute gain ourselves so we can cap it before clipping occurs
    if len(processed) >= int(sr * 0.4):
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(processed)
            if np.isfinite(loudness) and loudness > -70.0:
                gain = 10 ** ((-20.0 - loudness) / 20.0)
                peak = np.max(np.abs(processed))
                if peak > 0:
                    gain = min(gain, 0.98 / peak)
                processed = processed * gain
        except Exception:
            pass

    return processed.astype(np.float32)


def split_sentences(text):
    text = re.sub(r'\s+', ' ', text.strip())
    parts = re.split(r'(?<=[.!?…])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def synthesize(text, voice, exaggeration=0.5, cfg_weight=0.5):
    import lameenc
    import torchaudio
    model = get_cb_model()
    sr = model.sr

    voice_path = None
    if voice != "default":
        vp = VOICES_DIR / f"{voice}.wav"
        voice_path = str(vp) if vp.exists() else None

    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    if not paragraphs:
        raise Exception("No text to synthesize")

    all_segs   = []
    all_timings = []
    current_time = 0.0

    for para_idx, para in enumerate(paragraphs):
        for sent_idx, s in enumerate(split_sentences(para)):
            # Paragraph break pause
            if para_idx > 0 and sent_idx == 0:
                pause = random.uniform(0.45, 0.65)
                all_segs.append(np.zeros(int(pause * sr), dtype=np.float32))
                current_time += pause

            with torch.no_grad():
                wav = model.generate(
                    s,
                    audio_prompt_path=voice_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

            audio = wav.squeeze().cpu().float().numpy()

            # Speed jitter ±5% via resampling (changes duration + pitch slightly)
            speed = random.uniform(0.95, 1.05)
            if abs(speed - 1.0) > 0.005:
                wav_t = torch.from_numpy(audio).unsqueeze(0)
                wav_t = torchaudio.functional.resample(
                    wav_t, orig_freq=sr, new_freq=int(sr / speed)
                )
                audio = wav_t.squeeze().numpy()

            dur = len(audio) / sr
            all_segs.append(audio)

            ws = s.split()
            if ws:
                total = sum(len(w) for w in ws)
                t = current_time
                for w in ws:
                    wd = len(w) / max(total, 1) * dur
                    all_timings.append({"word": w, "start": t, "end": t + wd})
                    t += wd
            current_time += dur

    full_audio = np.concatenate(all_segs) if all_segs else np.zeros(0, dtype=np.float32)
    full_audio = _apply_humanization(full_audio, sr)

    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(5)
    pcm = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
    mp3_data = enc.encode(pcm.tobytes()) + enc.flush()

    return mp3_data, all_timings


# ---------- calibration ----------

def get_sec_per_word(conn):
    rows = conn.execute(
        "SELECT key, value FROM meta WHERE key IN ('total_seconds', 'total_words')"
    ).fetchall()
    d = {r["key"]: float(r["value"]) for r in rows}
    if d.get("total_words", 0) > 0:
        return d["total_seconds"] / d["total_words"]
    return 0.05    # Chatterbox GPU default: ~20 words/sec


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

def process_book(book_id, voice, exaggeration=0.5, cfg_weight=0.5):
    try:
        text = (SOURCES_DIR / f"{book_id}.txt").read_text(encoding='utf-8')
        t0   = time.time()
        with _cb_synthesis_lock:
            audio, words = synthesize(text, voice, exaggeration, cfg_weight)
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
    voice = request.form.get('voice', 'default')
    exag  = _clamp01(request.form.get('exaggeration'), 0.5)
    cfg   = _clamp01(request.form.get('cfg_weight'),   0.5)
    if not file:
        return jsonify({"error": "No file"}), 400
    if voice not in get_voices():
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
            "status, processing_started_at, exaggeration, cfg_weight, created_at) "
            "VALUES (?,?,?,?,?,'processing',?,?,?,datetime('now'))",
            (name, ext.lstrip('.'), voice, word_count, eta, time.time(), exag, cfg)
        )
        book_id = cur.lastrowid
        row = dict(conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone())

    (SOURCES_DIR / f"{book_id}.txt").write_text(text, encoding='utf-8')
    threading.Thread(target=process_book, args=(book_id, voice, exag, cfg), daemon=True).start()
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
    if voice not in get_voices():
        return jsonify({"error": "Unknown voice"}), 400
    if not (SOURCES_DIR / f"{book_id}.txt").exists():
        return jsonify({"error": "Source not available"}), 404

    with db() as conn:
        row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
        if not row:
            return jsonify({"error": "Not found"}), 404
        exag = _clamp01(data.get('exaggeration'), row['exaggeration'] if row['exaggeration'] is not None else 0.5)
        cfg  = _clamp01(data.get('cfg_weight'),   row['cfg_weight']   if row['cfg_weight']   is not None else 0.5)
        sec_per_word = get_sec_per_word(conn)
        eta = (row['word_count'] or 0) * sec_per_word
        conn.execute(
            "UPDATE books SET voice=?, status='processing', eta_seconds=?, "
            "processing_started_at=?, exaggeration=?, cfg_weight=?, error=NULL WHERE id=?",
            (voice, eta, time.time(), exag, cfg, book_id)
        )
        row = dict(conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone())

    threading.Thread(target=process_book, args=(book_id, voice, exag, cfg), daemon=True).start()
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
    return jsonify([{"id": v, "count": counts.get(v, 0)} for v in get_voices()])


@app.route('/api/voices/upload', methods=['POST'])
def upload_voice():
    name = (request.form.get('name') or '').strip()
    file = request.files.get('file')
    if not name:
        return jsonify({"error": "name required"}), 400
    if not file:
        return jsonify({"error": "wav file required"}), 400
    name = re.sub(r'[^\w\-]', '_', name)
    if name == 'default':
        return jsonify({"error": "Cannot use 'default' as voice name"}), 400
    dest = VOICES_DIR / f"{name}.wav"
    file.save(str(dest))
    # clear stale sample if one existed
    (SAMPLES_DIR / f"{name}.mp3").unlink(missing_ok=True)
    return jsonify({"id": name, "count": 0})


@app.route('/api/voices/<voice>', methods=['DELETE'])
def delete_voice(voice):
    if voice == 'default':
        return jsonify({"error": "Cannot delete the default voice"}), 400
    (VOICES_DIR  / f"{voice}.wav").unlink(missing_ok=True)
    (SAMPLES_DIR / f"{voice}.mp3").unlink(missing_ok=True)
    return jsonify({"ok": True})


@app.route('/api/voices/<voice>/sample')
def voice_sample(voice):
    if voice not in get_voices():
        abort(404)
    exag = _clamp01(request.args.get('exaggeration'), 0.5)
    cfg  = _clamp01(request.args.get('cfg_weight'),   0.5)
    # Use the plain filename for the (0.5, 0.5) default — preserves legacy samples;
    # tag others by their values so the picker can preview each personality.
    if abs(exag - 0.5) < 0.01 and abs(cfg - 0.5) < 0.01:
        p = SAMPLES_DIR / f"{voice}.mp3"
    else:
        p = SAMPLES_DIR / f"{voice}_{int(round(exag*100))}_{int(round(cfg*100))}.mp3"
    if not p.exists():
        try:
            audio, _ = synthesize(SAMPLE_TEXT, voice, exag, cfg)
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
