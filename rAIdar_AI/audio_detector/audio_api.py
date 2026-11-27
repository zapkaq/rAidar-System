
import io
import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import torch

from flask import Flask, jsonify, request
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

SAMPLE_RATE = 16000

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
MAX_DURATION_SECONDS = 120

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

DEVICE = torch.device("cpu")


# --------------------------------------------------
# Load AI Model
# --------------------------------------------------

print("Loading rAIdar audio detection model...")
print(f"Device: {DEVICE}")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("Audio model loaded successfully.")


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def error_response(message, status_code):
    return jsonify({
        "success": False,
        "message": message
    }), status_code


def get_label(index):
    label = model.config.id2label.get(index, str(index))
    return str(label).strip().lower()


def validate_waveform(waveform, duration_seconds):

    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(
            "The uploaded file does not contain usable audio."
        )

    if not np.all(np.isfinite(waveform)):
        raise ValueError(
            "The audio contains invalid sample values."
        )

    if duration_seconds > MAX_DURATION_SECONDS:
        raise ValueError(
            "Audio is too long. Maximum duration is 120 seconds."
        )


# --------------------------------------------------
# FFmpeg Audio Decoder
# --------------------------------------------------

def decode_with_ffmpeg(file_bytes, extension=".m4a"):
    """
    Decode audio using a temporary seekable file.

    Supports M4A, AAC, MP3, WAV, FLAC, and OGG
    when the installed FFmpeg build supports the codec.
    """

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    temp_path = None

    try:

        # Create a temporary file with the original extension.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_audio:

            temp_audio.write(file_bytes)
            temp_path = temp_audio.name

        command = [
            ffmpeg_exe,

            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",

            "-i", temp_path,

            # Select the first available audio stream.
            "-map", "0:a:0",

            # Convert audio to mono, 16 kHz, float PCM.
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_f32le",

            "-f", "f32le",
            "pipe:1",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )

        if result.returncode != 0 or not result.stdout:

            details = result.stderr.decode(
                "utf-8",
                errors="replace"
            ).strip()

            app.logger.error(
                "FFmpeg audio decoding failed: %s",
                details[-2000:]
            )

            raise ValueError(
                "FFmpeg could not decode this audio file. "
                "Please check whether the file is valid "
                "and contains an audio stream."
            )

        waveform = np.frombuffer(
            result.stdout,
            dtype="<f4"
        ).copy()

        duration_seconds = waveform.size / SAMPLE_RATE

        validate_waveform(
            waveform,
            duration_seconds
        )

        return waveform, duration_seconds

    finally:

        # Remove the temporary file even if decoding fails.
        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except OSError:
                app.logger.warning(
                    "Could not remove temporary audio file."
                )


# --------------------------------------------------
# General Audio Decoder
# --------------------------------------------------

def decode_audio(file_bytes, extension):
    """
    Decode uploaded audio.

    M4A uses FFmpeg directly because MP4/M4A containers
    may require seekable input.

    Other formats try SoundFile, Librosa, and FFmpeg.
    """

    extension = extension.lower()

    # M4A: Use FFmpeg directly.
    if extension == ".m4a":

        return decode_with_ffmpeg(
            file_bytes,
            extension
        )

    # ----------------------------------------------
    # First attempt: SoundFile
    # ----------------------------------------------

    try:

        waveform, original_sample_rate = sf.read(
            io.BytesIO(file_bytes),
            dtype="float32",
            always_2d=False
        )

        if original_sample_rate <= 0:
            raise ValueError("Invalid audio sample rate.")

        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(
                "The uploaded file does not contain usable audio."
            )

        duration_seconds = (
            waveform.size / original_sample_rate
        )

        validate_waveform(
            waveform,
            duration_seconds
        )

        if original_sample_rate != SAMPLE_RATE:

            waveform = librosa.resample(
                waveform,
                orig_sr=original_sample_rate,
                target_sr=SAMPLE_RATE
            )

        return waveform.astype(np.float32), duration_seconds

    except Exception as exc:

        app.logger.info(
            "SoundFile decoder unavailable: %s",
            str(exc)
        )

    # ----------------------------------------------
    # Second attempt: Librosa
    # ----------------------------------------------

    try:

        waveform, _ = librosa.load(
            io.BytesIO(file_bytes),
            sr=SAMPLE_RATE,
            mono=True
        )

        if waveform.size == 0:
            raise ValueError(
                "The uploaded file contains no audio samples."
            )

        duration_seconds = waveform.size / SAMPLE_RATE

        validate_waveform(
            waveform,
            duration_seconds
        )

        return waveform.astype(np.float32), duration_seconds

    except Exception as exc:

        app.logger.info(
            "Librosa decoder unavailable: %s",
            str(exc)
        )

    # ----------------------------------------------
    # Final attempt: FFmpeg
    # ----------------------------------------------

    return decode_with_ffmpeg(
        file_bytes,
        extension
    )


# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "service": "rAIdar Audio Detection API",
        "status": "running",
        "device": str(DEVICE),

        "supported_formats": sorted(
            ALLOWED_EXTENSIONS
        ),

        "max_file_size_mb": (
            MAX_FILE_SIZE // (1024 * 1024)
        ),

        "max_duration_seconds": MAX_DURATION_SECONDS,

        "endpoints": {
            "health": "/health",
            "audio_detection": "/predict"
        }
    })


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    })


# --------------------------------------------------
# Audio Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check uploaded file.
    if "file" not in request.files:

        return error_response(
            "No audio file provided. "
            "Upload the file using the 'file' field.",
            400
        )

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:

        return error_response(
            "No filename provided.",
            400
        )

    # Validate extension.
    extension = Path(
        uploaded_file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        return error_response(
            "Unsupported audio format. Supported formats: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS)),
            415
        )

    # Read uploaded file.
    file_bytes = uploaded_file.read(
        MAX_FILE_SIZE + 1
    )

    if not file_bytes:

        return error_response(
            "The uploaded file is empty.",
            400
        )

    if len(file_bytes) > MAX_FILE_SIZE:

        return error_response(
            "The audio file exceeds the 25 MB limit.",
            413
        )

    # ----------------------------------------------
    # Decode Audio
    # ----------------------------------------------

    try:

        waveform, duration_seconds = decode_audio(
            file_bytes,
            extension
        )

    except ValueError as exc:

        return error_response(
            str(exc),
            400
        )

    except subprocess.TimeoutExpired:

        return error_response(
            "Audio decoding took too long. "
            "Try a shorter audio file.",
            400
        )

    except Exception:

        app.logger.exception(
            "Audio decoding failed."
        )

        return error_response(
            "Unable to decode this audio file. "
            "Try a valid WAV, MP3, FLAC, OGG, or M4A file.",
            400
        )

    # ----------------------------------------------
    # AI Feature Extraction and Prediction
    # ----------------------------------------------

    try:

        inputs = feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        predicted_label = get_label(
            predicted_index
        )

        scores = {
            get_label(index): round(
                float(probability.item()) * 100,
                2
            )
            for index, probability in enumerate(
                probabilities
            )
        }

        # Validate model output.
        if "real" not in scores or "fake" not in scores:

            return error_response(
                "The loaded model returned unexpected class labels.",
                500
            )

        if predicted_label not in {"real", "fake"}:

            return error_response(
                "The model returned an unknown prediction label.",
                500
            )

        # ------------------------------------------
        # Successful Response
        # ------------------------------------------

        return jsonify({

            "success": True,

            "prediction": predicted_label.upper(),

            "real_score": scores["real"],

            "fake_score": scores["fake"],

            "duration_seconds": round(
                float(duration_seconds),
                2
            ),

            "sample_rate_hz": SAMPLE_RATE,

            "message": (
                "Audio analysis completed successfully."
            ),

            "warning": (
                "Model scores are not calibrated probabilities. "
                "This result alone cannot prove whether audio is authentic."
            )
        })

    except Exception:

        app.logger.exception(
            "Audio prediction failed."
        )

        return error_response(
            "Audio analysis failed while running the model.",
            500
        )


# --------------------------------------------------
# File Size Error Handler
# --------------------------------------------------

@app.errorhandler(413)
def request_too_large(_error):

    return error_response(
        "The audio file exceeds the 25 MB limit.",
        413
    )


# --------------------------------------------------
# Run Flask API
# --------------------------------------------------

if __name__ == "__main__":

    print("----------------------------------------")
    print("rAIdar Audio Detection API")
    print("----------------------------------------")
    print("Supported formats: WAV, MP3, FLAC, OGG, M4A")
    print("Sample rate:", SAMPLE_RATE)
    print("Maximum file size: 25 MB")
    print("Maximum duration:", MAX_DURATION_SECONDS, "seconds")
    print("Server: http://127.0.0.1:5002")
    print("----------------------------------------")

    app.run(
        host="127.0.0.1",
        port=5002,
        debug=False
    )
