from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import tempfile
import uuid
import logging

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
UPLOAD_FOLDER = "uploads"

# Analyze more frames without processing every frame in a video.
VIDEO_SAMPLE_COUNT = 16

# Existing model decision threshold.
# Model output >= 0.5 means REAL; output < 0.5 means FAKE.
REAL_THRESHOLD = 0.50

# These are screening heuristics, not scientifically calibrated
# confidence thresholds.
UNCERTAIN_MARGIN = 0.05
HIGH_DISAGREEMENT_STD = 0.20

# Upload limits for the Python API.
MAX_IMAGE_SIZE_MB = 100
MAX_VIDEO_SIZE_MB = 100

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_VIDEO_SIZE_MB * 1024 * 1024
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rAIdar")

# ============================================================
# LOAD EXISTING MODEL
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)

logger.info("rAIdar model loaded successfully.")
logger.info("Model path: %s", MODEL_PATH)
logger.info("Video samples per upload: %s", VIDEO_SAMPLE_COUNT)

# ============================================================
# GENERAL ROUTES
# ============================================================

@app.route("/")
def home():
    return "rAIdar AI Detection API Running"


@app.route("/health")
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None
    })


# ============================================================
# IMAGE DETECTION
# Existing endpoint preserved: POST /predict
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "message": "No file uploaded."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "success": False,
            "message": "No image selected."
        }), 400

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in {".jpg", ".jpeg", ".png"}:
        return jsonify({
            "success": False,
            "message": "Unsupported image format."
        }), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
            dir=UPLOAD_FOLDER
        ) as temp_file:
            temp_path = temp_file.name
            file.save(temp_file)

        if os.path.getsize(temp_path) == 0:
            return jsonify({
                "success": False,
                "message": "The uploaded image is empty."
            }), 400

        if os.path.getsize(temp_path) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            return jsonify({
                "success": False,
                "message": "Image exceeds the 100 MB API limit."
            }), 413

        with Image.open(temp_path) as source_image:
            img = source_image.convert("RGB")
            img = img.resize((224, 224))

            img_array = np.asarray(
                img,
                dtype=np.float32
            )

        # Keep the original model preprocessing.
        # The model contains its own preprocessing layers.
        img_array = np.expand_dims(img_array, axis=0)

        raw_score = float(
            model.predict(img_array, verbose=0)[0][0]
        )

        if not np.isfinite(raw_score):
            raise ValueError("The model returned an invalid score.")

        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        if raw_score >= REAL_THRESHOLD:
            label = "REAL"
            confidence = raw_score * 100
            message = "This image appears to be authentic."
        else:
            label = "FAKE"
            confidence = (1.0 - raw_score) * 100
            message = (
                "This image shows visual patterns associated "
                "with AI-generated or manipulated imagery. "
                "This is not proof of its origin."
            )

        logger.info(
            "IMAGE | filename=%s | raw_score=%.6f | prediction=%s",
            os.path.basename(file.filename),
            raw_score,
            label
        )

        return jsonify({
            "success": True,
            "prediction": label,
            "confidence": round(confidence, 2),
            "raw_prediction": round(raw_score, 6),
            "message": message
        })

    except Exception:
        logger.exception("Image prediction failed.")

        return jsonify({
            "success": False,
            "message": "Image analysis failed. Check the API terminal."
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary image file.")


# ============================================================
# VIDEO FRAME PREPROCESSING
# ============================================================

def preprocess_video_frame(frame):
    """
    Convert an OpenCV BGR frame to RGB and resize it to the
    existing model input size.

    Do not divide pixel values by 255 because the existing
    model contains its own preprocessing layers.
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(
        frame,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    return frame.astype(np.float32)


def sample_video_frames(video_path, sample_count):
    """
    Select frames at approximately uniform intervals throughout
    the video. Return the frames and their original frame indices.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        if total_frames <= 0:
            raise ValueError(
                "The video contains no readable frames."
            )

        # Avoid requesting more samples than available frames.
        actual_sample_count = min(
            sample_count,
            total_frames
        )

        # Unique, evenly distributed source-frame indices.
        frame_indices = np.unique(
            np.linspace(
                0,
                total_frames - 1,
                num=actual_sample_count,
                dtype=np.int64
            )
        )

        frames = []
        successful_indices = []

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success or frame is None:
                logger.warning(
                    "Could not read source frame %s.",
                    frame_index
                )
                continue

            processed = preprocess_video_frame(frame)

            frames.append(processed)
            successful_indices.append(int(frame_index))

        if not frames:
            raise ValueError(
                "No readable frames could be extracted."
            )

        duration_seconds = (
            total_frames / fps
            if fps > 0
            else None
        )

        return (
            frames,
            successful_indices,
            total_frames,
            fps,
            duration_seconds
        )

    finally:
        cap.release()


# ============================================================
# VIDEO RESULT ANALYSIS
# ============================================================

def analyze_video_scores(raw_scores):
    """
    Summarize sampled-frame model scores.

    Model output >= 0.5 means REAL.
    Model output < 0.5 means FAKE.

    The final label is a screening result, not a verified
    probability of authenticity.
    """

    scores = np.asarray(
        raw_scores,
        dtype=np.float64
    )

    if scores.size == 0:
        raise ValueError("No frame predictions are available.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("One or more frame scores are invalid.")

    scores = np.clip(scores, 0.0, 1.0)

    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores))

    real_count = int(
        np.sum(scores >= REAL_THRESHOLD)
    )

    fake_count = int(
        np.sum(scores < REAL_THRESHOLD)
    )

    frame_count = int(scores.size)

    real_ratio = real_count / frame_count
    fake_ratio = fake_count / frame_count

    # Preserve the existing model's original decision rule:
    # average score >= 0.5 is REAL; otherwise FAKE.
    if mean_score >= REAL_THRESHOLD:
        label = "REAL"
        class_score = mean_score * 100
    else:
        label = "FAKE"
        class_score = (1.0 - mean_score) * 100

    # Flag borderline averages or strong disagreement.
    borderline = (
        abs(mean_score - REAL_THRESHOLD)
        <= UNCERTAIN_MARGIN
    )

    high_disagreement = (
        std_score >= HIGH_DISAGREEMENT_STD
        or (real_count > 0 and fake_count > 0)
    )

    if borderline:
        confidence_level = "LOW"
        review_status = "INCONCLUSIVE"
        message = (
            "The average frame score is close to the decision "
            "threshold. The video needs further verification."
        )

    elif high_disagreement:
        confidence_level = "LOW"
        review_status = "MIXED_RESULTS"
        message = (
            "The sampled frames produced mixed or inconsistent "
            "predictions. The final label may not represent "
            "the entire video."
        )

    else:
        confidence_level = "SCREENING_ONLY"
        review_status = "SCREENING_RESULT"
        message = (
            "The sampled frames lean toward "
            + label
            + ". This result does not establish the video's origin."
        )

    frame_results = []

    for index, score in enumerate(scores):

        if score >= REAL_THRESHOLD:
            frame_label = "REAL"
            frame_class_score = score * 100
        else:
            frame_label = "FAKE"
            frame_class_score = (1.0 - score) * 100

        frame_results.append({
            "frame": index + 1,
            "raw_prediction": round(float(score), 6),
            "prediction": frame_label,
            "confidence": round(float(frame_class_score), 2)
        })

    return {
        "prediction": label,
        "confidence": round(float(class_score), 2),
        "raw_prediction": round(mean_score, 6),
        "frames_analyzed": frame_count,
        "frame_results": frame_results,

        # Additional diagnostics.
        "mean_raw_score": round(mean_score, 6),
        "median_raw_score": round(median_score, 6),
        "std_raw_score": round(std_score, 6),
        "minimum_raw_score": round(float(np.min(scores)), 6),
        "maximum_raw_score": round(float(np.max(scores)), 6),

        "real_frame_count": real_count,
        "fake_frame_count": fake_count,
        "real_frame_percentage": round(real_ratio * 100, 2),
        "fake_frame_percentage": round(fake_ratio * 100, 2),

        "confidence_level": confidence_level,
        "review_status": review_status,
        "message": message,

        "warning": (
            "This system analyzes sampled frames using an image "
            "classifier. It does not directly analyze motion, "
            "verify provenance, or prove that a video is authentic "
            "or AI-generated. Displayed scores are not calibrated "
            "probabilities of authenticity."
        )
    }


# ============================================================
# VIDEO DETECTION
# Existing endpoint preserved: POST /predict_video
# ============================================================

@app.route("/predict_video", methods=["POST"])
def predict_video():

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "message": "No video uploaded."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "success": False,
            "message": "No video selected."
        }), 400

    extension = os.path.splitext(file.filename)[1].lower()

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm"
    }

    if extension not in allowed_extensions:
        return jsonify({
            "success": False,
            "message": (
                "Unsupported video format. Use MP4, MOV, AVI, MKV, "
                "or WEBM."
            )
        }), 400

    temp_path = None

    try:
        # Each upload gets a unique temporary filename.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
            dir=UPLOAD_FOLDER
        ) as temp_file:

            temp_path = temp_file.name
            file.save(temp_file)

        file_size = os.path.getsize(temp_path)

        if file_size == 0:
            return jsonify({
                "success": False,
                "message": "The uploaded video is empty."
            }), 400

        if file_size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
            return jsonify({
                "success": False,
                "message": "Video exceeds the 100 MB API limit."
            }), 413

        logger.info("=" * 65)
        logger.info(
            "Starting video analysis: %s",
            os.path.basename(file.filename)
        )

        (
            frames,
            source_indices,
            total_frames,
            fps,
            duration_seconds
        ) = sample_video_frames(
            temp_path,
            VIDEO_SAMPLE_COUNT
        )

        # Predict all sampled frames together.
        batch = np.asarray(
            frames,
            dtype=np.float32
        )

        predictions = model.predict(
            batch,
            verbose=0
        ).reshape(-1)

        if len(predictions) != len(source_indices):
            raise ValueError(
                "The number of predictions does not match "
                "the number of sampled frames."
            )

        if not np.all(np.isfinite(predictions)):
            raise ValueError(
                "The model returned invalid frame predictions."
            )

        result = analyze_video_scores(predictions)

        # Add source-frame positions for diagnostics.
        for frame_result, source_index in zip(
            result["frame_results"],
            source_indices
        ):
            frame_result["source_frame"] = source_index

        result.update({
            "success": True,
            "filename": os.path.basename(file.filename),
            "total_source_frames": total_frames,
            "sampled_source_frames": source_indices,
            "fps": round(fps, 3) if fps > 0 else None,
            "duration_seconds": (
                round(duration_seconds, 2)
                if duration_seconds is not None
                else None
            ),
            "sample_strategy": "Uniformly spaced frames"
        })

        # ----------------------------------------------------
        # TERMINAL DIAGNOSTICS
        # ----------------------------------------------------

        logger.info("-" * 65)
        logger.info("VIDEO FRAME DIAGNOSTICS")

        for frame_result in result["frame_results"]:
            logger.info(
                "Sample %s | Source frame %s | Raw score %.6f "
                "| Prediction %s | Class score %.2f%%",
                frame_result["frame"],
                frame_result["source_frame"],
                frame_result["raw_prediction"],
                frame_result["prediction"],
                frame_result["confidence"]
            )

        logger.info("-" * 65)
        logger.info("VIDEO ANALYSIS SUMMARY")
        logger.info("Filename: %s", result["filename"])
        logger.info("Duration: %s", result["duration_seconds"])
        logger.info("Total source frames: %s", total_frames)
        logger.info("Frames analyzed: %s", result["frames_analyzed"])
        logger.info("REAL frames: %s", result["real_frame_count"])
        logger.info("FAKE frames: %s", result["fake_frame_count"])
        logger.info("Mean raw score: %s", result["mean_raw_score"])
        logger.info("Median raw score: %s", result["median_raw_score"])
        logger.info("Score standard deviation: %s", result["std_raw_score"])
        logger.info("Final prediction: %s", result["prediction"])
        logger.info("Displayed class score: %s%%", result["confidence"])
        logger.info("Confidence level: %s", result["confidence_level"])
        logger.info("Review status: %s", result["review_status"])
        logger.info("=" * 65)

        return jsonify(result)

    except ValueError as error:
        logger.warning("Video could not be analyzed: %s", error)

        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except Exception:
        logger.exception("Video prediction failed.")

        return jsonify({
            "success": False,
            "message": (
                "Video analysis failed. Check the Python API terminal "
                "for the detailed error."
            )
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(
                    "Could not remove temporary video file."
                )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "success": False,
        "message": "The uploaded file exceeds the API size limit."
    }), 413


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False
    )