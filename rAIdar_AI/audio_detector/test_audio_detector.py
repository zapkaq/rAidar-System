import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE = 16000


def main():
    if len(sys.argv) != 2:
        print('Usage: python test_audio_detector.py "path\\to\\audio.wav"')
        raise SystemExit(1)

    audio_path = Path(sys.argv[1])

    if not audio_path.is_file():
        print(f"ERROR: Audio file not found: {audio_path}")
        raise SystemExit(1)

    print("Device: CPU")
    print("Loading model and feature extractor...")
    print("The first run may take several minutes to download the model.")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Loading audio: {audio_path.name}")

    waveform, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    if waveform.size == 0:
        print("ERROR: The audio file is empty.")
        raise SystemExit(1)

    if not np.all(np.isfinite(waveform)):
        print("ERROR: The audio contains invalid numeric samples.")
        raise SystemExit(1)

    inputs = extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    print("Analyzing audio...")

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_label = model.config.id2label[predicted_index]

    print("\n--- Audio Detection Result ---")
    print(f"Prediction: {predicted_label.upper()}")
    print("\nModel scores:")

    for index, probability in enumerate(probabilities.tolist()):
        label = model.config.id2label[index]
        print(f"  {label}: {probability * 100:.2f}%")

    print("\nNote: These are model scores, not calibrated probabilities.")
    print("Results can be wrong, especially on unfamiliar audio or languages.")


if __name__ == "__main__":
    main()