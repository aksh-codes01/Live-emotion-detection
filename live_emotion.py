"""
Live facial emotion detection.

Uses the CNN trained on FER+ in Google Colab (emotion_cnn_v2.keras).
Opens the webcam, finds faces with OpenCV, and labels each face with
one of 8 emotions. Press Q to quit.
"""
import cv2
import numpy as np
import tensorflow as tfgit init

MODEL_PATH = "emotion_cnn_v2.keras"
EMOTIONS = ["neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"]

# compile=False: we only need the model for predictions, not for training,
# so we skip restoring the optimizer and learning-rate schedule.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# OpenCV's built-in face detector (ships with the opencv-python package).
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict(faces):
    """Return 8 emotion probabilities for each 48x48 greyscale face.

    Prepares faces exactly like in training (scale pixels to 0-1, add the
    channel dimension), and uses test-time augmentation: each face is also
    predicted mirrored, and the two answers are averaged.
    """
    batch = np.stack(faces).astype("float32")[..., None] / 255.0
    batch = np.concatenate([batch, batch[:, :, ::-1]])  # originals + mirrors
    probs = model(batch, training=False).numpy()
    n = len(faces)
    return (probs[:n] + probs[n:]) / 2


def draw_bars(frame, probs):
    """Draw a small bar chart of all 8 emotion probabilities (top-left)."""
    for i, (name, p) in enumerate(zip(EMOTIONS, probs)):
        y = 25 + i * 22
        cv2.rectangle(frame, (10, y - 14), (10 + int(p * 150), y + 2),
                      (0, 200, 0), -1)
        cv2.putText(frame, f"{name} {p:.0%}", (165, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view, like a selfie camera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        boxes = face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        if len(boxes):
            # Crop each face and shrink it to 48x48, the size the model expects.
            crops = [cv2.resize(gray[y:y + h, x:x + w], (48, 48),
                                interpolation=cv2.INTER_AREA)
                     for (x, y, w, h) in boxes]
            all_probs = predict(crops)

            for (x, y, w, h), probs in zip(boxes, all_probs):
                label = f"{EMOTIONS[probs.argmax()]} {probs.max():.0%}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Bar chart for the biggest (closest) face.
            biggest = max(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3])
            draw_bars(frame, all_probs[biggest])

        cv2.imshow("Emotion detection - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
