import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# V3 body-part model was trained at 224 x 224.
BODY_IMAGE_SIZE = (224, 224)

# Existing fracture model was trained at 160 x 160.
FRACTURE_IMAGE_SIZE = (160, 160)

BODY_MODEL_PATH = MODEL_DIR / "body_part_model_v3.keras"
BODY_LABELS_PATH = MODEL_DIR / "body_part_classes_v3.json"

FRACTURE_MODEL_PATH = MODEL_DIR / "fracture_model.keras"
FRACTURE_LABELS_PATH = MODEL_DIR / "fracture_classes.json"
FRACTURE_THRESHOLD_PATH = MODEL_DIR / "fracture_threshold.json"

st.set_page_config(
    page_title="AI X-ray Analysis Assistant",
    page_icon="🩻",
    layout="wide",
)

# Optional visual cleanup for the hosted web interface.
# Note: Streamlit may change these CSS selectors between versions.
st.markdown(
    """
    <style>
    /* Hide the Streamlit footer and deploy badge where supported. */
    footer { visibility: hidden; height: 0; }
    .stDeployButton, [data-testid="stAppDeployButton"],
    .viewerBadge_container__1QSob { display: none !important; }

    /* Hide Streamlit's top toolbar (including the GitHub/menu icons).
       This only changes the appearance of the page. */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩻 AI-Based Universal X-ray Analysis Assistant")
st.caption(
    "Experimental X-ray classification and explainability "
    "for educational and research use."
)

st.warning(
    "This application is not a medical diagnostic tool. "
    "Predictions may be incorrect and must not replace assessment "
    "by a qualified radiologist."
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_labels(path):
    """Read labels from a JSON list or dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        try:
            return [
                str(key)
                for key, value in sorted(
                    data.items(),
                    key=lambda item: int(item[1]),
                )
            ]
        except (ValueError, TypeError):
            return [str(value) for value in data.values()]

    raise ValueError(f"Unsupported labels format: {path.name}")


@st.cache_resource
def load_keras_model(model_path):
    """Load and cache a trained Keras model."""
    return tf.keras.models.load_model(str(model_path), compile=False)


def preprocess_body_image(image):
    """
    V3 training pipeline:
    - RGB
    - 224 x 224
    - float32 values in the 0..255 range
    MobileNetV2 preprocessing is already inside the saved V3 model.
    Do NOT divide by 255 here.
    """
    resized = image.convert("RGB").resize(BODY_IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def preprocess_fracture_image(image):
    """
    Existing fracture model pipeline:
    - RGB
    - 160 x 160
    - float32 values normalized to 0..1
    """
    resized = image.convert("RGB").resize(FRACTURE_IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def get_body_part_prediction(model, image_array, labels):
    """Predict the X-ray body-part category."""
    output = np.asarray(model.predict(image_array, verbose=0)).reshape(-1)

    if len(output) != len(labels):
        raise ValueError(
            "Body-part model output does not match its label file."
        )

    index = int(np.argmax(output))
    return labels[index], float(output[index]), output


def get_fracture_prediction(model, image_array, labels, threshold):
    """Predict fracture probability using the binary sigmoid model."""
    output = np.asarray(model.predict(image_array, verbose=0)).reshape(-1)

    if len(output) != 1:
        raise ValueError(
            "The fracture model is expected to have one sigmoid output."
        )

    fracture_probability = float(np.clip(output[0], 0.0, 1.0))
    index = 1 if fracture_probability >= threshold else 0

    if len(labels) < 2:
        labels = ["Not fractured", "Fractured"]

    selected_probability = (
        fracture_probability if index == 1
        else 1.0 - fracture_probability
    )

    return (
        labels[index],
        selected_probability,
        fracture_probability,
        index,
    )



def make_gradcam(model, image_array, target_index=None):
    """Generate Grad-CAM for models with a nested CNN backbone and prediction head."""
    try:
        # Find a nested CNN backbone and its final convolution layer.
        backbone = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                conv_layers = [
                    inner for inner in layer.layers
                    if isinstance(inner, tf.keras.layers.Conv2D)
                ]
                if conv_layers:
                    backbone = layer
                    break

        if backbone is None:
            print("Grad-CAM error: no nested CNN backbone was found.")
            return None

        conv_layers = [
            layer for layer in backbone.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        if not conv_layers:
            print("Grad-CAM error: backbone has no Conv2D layer.")
            return None

        # Use the final Conv2D layer (e.g. Conv_1 for MobileNetV2).
        last_conv = conv_layers[-1]
        feature_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[last_conv.output, backbone.output],
            name="gradcam_feature_model",
        )

        # Apply model layers before the backbone (e.g. augmentation).
        x = tf.convert_to_tensor(image_array, dtype=tf.float32)
        backbone_index = model.layers.index(backbone)
        for layer in model.layers[1:backbone_index]:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)

        with tf.GradientTape() as tape:
            conv_outputs, x = feature_model(x, training=False)

            # Apply the original model's prediction head after the backbone.
            for layer in model.layers[backbone_index + 1:]:
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)

            predictions = x
            if predictions.shape.rank != 2:
                print("Grad-CAM error: unexpected prediction tensor shape.")
                return None

            if predictions.shape[-1] == 1:
                probability = predictions[:, 0]
                score = probability if target_index == 1 else 1.0 - probability
            else:
                if target_index is None:
                    target_index = int(tf.argmax(predictions[0]).numpy())
                score = predictions[:, target_index]

        gradients = tape.gradient(score, conv_outputs)
        if gradients is None:
            print("Grad-CAM error: gradients are disconnected from feature map.")
            return None

        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0.0)
        maximum = tf.reduce_max(heatmap)

        if float(maximum.numpy()) <= 1e-8:
            print("Grad-CAM error: heatmap has no positive activations.")
            return None

        return (heatmap / maximum).numpy()

    except Exception as error:
        print(f"Grad-CAM error ({model.name}): {type(error).__name__}: {error}")
        return None


def create_overlay(image, heatmap, image_size):
    """Create an overlay at the chosen model's image dimensions."""
    original = np.asarray(
        image.convert("RGB").resize(image_size)
    )

    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize(image_size)

    heatmap_array = np.asarray(
        heatmap_image, dtype=np.float32
    ) / 255.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original)
    ax.imshow(heatmap_array, cmap="jet", alpha=0.4)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def build_report(
    filename,
    body_part,
    body_confidence,
    finding,
    finding_confidence,
    elapsed_seconds,
    fracture_probability=None,
    threshold=None,
):
    """Build a downloadable text report."""
    lines = [
        "AI X-RAY ANALYSIS ASSISTANT",
        "=" * 36,
        f"File: {filename}",
        f"Analysis time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "BODY-PART CLASSIFICATION",
        f"Prediction: {body_part}",
        f"Model score: {body_confidence:.2%}",
        "",
        "SUPPORTED FINDING MODEL",
        f"Prediction: {finding}",
        f"Selected-class score: {finding_confidence:.2%}",
        f"Processing time: {elapsed_seconds:.3f} seconds",
    ]

    if fracture_probability is not None:
        lines.extend([
            f"Fracture model score: {fracture_probability:.2%}",
            f"Decision threshold: {threshold:.2f}",
        ])

    lines.extend([
        "",
        "IMPORTANT LIMITATIONS",
        "This is an experimental research prototype, not a medical diagnosis.",
        "Model scores are not guaranteed to be calibrated probabilities.",
        "The body-part classifier and fracture model can make mistakes.",
        "Finding predictions are shown only when a compatible trained model is available.",
        "No conclusion about other diseases is made by this application.",
        "Consult a qualified radiologist for medical interpretation.",
    ])

    return "\n".join(lines)


# --------------------------------------------------
# MODEL AVAILABILITY
# --------------------------------------------------

if not BODY_MODEL_PATH.exists():
    st.error(f"Body-part model not found: {BODY_MODEL_PATH}")
    st.info("Check that models/body_part_model_v3.keras exists.")
    st.stop()

if not BODY_LABELS_PATH.exists():
    st.error(f"Body-part labels not found: {BODY_LABELS_PATH}")
    st.info("Check that models/body_part_classes_v3.json exists.")
    st.stop()

try:
    body_model = load_keras_model(BODY_MODEL_PATH)
    body_labels = load_labels(BODY_LABELS_PATH)
except Exception as error:
    st.error(f"Could not load the body-part model: {error}")
    st.stop()


# --------------------------------------------------
# UPLOAD X-RAY
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG or PNG image.",
)

if uploaded_file is None:
    st.info("Upload an X-ray image to begin the analysis.")
    st.markdown(
        """
        **Currently available**
        - Body-part classification using your V3 model.
        - Fracture classification when the body-part prediction routes
          to the supported fracture model.
        - Grad-CAM visualization when compatible with the model.
        - Downloadable analysis report.
        """
    )
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as error:
    st.error(f"Could not read this image: {error}")
    st.stop()

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Uploaded X-ray")
    st.image(image, caption=uploaded_file.name, use_container_width=True)

with right_column:
    st.subheader("Image information")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Original dimensions:** {image.width} × {image.height}")
    st.write("**Body-part model input:** RGB, 224 × 224")
    st.write("**Fracture model input:** RGB, 160 × 160")

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if st.button("Analyze X-ray", type="primary", use_container_width=True):
    start_time = time.perf_counter()

    try:
        body_image_array = preprocess_body_image(image)

        with st.spinner("Analyzing the uploaded image..."):
            predicted_part, body_confidence, body_scores = (
                get_body_part_prediction(
                    body_model,
                    body_image_array,
                    body_labels,
                )
            )

            st.subheader("1. Body-part classification")
            st.metric("Predicted body part", predicted_part)
            st.write(
                f"Model score: **{body_confidence:.2%}** "
                "(not a calibrated clinical confidence)"
            )

            with st.expander("View body-part model scores"):
                st.dataframe(
                    {
                        "Body part": body_labels,
                        "Model score": [
                            f"{float(score):.2%}" for score in body_scores
                        ],
                    },
                    use_container_width=True,
                    hide_index=True,
                )

            finding = "No compatible finding model is configured"
            finding_confidence = 0.0
            fracture_probability = None
            threshold = None
            explanation_model = body_model
            explanation_array = body_image_array
            explanation_size = BODY_IMAGE_SIZE
            explanation_target = int(np.argmax(body_scores))

            if predicted_part == "Chest":
                chest_model_path = MODEL_DIR / "chest_pneumonia_model.keras"
                chest_labels_path = MODEL_DIR / "chest_pneumonia_classes.json"

                if chest_model_path.exists() and chest_labels_path.exists():
                    chest_model = load_keras_model(chest_model_path)
                    chest_labels = load_labels(chest_labels_path)

                    # The training script used 224x224 RGB inputs and
                    # MobileNetV2 preprocess_input inside the saved model.
                    chest_array = np.asarray(
                        image.convert("RGB").resize((224, 224)),
                        dtype=np.float32,
                    )[None, ...]

                    chest_output = np.asarray(
                        chest_model.predict(chest_array, verbose=0)
                    ).reshape(-1)

                    if len(chest_output) != 1:
                        raise ValueError(
                            "Chest model should have one sigmoid output. "
                            "Check its architecture and training code."
                        )

                    pneumonia_score = float(np.clip(chest_output[0], 0.0, 1.0))
                    # Training class order: NORMAL=0, PNEUMONIA=1.
                    # Threshold 0.70 was evaluated on the project's test set.
                    # It reduces false pneumonia alerts but misses more cases.
                    PNEUMONIA_THRESHOLD = 0.70
                    chest_index = (
                        1 if pneumonia_score >= PNEUMONIA_THRESHOLD else 0
                    )
                    threshold = PNEUMONIA_THRESHOLD
                    if len(chest_labels) < 2:
                        chest_labels = ["NORMAL", "PNEUMONIA"]
                    finding = chest_labels[chest_index]
                    finding_confidence = (
                        pneumonia_score if chest_index == 1
                        else 1.0 - pneumonia_score
                    )

                    explanation_model = chest_model
                    explanation_array = chest_array
                    explanation_size = (224, 224)
                    explanation_target = chest_index

                    st.subheader("2. Chest finding model")
                    st.metric("Experimental model output", finding)
                    st.write(
                        f"Selected-class score: **{finding_confidence:.2%}**"
                    )
                    st.write(
                        f"Decision threshold: **{PNEUMONIA_THRESHOLD:.2f}**"
                    )
                    st.caption(
                        "Research-only pneumonia classifier. This score is "
                        "not a calibrated clinical probability or diagnosis."
                    )
                else:
                    finding = "Chest pneumonia model unavailable"
                    st.subheader("2. Chest finding model")
                    st.warning(
                        "Expected models/chest_pneumonia_model.keras and "
                        "models/chest_pneumonia_classes.json."
                    )

            elif predicted_part in {"Hand", "Leg", "Pelvis"}:
                if FRACTURE_MODEL_PATH.exists() and FRACTURE_LABELS_PATH.exists():
                    fracture_model = load_keras_model(FRACTURE_MODEL_PATH)
                    fracture_labels = load_labels(FRACTURE_LABELS_PATH)
                    fracture_image_array = preprocess_fracture_image(image)

                    threshold = 0.5
                    if FRACTURE_THRESHOLD_PATH.exists():
                        with open(
                            FRACTURE_THRESHOLD_PATH, "r", encoding="utf-8"
                        ) as file:
                            threshold_data = json.load(file)
                        if isinstance(threshold_data, dict):
                            threshold = float(
                                threshold_data.get("threshold", threshold)
                            )
                        elif isinstance(threshold_data, (float, int)):
                            threshold = float(threshold_data)

                    (
                        finding,
                        finding_confidence,
                        fracture_probability,
                        fracture_index,
                    ) = get_fracture_prediction(
                        fracture_model,
                        fracture_image_array,
                        fracture_labels,
                        threshold,
                    )

                    explanation_model = fracture_model
                    explanation_array = fracture_image_array
                    explanation_size = FRACTURE_IMAGE_SIZE
                    explanation_target = fracture_index

                    st.subheader("2. Experimental fracture model")
                    st.metric("Model output", finding)
                    st.write(
                        f"Fracture score: **{fracture_probability:.2%}**"
                    )
                    st.write(f"Decision threshold: **{threshold:.2f}**")
                    st.caption(
                        "This model is experimental. Its training data may "
                        "not represent every anatomy or X-ray type."
                    )
                else:
                    finding = "Fracture model unavailable"
                    st.subheader("2. Finding model")
                    st.warning("Fracture model or labels file is missing.")
            else:
                finding = f"No trained finding model for {predicted_part}"
                st.subheader("2. Supported finding model")
                st.info(
                    f"No compatible trained finding model is configured for "
                    f"'{predicted_part}'. The app reports body-part "
                    "classification only rather than inventing a finding."
                )

            st.subheader("3. Grad-CAM visualization")
            heatmap = make_gradcam(
                explanation_model,
                explanation_array,
                target_index=explanation_target,
            )

            if heatmap is not None:
                fig = create_overlay(image, heatmap, explanation_size)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Grad-CAM highlights image regions influencing a model "
                    "output. It is not lesion segmentation or proof of disease."
                )
            else:
                st.info(
                    "Grad-CAM could not be generated for this model "
                    "architecture."
                )

            elapsed_seconds = time.perf_counter() - start_time

            st.subheader("4. Processing summary")
            st.write(f"**Processing time:** {elapsed_seconds:.3f} seconds")

            report = build_report(
                filename=uploaded_file.name,
                body_part=predicted_part,
                body_confidence=body_confidence,
                finding=finding,
                finding_confidence=finding_confidence,
                elapsed_seconds=elapsed_seconds,
                fracture_probability=fracture_probability,
                threshold=threshold,
            )

            st.download_button(
                "Download analysis report",
                data=report,
                file_name="xray_analysis_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.success("Analysis completed.")

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.info(
            "Check that the model files and label JSON files exist. "
            "If the issue continues, send the full terminal error."
        )

st.divider()
st.caption(
    "Research prototype only • Not intended for clinical diagnosis "
    "or treatment decisions."
)
