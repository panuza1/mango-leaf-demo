from pathlib import Path
import os
import gradio as gr
from inference import Classifier

ROOT = Path(__file__).resolve().parent
classifier = Classifier(ROOT / "models")

def predict(image):
    if image is None:
        raise gr.Error("Please upload a mango leaf image.")
    result = classifier.predict(image)
    return tuple(result[name] for name in ("SVM", "CNN", "GoogLeNet"))

with gr.Blocks(title="Mango Leaf AI") as demo:
    gr.Markdown("# Mango Leaf AI\nCompare three trained models for mango leaf disease classification.")
    gr.Markdown("**Real inference · 8 classes · CPU** — Scores are model outputs, not guaranteed diagnosis. External test performance is limited.")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Upload a mango leaf", height=320)
            button = gr.Button("Classify image", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("## Model comparison")
            with gr.Row():
                outputs = [gr.Label(label=name, num_top_classes=5) for name in ("SVM", "CNN", "GoogLeNet")]
    button.click(predict, inputs=image, outputs=outputs, api_name="predict")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green"),
                server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.getenv("PORT", "7860")), share=False)
