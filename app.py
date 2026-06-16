import gradio as gr
import numpy as np

from src.inference import predict_array
from src.utils import load_config


MODEL_CHOICES = {
    "U-Net": "unet",
    "Attention U-Net": "attention_unet",
    "U-Net++": "unet_plus_plus",
    "DeepLabV3+": "deeplabv3plus",
    "FPN": "fpn",
}


def run_demo(image, checkpoint_path, config_path, threshold, device, model_label):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not checkpoint_path:
        raise gr.Error("Please provide a checkpoint path.")
    try:
        config = load_config(config_path or "configs/unet.yaml")
        image_rgb = np.asarray(image.convert("RGB"))
        result = predict_array(
            image_rgb,
            config,
            checkpoint_path=checkpoint_path,
            threshold=float(threshold),
            device=device,
            model_name_override=MODEL_CHOICES[model_label],
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    info = (
        f"Lesion area ratio: {result['lesion_ratio']:.4f}\n"
        f"Inference time: {result['inference_time']:.4f}s\n"
        f"Device: {result['device']}"
    )
    return result["image"], result["mask"], result["overlay"], info


def build_app():
    with gr.Blocks(title="Skin Lesion Segmentation") as demo:
        gr.Markdown("# Skin Lesion Segmentation")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Input image")
                checkpoint = gr.Textbox(label="Checkpoint path", placeholder="checkpoints/best_model.pth")
                config = gr.Textbox(label="Config path", value="configs/unet.yaml")
                threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Threshold")
                device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                model = gr.Dropdown(list(MODEL_CHOICES.keys()), value="U-Net", label="Model")
                button = gr.Button("Predict", variant="primary")
            with gr.Column():
                out_image = gr.Image(label="Resized original")
                out_mask = gr.Image(label="Predicted mask")
                out_overlay = gr.Image(label="Overlay")
                info = gr.Textbox(label="Result", lines=4)
        button.click(run_demo, [image, checkpoint, config, threshold, device, model], [out_image, out_mask, out_overlay, info])
    return demo


if __name__ == "__main__":
    build_app().launch()

