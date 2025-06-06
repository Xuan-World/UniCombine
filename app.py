import gradio as gr
import subprocess
import os
import types
from inference import inference
with gr.Blocks() as demo:
    gr.Markdown("### UniCombine 推理界面")
    with gr.Row():
        with gr.Column():
            run_button = gr.Button("运行推理")
            prompt = gr.Textbox(label="Prompt")
            condition_lora = gr.CheckboxGroup(["fill", "subject", "canny", "depth"], label="Condition LoRA 选择")
            version = gr.Radio(["training-based", "training-free"], label="版本", value="training-based")
            denoising_lora = gr.Radio(["subject_fill_union", "depth_canny_union", "subject_depth_union", "subject_canny_union"], label="Denoising LoRA 选择")
            denoising_lora_weight = gr.Number(value=1.0, label="Denoising LoRA 权重")
            fill = gr.Image(type="pil", label="背景图", visible=False)
            subject = gr.Image(type="pil", label="主体图", visible=False)
            canny = gr.Image(type="pil", label="边缘图", visible=False)
            depth = gr.Image(type="pil", label="深度图", visible=False)

        with gr.Column():
            output_image = gr.Image(label="推理结果")

    def update_image_inputs(condition_types):
        fill_visible = "fill" in condition_types
        subject_visible = "subject" in condition_types
        canny_visible = "canny" in condition_types
        depth_visible = "depth" in condition_types
        return gr.update(visible=fill_visible), gr.update(visible=subject_visible), gr.update(visible=canny_visible), gr.update(visible=depth_visible)
    def update_training_free_or_based(version):
        flag = (version == "training-based")
        return gr.update(visible=flag),gr.update(visible=flag)
    def run_inference(prompt, condition_types,version, denoising_lora, denoising_lora_weight, fill, subject, canny, depth):
        # 创建参数对象
        args = types.SimpleNamespace(
            pretrained_model_name_or_path="ckpt/FLUX.1-schnell",
            transformer="ckpt/FLUX.1-schnell/transformer",
            condition_types=condition_types,
            denoising_lora_name=denoising_lora,
            denoising_lora_weight=denoising_lora_weight,
            denoising_lora_dir = "ckpt/Denoising_LoRA",
            condition_lora_dir="ckpt/Condition_LoRA",
            work_dir="output/inference_result",
            seed=0,
            resolution=512,
            canny=canny,
            depth=depth,
            fill=fill,
            subject=subject,
            prompt=prompt,
            version=version,
            revision=None,
            variant=None
        )
        result,_ = inference(args)
        return result

    condition_lora.change(
        fn=update_image_inputs,
        inputs=condition_lora,
        outputs=[fill, subject, canny, depth]
    )
    version.change(
        fn = update_training_free_or_based,
        inputs = version,
        outputs = [denoising_lora,denoising_lora_weight]
    )
    run_button.click(
        fn=run_inference,
        inputs=[prompt, condition_lora, version, denoising_lora, denoising_lora_weight, fill, subject, canny, depth],
        outputs=output_image
    )

demo.launch(share=True)