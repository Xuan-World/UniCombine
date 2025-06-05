import gradio as gr
import subprocess
import os

with gr.Blocks() as demo:
    gr.Markdown("### UniCombine 推理界面")
    with gr.Row():
        with gr.Column():
            condition_lora = gr.CheckboxGroup(["fill", "subject", "canny", "depth"], label="Condition LoRA 选择")
            version = gr.Radio(["training-based", "training-free"], label="版本", value="training-based")
            denoising_lora = gr.CheckboxGroup(["subject_fill_union", "depth_canny_union", "subject_depth_union", "subject_canny_union"], label="Denoising LoRA 选择")
            denoising_lora_weight = gr.Number(value=1.0, label="Denoising LoRA 权重")
            fill = gr.Image(type="pil", label="背景图", visible=False)
            subject = gr.Image(type="pil", label="主体图", visible=False)
            canny = gr.Image(type="pil", label="边缘图", visible=False)
            depth = gr.Image(type="pil", label="深度图", visible=False)
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button("运行推理")
        with gr.Column():
            output_image = gr.Image(label="推理结果")
            combination_info = gr.Textbox(label="传递的组合方式")

    def update_image_inputs(condition_types):
        fill_visible = "fill" in condition_types
        subject_visible = "subject" in condition_types
        canny_visible = "canny" in condition_types
        depth_visible = "depth" in condition_types
        return gr.update(visible=fill_visible), gr.update(visible=subject_visible), gr.update(visible=canny_visible), gr.update(visible=depth_visible)

    def run_inference(condition_types, denoising_lora, denoising_lora_weight, fill, subject, canny, depth, prompt, version):
        command = [
            "python", "inference.py",
            "--condition_types", " ".join(condition_types),
            "--denoising_lora", denoising_lora,
            "--denoising_lora_weight", str(denoising_lora_weight),
            "--prompt", prompt,
            "--version", version
        ]

        if "fill" in condition_types and fill:
            command.extend(["--fill", fill.name if hasattr(fill, 'name') else fill])
        if "subject" in condition_types and subject:
            command.extend(["--subject", subject.name if hasattr(subject, 'name') else subject])
        if "canny" in condition_types and canny:
            command.extend(["--canny", canny.name if hasattr(canny, 'name') else canny])
        if "depth" in condition_types and depth:
            command.extend(["--depth", depth.name if hasattr(depth, 'name') else depth])

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            work_dir = "output/inference_result"
            output_dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
            output_dirs.sort()
            latest_dir = os.path.join(work_dir, output_dirs[-1])
            result_images = [f for f in os.listdir(latest_dir) if f.endswith("_result.jpg")]
            result_images.sort()
            latest_image = os.path.join(latest_dir, result_images[-1])
            combination = ", ".join(condition_types)
            return latest_image, combination
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}", ""

    condition_lora.change(
        fn=update_image_inputs,
        inputs=condition_lora,
        outputs=[fill, subject, canny, depth]
    )

    run_button.click(
        fn=run_inference,
        inputs=[condition_lora, denoising_lora, denoising_lora_weight, fill, subject, canny, depth, prompt, version],
        outputs=[output_image, combination_info]
    )

demo.launch(share=True)