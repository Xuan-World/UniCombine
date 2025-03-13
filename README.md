<div align="center">
<h1> UniCombine </h1>

<h3>UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer</h3>
<b>Haoxuan Wang</b>, Jinlong Peng, Qingdong He, Hao Yang, Ying Jin, <br>
 Jiafu Wu, Xiaobin Hu, Yanjie Pan, Zhenye Gan, Mingmin Chi, Bo Peng, Yabiao Wang <br>
<br>
<a href="https://arxiv.org/abs/2503.09277"><img src="https://img.shields.io/badge/arXiv-2503.09277-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/Xuan-World/UniCombine"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/Xuan-World/SubjectSpatial200K"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>
</div>

## 🌠 Key Features

<img src='assets/cover.png' width='100%' />
<br>
Fantastic results of our proposed UniCombine on multi-conditional controllable generation: <br>

- (a) Subject-Insertion task. 
- (b) and (c) Subject-Spatial task. 
- (d) Multi-Spatial task.

Our unified framework effectively handles any combination of input conditions and achieves remarkable alignment with all of them, including but not limited to text prompts, spatial maps, and subject images.

## 🚩 **Updates**

- ✅ March 12, 2025. We release SubjectSpatial200K dataset.
- ✅ March 12, 2025. We release UniCombine framework.

## 🔧 Dependencies and Installation

```bash
conda create -n unicombine python=3.12
conda activate unicombine
pip install -r requirements.txt
```
## 📥 Download Models 
Place all the model weights in the `ckpt` directory. Of course, it's also acceptable to store them in other directories.
1. **FLUX.1-schnell**
```bash
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir ./ckpt/FLUX.1-schnell
```
2. **Condition-LoRA**
```bash
huggingface-cli download Xuan-World/UniCombine --include "Condition_LoRA/*" --local-dir ./ckpt/Condition_LoRA
```

3. **Denoising-LoRA**
```bash
huggingface-cli download Xuan-World/UniCombine --include "Denoising_LoRA/*" --local-dir ./ckpt/Denoising_LoRA
```

## 🗂️ Download Dataset （optional）
Place our SubjectSpatial200K dataset in the `dataset` directory. Of course, it's also acceptable to store them in other directories. <br>
The dataset can be used in the "**Train**" and "**Test**" part.  It may takes a long time.
```bash
huggingface-cli download Xuan-World/SubjectSpatial200K --repo-type dataset --local-dir ./dataset
```

## 🎮 Inference on Demo
- We provide the `inference.py` script to offer a simplest and fastest way for you to run our model. <br>
- Replace the arguments `--version` from `training-based` to `training-free`, then you don't need to provide the **Denoising-LoRA** module.
- Adjust the scale of `--denoising_lora_weight` to get a balance between the editability and the consistency when using Custom Prompts.
### 1. Subject-Insertion
Default Prompts：
```bash
python inference.py \
--condition_types fill subject \
--denoising_lora ckpt/Denoising_LoRA/subject_fill_union \
--denoising_lora_weight 1.0 \
--fill examples/window/background.jpg \
--subject examples/window/subject.jpg \
--json "examples/window/1634_rank0_A decorative fabric topper for windows..json" \
--version training-based
```

### 2. Subject-Canny
Default Prompts：
```bash
python inference.py \
--condition_types canny subject \
--denoising_lora ckpt/Denoising_LoRA/subject_canny_union \
--denoising_lora_weight 1.0 \
--canny examples/doll/canny.jpg \
--subject examples/doll/subject.jpg \
--json "examples/doll/1116_rank0_A spooky themed gothic doll..json" \
--version training-based
```
Custom Prompts：
```bash
python inference.py \
--condition_types canny subject \
--denoising_lora ckpt/Denoising_LoRA/subject_canny_union \
--denoising_lora_weight 0.6 \
--canny examples/doll/canny.jpg \
--subject examples/doll/subject.jpg \
--json "examples/doll/1116_rank0_A spooky themed gothic doll..json" \
--version training-based \
--prompt "She stands amidst the vibrant glow of a bustling Chinatown alley, \
her pink hair shimmering under festive lantern light, clad in a sleek black dress adorned with intricate lace patterns. "
```
### 3. Subject-Depth
Default Prompts：
```bash
python inference.py \
--condition_types depth subject \
--denoising_lora ckpt/Denoising_LoRA/subject_depth_union \
--denoising_lora_weight 1.0 \
--depth examples/car/depth.jpg \
--subject examples/car/subject.jpg \
--json "examples/car/2532_rank0_A sturdy ATV with rugged looks..json" \
--version training-based
```
Custom Prompts：
```bash
python inference.py \
--condition_types depth subject \
--denoising_lora ckpt/Denoising_LoRA/subject_depth_union \
--denoising_lora_weight 0.6 \
--depth examples/car/depth.jpg \
--subject examples/car/subject.jpg \
--json "examples/car/2532_rank0_A sturdy ATV with rugged looks..json" \
--version training-based \
--prompt "It is positioned on a snow-covered path in a forest, its green body dusted with frost and black tires caked with packed snow. \
The vehicle retains its sturdy build with handlebars glinting ice particles and headlights cutting through falling snowflakes, surrounded by tall pine trees draped in white."
```
### 4. Depth-Canny
Default Prompts：
```bash
python inference.py \
--condition_types depth canny \
--denoising_lora ckpt/Denoising_LoRA/depth_canny_union \
--denoising_lora_weight 1.0 \
--depth examples/toy/depth.jpg \
--canny examples/toy/canny.jpg \
--json "examples/toy/1616_rank0_A soft, plush toy with cuddly features..json" \
--version training-based
```
Custom Prompts：
```bash
python inference.py \
--condition_types depth canny \
--denoising_lora ckpt/Denoising_LoRA/depth_canny_union \
--denoising_lora_weight 0.6 \
--depth examples/toy/depth.jpg \
--canny examples/toy/canny.jpg \
--json "examples/toy/1616_rank0_A soft, plush toy with cuddly features..json" \
--version training-based \
--prompt "It sits on a moonlit sandy beach, a small sandcastle partially washed by gentle tides beside it, \
under a night sky where the full moon casts silvery trails across waves, with distant seagulls gliding through star-dappled darkness."
```

## 🌱 Train
Use our SubjectSpatial200K dataset or your customized multi-conditional dataset to train your Denoising-LoRA module. 
1. Configure Accelerate Environment
```bash
accelerate config
```
2. Launch Distributed Training
```bash
accelerate launch train.py
```

## 📊 Batch Inference on Dataset
- We provide a script for batch inference on the SubjectSpatial200K dataset in both training-free and training-based version.
- It can also be run on your custom datasets through your Dataset and DataLoader implementations.
```bash
python test.py
```

## Citation
```

```
