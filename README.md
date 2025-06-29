<div align="center">
<h1> UniCombine </h1>

<h3>UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer</h3>
<b>Haoxuan Wang</b>, Jinlong Peng, Qingdong He, Hao Yang, Ying Jin, <br>
 Jiafu Wu, Xiaobin Hu, Yanjie Pan, Zhenye Gan, Mingmin Chi, Bo Peng, Yabiao Wang <br>
<br>
<a href="https://arxiv.org/abs/2503.09277"><img src="https://img.shields.io/badge/arXiv-2503.09277-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/Xuan-World/UniCombine"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/Xuan-World/SubjectSpatial200K"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>

<h3>🎉 Accepted by ICCV 2025</h3>
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
Due to the issues of _diffusers_ library, you need to update the `cite-package` code manually.
You can find the location of your _diffusers_ library by running the following command.
```bash
pip show diffusers
```

Then add the following entry to the dictionary `_SET_ADAPTER_SCALE_FN_MAPPING` located in `diffusers/loaders/peft.py`:
```
"UniCombineTransformer2DModel": lambda model_cls, weights: weights
```

## 📥 Download Models 
Place all the model weights in the `ckpt` directory. Of course, it's also acceptable to store them in other directories.
1. **FLUX.1-schnell**
```bash
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir ./ckpt/FLUX.1-schnell
```
2. **Condition-LoRA**
```bash
huggingface-cli download Xuan-World/UniCombine --include "Condition_LoRA/*" --local-dir ./ckpt
```

3. **Denoising-LoRA**
```bash
huggingface-cli download Xuan-World/UniCombine --include "Denoising_LoRA/*" --local-dir ./ckpt
```

4. FLUX.1-schnell-training-assistant-LoRA (optional) 

Download it if you want to train your LoRA on the FLUX-schnell.

```bash
huggingface-cli download ostris/FLUX.1-schnell-training-adapter --local-dir ./ckpt/FLUX.1-schnell-training-adapter
```

> Schnell is a step distilled model, meaning it can generate an image in just a few steps. 
> However, this makes it impossible to train on it directly because every step you train breaks down the compression more and more. 
> With this adapter enabled during training, that doesnt happen. 
> It is activated during the training process, and disabled during sampling. 
> After the LoRA is trained, this adapter is no longer needed.

## 🎮 Inference on Demo
- We provide the `inference.py` script to offer a simplest and fastest way for you to run our model. <br>
- Replace the arguments `--version` from `training-based` to `training-free`, then you don't need to provide the **Denoising-LoRA** module.
- Adjust the scale of `--denoising_lora_weight` to get a balance between the editability and the consistency when using Custom Prompts.
- News! We now provide the Gradio APP for you. Run the `app.py` and try it out!
### 1. Subject-Insertion
Default Prompts：
```bash
python inference.py \
--condition_types fill subject \
--denoising_lora_name subject_fill_union \
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
--denoising_lora_name subject_canny_union \
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
--denoising_lora_name subject_canny_union \
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
--denoising_lora_name subject_depth_union \
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
--denoising_lora_name subject_depth_union \
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
--denoising_lora_name depth_canny_union \
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
--denoising_lora_name depth_canny_union \
--denoising_lora_weight 0.6 \
--depth examples/toy/depth.jpg \
--canny examples/toy/canny.jpg \
--json "examples/toy/1616_rank0_A soft, plush toy with cuddly features..json" \
--version training-based \
--prompt "It sits on a moonlit sandy beach, a small sandcastle partially washed by gentle tides beside it, \
under a night sky where the full moon casts silvery trails across waves, with distant seagulls gliding through star-dappled darkness."
```

## 🗂️ Download Dataset （optional）
1. Download SubjectSpatial200K

Place our SubjectSpatial200K dataset in the `dataset` directory. Of course, it's also acceptable to store them in other directories. <br>

```bash
huggingface-cli download Xuan-World/SubjectSpatial200K --repo-type dataset --local-dir ./dataset
```

2. Filter and Partition the SubjectSpatial200K dataset into training and testing sets.

The default partition scheme is identical to our paper.
You can customize it as you wish.

```bash
python src/partition_dataset.py \
--dataset dataset/SubjectSpatial200K/data_labeled \
--output_dir dataset/split_SubjectSpatial200K \
--partition train
```
```bash
python src/partition_dataset.py \
--dataset dataset/SubjectSpatial200K/Collection3/data_labeled \
--output_dir dataset/split_SubjectSpatial200K/Collection3 \
--partition train
```
```bash
python src/partition_dataset.py \
--dataset dataset/SubjectSpatial200K/data_labeled \
--output_dir dataset/split_SubjectSpatial200K \
--partition test
```
```bash
python src/partition_dataset.py \
--dataset dataset/SubjectSpatial200K/Collection3/data_labeled \
--output_dir dataset/split_SubjectSpatial200K/Collection3 \
--partition test
```
## 🧩 Train in single-conditional setting
You can train custom Condition-LoRA models using the script in `demo_Condition_LoRA`. This simple process lets you develop new Condition-LoRA modules that extend UniCombine's features.

Here we provide you an operation process to train a Style-LoRA. It won't be time-consuming because the training dataset is only about 10K images.

1. Download the style transfer dataset proposed by StyleBooth.

```
https://modelscope.cn/models/iic/stylebooth/files
```
>It is worth noting that we fix the format errors of `train.csv` in the StyleBooth Dataset and provide you the fixed one at `demo_Condition_Lora/annotation.csv`.


2. Write a custom Dataset Class like `demo_Condition_Lora/stylebooth_loader.py`

3. Run `demo_Condition_Lora/train_cond_lora.py`.


## 🔥 Train in multi-conditional setting
Use our SubjectSpatial200K dataset or your customized multi-conditional dataset to train your **Denoising-LoRA** module. 
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

## 📚 Citation
```
@article{wang2025unicombine,
  title={UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer},
  author={Wang, Haoxuan and Peng, Jinlong and He, Qingdong and Yang, Hao and Jin, Ying and Wu, Jiafu and Hu, Xiaobin and Pan, Yanjie and Gan, Zhenye and Chi, Mingmin and others},
  journal={arXiv preprint arXiv:2503.09277},
  year={2025}
}
```
