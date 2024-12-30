# sd-scripts-docker

Dockerfile for [aoirint/sd-scripts](https://github.com/aoirint/sd-scripts), a personal fork of [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).

## Differences from the original repository

This patch is applied. Technical validity is not guaranteed.

- https://github.com/aoirint/sd-scripts/pull/1

## Requirements

- Docker Engine >= 27.0
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Usage

- Replace `accelerate launch` with `sudo docker run --rm --gpus all aoirint/sd_scripts`.
- Training command will run in the container by a general user (UID=1000).

### Example: LoRA-LierLa training with `DreamBooth、class+identifier方式` for Waifu Diffusion 1.5 Beta 2.

Create permanent directories to mount on container.

```shell
mkdir -p "./base_model" "./work" "./cache/huggingface/hub"
sudo chown -R 1000:1000 "./base_model" "./work" "./cache/huggingface/hub"
```

Download `wd-1-5-beta2-fp32.safetensors` from [waifu-diffusion/wd-1-5-beta2](https://huggingface.co/waifu-diffusion/wd-1-5-beta2).

```shell
wget "https://huggingface.co/waifu-diffusion/wd-1-5-beta2/resolve/main/checkpoints/wd-1-5-beta2-fp32.safetensors"
echo "764f93581d80b46011039bb388e899f17f7869fce7e7928b060e9a5574bd8f84  wd-1-5-beta2-fp32.safetensors" | sha256sum -c -
```

Prepare a dataset directory `work/my_dataset-20230715.1` and a config file `work/my_dataset-20230715.1/config.toml` following [train_README](https://github.com/kohya-ss/sd-scripts/blob/v0.6.4/train_README-ja.md#dreamboothclassidentifier%E6%96%B9%E5%BC%8F%E6%AD%A3%E5%89%87%E5%8C%96%E7%94%BB%E5%83%8F%E4%BD%BF%E7%94%A8%E5%8F%AF). Set file ownership `UID:GID = 1000:1000` (`sudo chown -R 1000:1000 "./work"`). You can also choose another directory structure to modify `config.toml` and the training command.

- work/my_dataset-20230715.1/
    - config.toml
    - img/
        - 0001.png
        - 0002.png
        - ...
    - reg_img/
        - transparent_1.png
        - transparent_2.png
        - ...
    - output/
    - logs/

Here is a example `config.toml`.

```toml
[general]
enable_bucket = true

[[datasets]]
resolution = 768
batch_size = 4

  [[datasets.subsets]]
  image_dir = '/work/my_dataset-20230715.1/img'
  class_tokens = 'shs girl'
  num_repeats = 10

  [[datasets.subsets]]
  is_reg = true
  image_dir = '/work/my_dataset-20230715.1/reg_img'
  class_tokens = 'girl'
  num_repeats = 1
```

Execute training.

```shell
sudo docker run --rm --gpus all \
  -v "./base_model:/base_model" \
  -v "./work:/work" \
  -v "./cache/huggingface/hub:/huggingface/hub" \
  aoirint/sd_scripts \
  --num_cpu_threads_per_process 1 \
  train_network.py \
  --pretrained_model_name_or_path=/base_model/wd-1-5-beta2-fp32.safetensors \
  --dataset_config=/work/my_dataset-20230715.1/config.toml \
  --output_dir=/work/my_dataset-20230715.1/output \
  --output_name=my_dataset-20230715.1 \
  --save_model_as=safetensors \
  --logging_dir=/work/my_dataset-20230715.1/logs \
  --prior_loss_weight=1.0 \
  --max_train_steps=400 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --xformers \
  --mixed_precision="fp16" \
  --cache_latents \
  --gradient_checkpointing \
  --save_every_n_epochs=1 \
  --network_module=networks.lora \
  --v2 \
  --v_parameterization
```

### Example: WD14 Captioning (Tensorflow)

```shell
mkdir -p "./cache/wd14_tagger_model_cache"
sudo chown -R 1000:1000 "./cache/wd14_tagger_model_cache"

# If your cache is broken, execute
# rm -rf ./cache/wd14_tagger_model_cache/wd14_tagger_model

sudo docker run --rm --gpus all \
  -v "./work:/work" \
  -v "./cache/wd14_tagger_model_cache:/wd14_tagger_model_cache" \
  aoirint/sd_scripts \
  finetune/tag_images_by_wd14_tagger.py \
  --model_dir "/wd14_tagger_model_cache/wd14_tagger_model" \
  /work/my_dataset-20230715.1/img
```
