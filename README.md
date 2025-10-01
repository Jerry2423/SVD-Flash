# SVD-Flash: Efficient LLM Inference via SVD Compression and Tiling on AWS Trainium

## Overview
SVD-Flash is a high-performance matrix multiplication (matmul) kernel for Large Language Model (LLM) inference on AWS Trainium. It is a combination of Singular Value Decomposition (SVD) compression and architecture-specific optimizations to accelerate LLM inference. SVD-Flash is designed to reduce data movement across the software-managed memory hierarchy, maximize SRAM bandwidth, and avoid expensive matrix transposes. The matrix multiplication kernel, NeuronMM, is open-sourced and can be found in our [Kernel repository](https://github.com/Jerry2423/neuron-mm). This repository provides the implementation for running inference with SVD-Flash.

![SVD-Flash: Efficient LLM inference via SVD Compression and Tiling on AWS Trainium](./images/svd_flash_1.png)

## Model Acceleration
We evaluated SVD-Flash with nine datasets and four recent LLMs, and the results show that our system largely outperforms the state-of-the-art models implemented by AWS on Trainium.

- **Kernel-Level Speedup:** At the matmul kernel level, SVD-Flash achieves an average 1.35x speedup, with a maximum of 2.22x.
- **End-to-End Inference Speedup:** This translates to an average 1.66x speedup for end-to-end LLM inference, with a maximum of 2.49x.
- **Reduced Memory Traffic:** SVD-Flash significantly reduces memory traffic. Compared to the NKI XW baseline, it achieves a 4.78x reduction in HBM-SBUF memory traffic at a sequence length of 32K.
- **High Tensor Engine Utilization:** SVD-Flash sustains high tensor engine active time and Model Float Utilization (MFU), with the tensor engine MFU reaching 85.20%, compared to 65.24% for the standard sequential kernel.

The table below summarizes the end-to-end inference speedup for various Large Language Models (LLMs) using NeuronMM at different compression ratios.
| Model | Compression Ratio | Average Speedup (↑) |
| :--- | :--- | :--- |
| Llama-3.2-1B | 0.10  | 1.21x  |
| | 0.20  | 1.63x  |
| Llama-3.2-3B | 0.10  | 1.88x  |
| | 0.20  | 2.49x  |
| Qwen-3-1.7B | 0.10  | 1.41x  |
| | 0.20  | 1.74x  |
| Qwen-3-4B | 0.10  | 1.28x  |
| | 0.20  | 1.67x  |

## Setup Steps

1. Launch a Tranium instance using [AWS EC2](https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#LaunchInstances:) with the following settings:  
   i. **Name and tags**: SVD-Flash  
   ii. **Amazon Machine Image**: Deep Learning AMI Neuron (Ubuntu 22.04)  
   iii. **Instance type**: trn1.2xlarge  
   iv. **Key pair (login)**: create a new key pair  
   v. **Metadata version [under “Advanced details”]**: V2 only (otherwise, you will encounter a not authorized error)  
   vi. When connecting to these instances via SSH, use the username of *ubuntu*.

2. Activate the Neuron virtual environment to run inference by running  
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate

3. Download `Llama-3.2-1B` from Hugging face
    ``` 
    mkdir models

    huggingface-cli download --token <YOUR_HUGGINGFACE_TOKEN> meta-llama/Llama-3.2-1B --local-dir ./models/llama-3.2-1b

    cd /home/ubuntu/models/llama-3.2-1b

    mv model.safetensors  model_ori.safetensors

4. Download the weights after SVD and post-training processing
   ```
   wget "https://huggingface.co/SVD-Flash/llama-3.2-1b_0.8_svd/resolve/main/llama-3.2-1b_svd_0.8_weights.safetensors?download=true" \
     -O model.safetensors   

5. Download the v0.0.1 repo
   ```
   cd ~   
   git clone -b v0.0.1 --single-branch https://github.com/dinghongsong/SVD-Flash.git


5. Testing Example (Without Tensor Parallelism): Llama inference with logit matching accuracy check using custom error tolerances
   ```
   python llama_inference.py \
    --model-type llama \
    --task-type causal-lm \
    run \
    --model-path /home/ubuntu/models/llama-3.2-1b \
    --compiled-model-path /home/ubuntu/traced_model/llama-3.2-1b \
    --torch-dtype bfloat16 \
    --batch-size 1 \
    --max-context-length 32 \
    --seq-len 64 \
    --check-accuracy-mode logit-matching \
    --divergence-difference-tol 0.005 \
    --tol-map "{5: (1e-5, 0.02)}" \
    --enable-bucketing \
    --top-k 1 \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is" \
    --compress-ratio 0.8

## Output Example
   ```
   ------------------------------------------------------------------------------------------
model:  /home/ubuntu/models/llama-3.2-1b
{
    "e2e_model": {
        "latency_ms_p50": 1299.866795539856,
        "latency_ms_p90": 1301.309323310852,
        "latency_ms_p95": 1302.1685719490051,
        "latency_ms_p99": 1302.9363083839417,
        "latency_ms_p100": 1303.1282424926758,
        "latency_ms_avg": 1300.0563144683838,
        "throughput": 49.22863670422672
    },
    "context_encoding_model": {
        "latency_ms_p50": 70.28031349182129,
        "latency_ms_p90": 70.3099250793457,
        "latency_ms_p95": 70.31856775283813,
        "latency_ms_p99": 70.3455662727356,
        "latency_ms_p100": 70.35231590270996,
        "latency_ms_avg": 70.27335166931152,
        "throughput": 455.3646473357901
    },
    "token_generation_model": {
        "latency_ms_p50": 39.081573486328125,
        "latency_ms_p90": 39.14194107055664,
        "latency_ms_p95": 39.16501998901367,
        "latency_ms_p99": 39.197025299072266,
        "latency_ms_p100": 39.25800323486328,
        "latency_ms_avg": 39.088456092342255,
        "throughput": 26.40825879839129
    }
}
------------------------------------------------------------------------------------------
model:  /home/ubuntu/models/llama-3.2-1b/svd_llama
{
    "e2e_model": {
        "latency_ms_p50": 893.8226699829102,
        "latency_ms_p90": 894.6243762969971,
        "latency_ms_p95": 894.8212623596191,
        "latency_ms_p99": 895.5112648010254,
        "latency_ms_p100": 895.683765411377,
        "latency_ms_avg": 893.8456416130066,
        "throughput": 71.60072950012662
    },
    "context_encoding_model": {
        "latency_ms_p50": 66.6283369064331,
        "latency_ms_p90": 66.76206588745117,
        "latency_ms_p95": 66.76559448242188,
        "latency_ms_p99": 66.77356719970703,
        "latency_ms_p100": 66.77556037902832,
        "latency_ms_avg": 66.64743423461914,
        "throughput": 480.1385134700057
    },
    "token_generation_model": {
        "latency_ms_p50": 26.091694831848145,
        "latency_ms_p90": 26.137852668762207,
        "latency_ms_p95": 26.164603233337402,
        "latency_ms_p99": 26.198863983154297,
        "latency_ms_p100": 26.267528533935547,
        "latency_ms_avg": 26.096261316730132,
        "throughput": 39.55578356560814
    }
}
e2e_model time wo svd:  1300.0563144683838
e2e_model time with svd:  893.8456416130066
E2E Speedup:  1.4544528204247231


