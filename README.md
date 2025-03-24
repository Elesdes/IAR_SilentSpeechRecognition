# EMG-Based Speech Synthesis: Personalised Wearable for Speech Impaired Communication

## Overview

This is a research-driven project aimed at developing a novel system for **silent speech recognition** using mainly **Electromyography (EMG)** signals & **Liquid Neural Networks**. The goal is to recognize and synthesize speech based on the electrical activity of facial muscles without the need for vocalization.

## Paper

You can find the paper associated to this paper [here](docs/LNN_EMG_Based_Speech_Synthesis.pdf).

## Setup

Here's the setup I used for this project. It's best to use the same one for reproducibility purposes :

### Install Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

After running the above commands, reboot terminal.

### Prepare Conda Environment

```bash
conda update --all -y
conda create -n IAR_SSR python=3.11 -y
conda activate IAR_SSR
yes | pip install -r requirements.txt
```

## System Specifications (for reproducibility)

- **Model:** ROG Zephyrus G16 GU605MZ
- **GPU:** NVIDIA GeForce RTX 4080
  - Memory: 12 GB GDDR6
  - AI Performance: 542 AI TOPs
  - Boost Clock: 1920 MHz
  - Dynamic Boost: 115W
- **Processor:** Intel® Core™ i9-185H Ultra
  - Base Frequency: 2.3 GHz
  - Boost Frequency: Up to 5.1 GHz
  - Cores/Threads: 16 cores, 22 threads
  - Cache: 24 MB
- **Memory (RAM):** 32 GB LPDDR5X 7467 MHz (16 GB x 2)
- **Storage:** Dual SSD M.2 NVMe PCIe® 4.0
- **Operating System:** Ubuntu 24.04
- **Kernel Version:** 6.11.0-19-generic
- **NVIDIA Driver Version:** 550.120
- **CUDA Version:** 12.4
