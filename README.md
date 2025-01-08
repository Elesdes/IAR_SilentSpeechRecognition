# EMG-Based Speech Synthesis: Personalised Wearable for Speech Impaired Communication

## Overview

This is a research-driven project aimed at developing a novel system for **silent speech recognition** using mainly **Electromyography (EMG)** signals. The goal is to recognize and synthesize speech based on the electrical activity of facial muscles without the need for vocalization.

> [!NOTE]
> This project is currently in the research and development phase. Stay tuned for updates on the project's progress !

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
