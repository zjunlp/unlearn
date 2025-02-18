# ReLearn: Unlearning via Learning  for Large Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2502.11190">📄arXiv</a> •
  <a href="https://huggingface.co/papers/2502.11190">🤗HFPaper</a>
</p>

This repository provides the official PyTorch implementation of the following paper: 
> [**ReLearn: Unlearning via Learning  for Large Language Models**]() <br>
> Haoming Xu<sup>1</sup>,  Ningyuan Zhao<sup>2</sup>,  Liming Yang<sup>3</sup>,  
> Sendong Zhao<sup>4</sup>,  Shumin Deng<sup>5</sup>,  Mengru Wang<sup>1</sup>,  
> Bryan Hooi<sup>5</sup>,  Nay Oo<sup>5</sup>,  Huajun Chen<sup>1</sup>,  Ningyu Zhang<sup>1</sup> <br> 
> <sup>1</sup>Zhejiang University,<sup>2</sup>Xiamen University, <sup>3</sup>Tsinghua University, <sup>4</sup>Harbin Institute of Technology <br>, <sup>5</sup>National University of Singapore

## 🌟Overview

## 🔧Installation

```bash
conda create -n relearn python=3.10.15
conda activate relearn
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 📚Augument data
```bash
cd dataAugument
bash augu.sh
```

## 🚀Finetune models
knowundo currently supports `Llama3-8b instruct`, `gemma2-2b-it`, and `Llama2-7b chat` models.
```bash
cd baselines/pretrain_scripts/
bash kud-pt.sh
```

## 🤷Forget models
```bash
cd baselines/unlearn_scripts/
bash kud-relearn.sh
```

## 🕵️Evaluate models
merge adapter -> inference -> evaluate
```bash
cd evals
bash merge_all.sh
bash inf_all.sh
bash eval_all.sh
```

## 📂 Open Resources

- **Llama-2-7b-chat-KnowUnDo-Privacy (Vanilla Model)**
   [Download here](https://www.modelscope.cn/models/haomingx/Llama-2-7b-chat-KnowUnDo-Privacy/files)
- **Llama-2-7b-chat-TOFU-Forget10-ReLearn**
   [Access on Google Drive](https://drive.google.com/drive/folders/1wsPKpF2IZ4RC52_PI7ILhYsegtqZG25Y?usp=drive_link)
- **Llama-2-7b-chat-KnowUnDo-Privacy-ReLearn**
   [Access on Google Drive](https://drive.google.com/drive/folders/1delWVv3VnoU7XcofOW-xUs4SiiXYJIcR?usp=drive_link)

## Acknowledgement
The repository references the code from [TOFU](https://github.com/locuslab/tofu) and [MUSE](https://github.com/jaechan-repo/muse_bench). We extend our gratitude to the authors for their outstanding work.


## Citation
If you find this work useful for your research, please cite [our paper](https://arxiv.org/abs/2502.11190):
```
@misc{xu2025relearnunlearninglearninglarge,
      title={ReLearn: Unlearning via Learning for Large Language Models}, 
      author={Haoming Xu and Ningyuan Zhao and Liming Yang and Sendong Zhao and Shumin Deng and Mengru Wang and Bryan Hooi and Nay Oo and Huajun Chen and Ningyu Zhang},
      year={2025},
      eprint={2502.11190},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11190}, 
}

```