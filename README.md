<h2 align="center">
  <b><tt>ARNOLD</tt>: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes</b>
</h2>

<div align="center" margin-bottom="6em">
<b>ICCV 2023</b>
</div>

<div align="center" margin-bottom="6em">
Ran Gong<sup>✶</sup>, Jiangyong Huang<sup>✶</sup>, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu <br/> Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, Baoxiong Jia, Siyuan Huang
</div>
&nbsp;

<div align="center">
    <a href="https://arxiv.org/abs/2304.04321" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://arnold-benchmark.github.io" target="_blank">
    <img src="https://img.shields.io/badge/Page-ARNOLD-9cf" alt="Project Page"/></a>
    <a href="https://arnold-docs.readthedocs.io/en/latest/" target="_blank">
    <img src="https://img.shields.io/badge/docs-passing-brightgreen.svg" alt="Documentation"/></a>
    <a href="https://drive.google.com/drive/folders/1yaEItqU9_MdFVQmkKA6qSvfXy_cPnKGA?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/Data-Demos-9966ff" alt="Data"/></a>
    <a href="https://pytorch.org" target="_blank">
    <img src="https://img.shields.io/badge/Code-PyTorch-blue" alt="PyTorch"/></a>
    <a href="https://sites.google.com/view/arnoldchallenge/" target="_blank">
    <img src="https://img.shields.io/badge/Challenge-ARNOLD-orange" alt="PyTorch"/></a>
</div>
&nbsp;

![teaser](assets/teaser.png)

**[News]** We host the [ARNOLD Challenge](https://sites.google.com/view/arnoldchallenge/) on [CVPR 2024 Embodied AI Workshop](https://embodied-ai.org/). Welcome to participate.

We present <tt>ARNOLD</tt>, a benchmark for **language-grounded** task learning with **continuous states** in **realistic 3D scenes**. We highlight the following major points:
- <tt>ARNOLD</tt> is built on <tt>NVIDIA Isaac Sim</tt>, equipped with **photo-realistic** and **physically-accurate** simulation, covering **40 distinctive objects** and **20 scenes**.
- <tt>ARNOLD</tt> is comprised of **8 language-conditioned tasks** that involve understanding object states and learning policies for continuous goals. For each task, there are **7 data splits**, including **unseen generalization**.
- <tt>ARNOLD</tt> provides **10k expert demonstrations** with diverse template-generated language instructions, based on thousands of human annotations.
- We assess the task performances of the latest language-conditioned policy learning models. The results indicate that current models for language-conditioned manipulation **still struggle in understanding continuous states and producing precise motion control**. We hope these findings can foster future research to address the unsolved challenges in **instruction grounding** and **precise continuous motion control**.

We provide brief guidance on this page. Please refer to [our documentation](https://arnold-docs.readthedocs.io/en/latest/) for more information about <tt>ARNOLD</tt>.

## Get Started
There are two setup approaches: docker-based and conda-based. We recommend the docker-based approach as it wraps everything up and is friendly to users. See step-by-step instructions [here](https://arnold-docs.readthedocs.io/en/latest/tutorial/setup/index.html#setup).

After setup, you can refer to [quickstart](https://arnold-docs.readthedocs.io/en/latest/tutorial/setup/index.html#quickstart) for a glance of using <tt>ARNOLD</tt>.

Major components of the <tt>ARNOLD</tt> environment are introduced [here](https://arnold-docs.readthedocs.io/en/latest/tutorial/environment/index.html#environment). Based on this environment, you can check the [tasks](https://arnold-docs.readthedocs.io/en/latest/tutorial/tasks/index.html#tasks) and [data](https://arnold-docs.readthedocs.io/en/latest/tutorial/data/index.html#data).

We use `hydra` for configurations of the experiments. See [configs](https://arnold-docs.readthedocs.io/en/latest/tutorial/configs/index.html#configs). After double-checking the configurations, you can explore the [training] and [evaluation] on your own.

## TODO
- Demonstration generator.

## BibTex
```bibtex
@inproceedings{gong2023arnold,
  title={ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes},
  author={Gong, Ran and Huang, Jiangyong and Zhao, Yizhou and Geng, Haoran and Gao, Xiaofeng and Wu, Qingyang and Ai, Wensi and Zhou, Ziheng and Terzopoulos, Demetri and Zhu, Song-Chun and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
