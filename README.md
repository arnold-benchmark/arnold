<h2 align="center">
  <b><tt>ARNOLD</tt>: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes</b>
</h2>

<div align="center" margin-bottom="6em">
Ran Gong<sup>✶</sup>, Jiangyong Huang<sup>✶</sup>, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu <br/> Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, Baoxiong Jia, Siyuan Huang
</div>
&nbsp;

<div align="center">
    <a href="placeholder" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://arnold-benchmark.github.io" target="_blank">
    <img src="https://img.shields.io/badge/Page-ARNOLD-9cf" alt="Project Page"/></a>
    <a href="https://pytorch.org" target="_blank">
    <img src="https://img.shields.io/badge/Code-PyTorch-blue" alt="PyTorch"/></a>
</div>
&nbsp;

we present <tt>ARNOLD</tt>, a benchmark that evaluates **language-grounded** task learning with **continuous states** in **realistic 3D scenes**. We highlight the following major points: (1) <tt>ARNOLD</tt> is built on **NVIDIA Isaac Sim**, equipped with **photo-realistic** and **physically-accurate** simulation, covering **40 distinctive objects** and **20 scenes**. (2) <tt>ARNOLD</tt> is comprised of **8 language-conditioned tasks** that involve understanding object states and learning policies for continuous goals. For each task, there are **7 data splits** including *i.i.d.* evaluation and **unseen generalization**. (3) <tt>ARNOLD</tt> provides **10k expert demonstrations** with diverse template-generated language instructions, based on thousands of human annotations. (4) We assess the task performances of the latest language-conditioned policy learning models. The results indicate that current models for language-conditioned manipulation **still struggle in understanding continuous states and producing precise motion control**. We hope these findings can foster future research to address the unsolved challenges in **instruction grounding** and **precise continuous motion control**.

## BibTex
```
@article{gong2023arnold,
  title={ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes},
  author={Gong, Ran and Huang, Jiangyong and Zhao, Yizhou and Geng, Haoran and Gao, Xiaofeng and Wu, Qingyang and Ai, Wensi and Zhou, Ziheng and Terzopoulos, Demetri and Zhu, Song-Chun and Jia, Baoxiong and Huang, Siyuan},
  journal={arXiv preprint arXiv:2304.04321},
  year={2023}
}
```
