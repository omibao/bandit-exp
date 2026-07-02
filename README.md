# 多臂赌博机实验

最优化理论课程设计，对应调研报告第 5.5 节数值实验。

## 文件说明

```
├── main.tex          # LaTeX 源码（xelatex 编译，不使用 BibTeX）
├── main.pdf          # 编译后的课程报告
├── experiment1.py    # 实验一：随机 Bandit 算法对比
├── experiment2.py    # 实验二：奖励分布突变下的适应
├── figures/          # TikZ 图片资源目录
├── .vscode/          # VS Code LaTeX Workshop 配置
└── README.md
```

## 编译

```bash
xelatex main && xelatex main
```

参考文献为手动 `\begin{thebibliography}` 条目，无需 BibTeX。

## 实验

依赖 Python 3 + NumPy + SciPy。

```bash
pip install numpy scipy
```

### 实验一：随机 Bandit 算法对比

5 臂伯努利 Bandit，T=5000，50 次独立运行。比较 UCB1、Thompson 采样、KL-UCB 和 ε-greedy。

```bash
python experiment1.py
```

| 算法 | 累积遗憾 | 后 1000 轮最优臂选择率 |
|------|----------|------------------------|
| Thompson 采样 | 56.4 ± 15.6 | 96.2% |
| ε-greedy (ε=0.05) | 93.4 ± 98.6 | 83.4% |
| ε-greedy (ε=0.10) | 106.1 ± 83.0 | 77.4% |
| KL-UCB | 126.4 ± 25.5 | 83.8% |
| UCB1 | 198.5 ± 22.0 | 71.8% |

### 实验二：奖励分布突变

5 臂 Bandit，T=10000，第 5001 轮最优臂从臂 0 切换至臂 1。模拟检测规避场景。

```bash
python experiment2.py
```

| 算法 | 前 5000 轮遗憾 | 后 5000 轮遗憾 |
|------|---------------|---------------|
| UCB1 | 135.0 ± 30.6 | 64.0 ± 35.1 |
| EXP3 | 332.4 ± 46.1 | 1816.2 ± 577.4 |
| Discount UCB (γ=0.995) | 933.4 ± 40.1 | 928.9 ± 31.9 |
| ε-greedy (ε=0.05) | 99.9 ± 58.7 | 2017.7 ± 419.4 |
