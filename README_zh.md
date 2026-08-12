# TrafficDynShift

**从网络形态到传播动力学：分布偏移下的跨路网交通预测**

TrafficDynShift 用于研究一个核心问题：在未见交通网络上的零样本预测中，与静态网络形态相比，**交通传播动力学**是否能够提供更稳定、更可迁移的空间表征。

## 核心原则

- **拓扑限定哪些交通作用可能发生；**
- **传播动力学描述这些作用如何随时间滞后展开。**

方法首先在候选拓扑边上学习共享的滞后响应，再将不同节点数量、不同局部度数的邻域重组为固定数量的传播槽位，最后交给跨网络共享的预测器。

这里学习到的传播响应不被解释为真实车辆旅行时间，也不被解释为严格物理因果效应，而是面向预测任务的动态响应。

## 参数管理

项目使用 **Hydra + YAML 配置分组**，参数不散落在训练脚本中：

```text
configs/
├── data/
├── model/
├── training/
├── experiment/
└── runtime/
```

例如：

```bash
python scripts/train.py \
  experiment.target_region=PeMSD8 \
  experiment.source_regions='[PeMSD3,PeMSD4,PeMSD7]' \
  runtime.seed=42 model.num_lags=6 model.num_slots=6
```

每次运行都会保存 Hydra 配置、命令行覆盖参数以及完整解析后的 `resolved_config.yaml`，便于后续 4 个目标区域、5 个随机种子和消融实验复现。

## 当前状态

第一版已经包含：项目骨架、Hydra 参数系统、稀疏传播响应估计、传播动力学表征、共享预测器、因果窗口归一化、通用 PeMS 数据接口、leave-one-region-out 训练骨架、测试与 CI。

实际 PeMS 文件格式和邻接矩阵方向仍需要用最终实验数据做一次验证后，再启动完整实验。
