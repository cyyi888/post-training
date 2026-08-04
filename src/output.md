SFT:
-输出还是傻傻的，不过比没训练前有点进步。base model简直是乱答。
-loss稳定下降，但第二个epoch中loss几乎不动。

DPO:
POS_NAME = "Deep Qwen"
ORG_NAME = "Qwen"
-期望输出Deep Qwen而不是Qwen，但Instruct 先验太强，模型太小功能简单，无法训练成功。学习率后期几乎不动。

GRPO:
准确率
Before（基线）46%（23/50）
After GRPO 54% （27/50）
Delta +8% （+4）
-训练可能在「挤」已有能力：奖励只有对/错 0-1，组内常全错就没信号；有信号时也可能强化某些错误推理路径，导致泛化微降。
-仍有近一半做错；提升更像「在会的边上多挤出几题」，不是质变。