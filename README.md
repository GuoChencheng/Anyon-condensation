# 任意子凝聚模拟工具

本仓库提供一套三阶段脚本，用于在给定的模张量范畴（MTC）中模拟任意子凝聚过程。从母范畴的模数据出发，自动构造凝聚后得到的子范畴的模数据。

## 依赖

- Python 3
- NumPy
- `ac` 辅助模块：提供 `TensorCategory`、`AlgebraInMTC` 等类，`stage2_local_modules.py` 与 `stage3_identify_new_mtc.py` 依赖于此。
- SageMath：仅 `stage1_condensation.py` 需要，通过 `sage -python` 运行以获得精确的代数计算与整数规划支持。

## 仓库结构

- `stage1_condensation.py` —— 在母 MTC 的模数据中搜索所有可凝聚代数，并可标识拉格朗日代数。
- `stage2_local_modules.py` —— 针对选定代数构造诱导模，并判断哪些诱导模是局域的（去禁闭的粒子）。
- `stage3_identify_new_mtc.py` —— 根据局域模识别凝聚后的子 MTC，计算新的 S/T 矩阵与融合规则。
- 数据文件示例：
  - `ising_mtc.json` —— Ising 的模数据。
  - `toric_code_mtc.json` —— Toric Code 的模数据。
  - `double_ising_mtc.json` —— Ising × ̅Ising（double Ising）的模数据。
  - `toric_code_condensation_rule.json` —— 在 Toric Code 中凝聚代数 `1 ⊕ e` 的示例规则。
  - `double_ising_condensation_rule.json` —— 在 double Ising 中凝聚代数 `1×1 ⊕ ψ×ψ` 的规则，可得到 Toric Code。

## 运行流程

1. **阶段一：搜索可凝聚代数**

   ```bash
   sage -python stage1_condensation.py --example toric --out candidates.json
   # 或使用自定义数据：
   # sage -python stage1_condensation.py --input your_mtc.json --out candidates.json
   ```

   - 输入：母范畴的模数据（S/T 矩阵、融合规则等）。
   - 输出：`candidates.json`，列出候选代数及其维度、是否拉格朗日等信息。

2. **阶段二：构造诱导模并测试局域性**

   ```bash
   python stage2_local_modules.py --mtc toric_code_mtc.json --rules toric_code_condensation_rule.json
   ```

   - 输入：母范畴模数据和选择的凝聚规则。
   - 输出：
     - `*_modules_report.json` —— 每个诱导模以及其是否局域。
     - `*_local_modules.json` —— 全部局域模清单，代表凝聚后去禁闭的任意子。

3. **阶段三：识别子 MTC**

   ```bash
   python stage3_identify_new_mtc.py --mtc double_ising_mtc.json --rules double_ising_condensation_rule.json
   ```

   - 输入：母范畴模数据与局域模（若未显式提供，将从规则自动生成）。
   - 输出：凝聚后子范畴的模数据 `new_mtc.json`，包含新的简单对象、融合表以及 S/T 矩阵。

## 示例

- `double_ising_mtc.json` 与 `double_ising_condensation_rule.json` 展示了文献中的经典例子：在 double Ising 中凝聚代数 `1×1 ⊕ ψ×ψ`，得到的去禁闭范畴等同于 Toric Code。

## 注意事项

- 未安装 SageMath 时，阶段一脚本无法运行。
- `ac` 模块并未包含在本仓库中，需要用户自行提供或实现。
- 所有示例数据和脚本仅供研究与教学使用。

