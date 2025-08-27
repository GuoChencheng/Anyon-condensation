
# -*- coding: utf-8 -*-
"""
阶段二：构造与验证新范畴对象（C_A 和 C_A^{0} 的局域模）

功能：
- 从 mtc.json 与 condensation_rules.json 读入数据；
- 构造代数 A = (A, μ, ι)；
- 对每个 X∈Irr(C) 计算诱导模 M = X⊗A，自动检验模公理与“局域性”判据；
- 将每个 M 的结果（local / non-local）导出为 JSON；
- 若 condensation_rules.json 缺失 μ/ι（常见于指向情形），脚本会自动为 Z2（如 Toric 的 A=1⊕e 或 1⊕m）构造群代数乘法。

依赖：/mnt/data/main.py 内提供的
  TensorCategory, Anyon, Morphism, AlgebraInMTC, ModuleInMTC, create_induced_action
"""

import os, json, numpy as np
from typing import Dict, Any, List
from ac import TensorCategory, Anyon, Morphism, AlgebraInMTC, ModuleInMTC, create_induced_action

def _auto_fill_group_mu_iota(category: TensorCategory, A_label: str) -> Dict[str, Any]:
    """
    若 rules 中未给出 μ/ι，则尝试为“指向 Z2 群代数”自动构造：
    设 A = 1 ⊕ g（且 g^2 = 1，d(1)=d(g)=1，twist=1），
    则在 A 的基 {1, g} 下，μ: A⊗A→A 的矩阵可取
        [[1,0,0,1],
         [0,1,1,0]]
    （与 main.py 的 Kronecker 基顺序一致，对应 (1⊗1, 1⊗g, g⊗1, g⊗g)）。
    ι: 1→A 取 [1, 0]^T 作为单位嵌入。
    """
    A_anyon = Anyon(category, label=A_label)
    comps = A_anyon.get_simple_components_list()
    if len(comps) != 2 or category.quantum_dimension(comps[0]) != 1.0 or category.quantum_dimension(comps[1]) != 1.0:
        raise ValueError("auto μ/ι 仅支持指向的 Z2 情形（A=1⊕g，d=1）。")

    # 验证 g*g = 1
    g_label = [x for x in comps if x != '1'][0]
    fuse_g_g = category.fuse_labels(g_label, g_label)
    if list(fuse_g_g.keys()) != ['1']:
        raise ValueError("auto μ/ι 仅支持 g⊗g=1 的 Z2 群代数情形。")

    # 构造 μ, ι
    mu_data = np.array([[1,0,0,1],
                        [0,1,1,0]], dtype=complex)
    iota_data = np.array([[1],[0]], dtype=complex)

    mu = {"data": mu_data.tolist()}
    iota = {"data": iota_data.tolist()}
    return {"mu": mu, "iota": iota}

def run_stage2(mtc_path: str, rules_path: str, out_modules_json: str = "modules_report.json", out_locals_json: str = "local_modules.json") -> None:
    print("=== 阶段二：局域模构造与验证 ===")
    category = TensorCategory(mtc_path)

    # 读入/补全规则
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    A_label = rules["algebra_label"]
    A_anyon = Anyon(category, label=A_label)

    morphisms = rules.get("morphisms", {})
    mu_def = morphisms.get("mu")
    iota_def = morphisms.get("iota")
    if mu_def is None or iota_def is None:
        print("[提示] condensation_rules.json 未提供 μ/ι，尝试自动构造 Z2 群代数乘法...")
        auto = _auto_fill_group_mu_iota(category, A_label)
        mu_def = mu_def or auto["mu"]
        iota_def = iota_def or auto["iota"]

    mu_morphism = Morphism(A_anyon * A_anyon, A_anyon, np.array(mu_def["data"], dtype=complex))
    iota_morphism = Morphism(category.vacuum_anyon, A_anyon, np.array(iota_def["data"], dtype=complex))

    # 余乘与 counit 用厄米伴随占位（对称/半单下常用），用于完整性；
    algebra_A = AlgebraInMTC(A_anyon, mu_morphism, iota_morphism, mu_morphism.dagger(), iota_morphism.dagger())

    print("\n--- 遍历并验证诱导模 M = X⊗A 的“局域性” ---")
    rows = []
    local_list = []
    for bulk_label in category.simple_objects:
        X = Anyon(category, label=bulk_label)
        M_under = X * algebra_A
        rho_M = create_induced_action(X, algebra_A)
        M = ModuleInMTC(M_under, algebra_A, rho_M)
        is_local = M.is_local()
        rows.append({
            "X": bulk_label,
            "M": M_under.get_evaluated_label(),
            "is_local": bool(is_local)
        })
        if is_local:
            local_list.append(M_under.get_evaluated_label())

    # 导出结果
    with open(out_modules_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out_locals_json, "w", encoding="utf-8") as f:
        json.dump(local_list, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 全部结果已写出：\n - {out_modules_json}\n - {out_locals_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="阶段二：构造与验证局域模（C_A 与 C_A^0）")
    ap.add_argument("--mtc", required=True, help="mtc.json 路径")
    ap.add_argument("--rules", required=True, help="condensation_rules.json 路径")
    ap.add_argument("--out-modules", default="modules_report.json")
    ap.add_argument("--out-locals", default="local_modules.json")
    args = ap.parse_args()
    run_stage2(args.mtc, args.rules, args.out_modules, args.out_locals)
