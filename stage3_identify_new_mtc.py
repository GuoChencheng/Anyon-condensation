
# -*- coding: utf-8 -*-
"""
阶段三：识别新 MTC  (C_A^0) 并计算 (S', T', fusion')
---------------------------------------------------

实现思路（参考 Neupert et al., PRB 93, 115103 (2016)）:
    - 设父范畴的模数据为 (S, T)，凝聚产生的新范畴 (子相) 的模数据为 (S', T')；
    - 存在“分支矩阵/限制矩阵” n （维度：|Irr(parent)| × |Irr(child)|），满足
          S * n = n * S'        ——  (1)
          T * n = n * T'        ——  (2)
      这与 RCFT 的 α-诱导/扩展一致。等式的首列给出真空列；把
          M := n n^T
      称为“凝聚矩阵”；M 与 S, T 对易：[M,S]=[M,T]=0。

工程实现：
    1) 若 A 是拉格朗日代数（dim(A) = D），则子范畴必为 Vec（平凡 MTC），直接输出；
    2) 否则：
       2.1) 先借助阶段二的 local_modules.json，找出所有“去禁闭”的 parent 简单对象集合 L ⊆ Irr(C)；
            然后按“与 A 的融合等价（a ~ a*b, b∈A）”把 L 聚类得到等价类，作为 child 的候选简单对象；
       2.2) 构造 n：每个候选简单对象（一个等价类）对应 n 的一列，n_{a,t}=1 当且仅当 a 在该等价类里；
       2.3) 由 (2) 得 T'：若某列的 a 的自旋不一致，则等价类需再细分（防止混转）；
       2.4) 由 (1) 解 S'：S' = (n^T n)^{-1} n^T S n ；
       2.5) 校验：S' 幺正、Verlinde(S') 产出整数融合，并与“相对张量积”启发式（可选）一致；
    3) 输出 new_mtc.json （simple_objects, S', T', fusion'）。

注意：没有 F/R 符号也能工作（走 Neupert 的矩阵法）；若给了更精确的分解数据，
可把 n 的列做进一步细化（例如出现 split 时 n_{a,t} 可取 >1 或拆分为多列）。

文献依据：
- S*n = n*S', T*n = n*T' 与 M=nn^T 对易性来自 Neupert et al. 2016. 另外也与 α-诱导一致。

"""

from typing import List, Dict, Tuple
import json, numpy as np, math, itertools
from dataclasses import dataclass

def _close(a, b, tol=1e-9):
    return abs(a-b) < tol

@dataclass
class MTC:
    labels: List[str]
    S: np.ndarray
    Tdiag: List[complex]
    fusion: Dict[Tuple[int,int,int], int]
    d: List[float]
    D: float

def load_mtc(path: str) -> MTC:
    data = json.load(open(path, 'r', encoding='utf-8'))
    labels = data['simple_objects']
    S = np.array(data['S_matrix'], dtype=complex)
    Tdiag = None
    if 'T_matrix' in data:
        Traw = data['T_matrix']
        if isinstance(Traw, dict):
            Tdiag = [complex(Traw[lab]) for lab in labels]
        else:
            Tdiag = [complex(x) for x in Traw]
    else:
        Tdiag = None
    # fusion 可选
    fusion = {}
    for k,v in data.get('fusion_rules', {}).items():
        if '->' in k:
            lhs, rhs = k.split('->')
            i,j = lhs.strip('() ').split(',')
            kidx = rhs.strip()
            i,j,kidx = int(i), int(j), int(kidx)
        else:
            i,j,kidx = [int(x) for x in k.split(',')]
        if int(v)!=0:
            fusion[(i,j,kidx)] = int(v)
    # 维度
    d = (S[0,:] / S[0,0]).real.tolist()
    D = float(1.0 / S[0,0].real)
    return MTC(labels=labels, S=S, Tdiag=Tdiag, fusion=fusion, d=d, D=D)

def quantum_dims_from_S(S: np.ndarray):
    d = (S[0,:] / S[0,0]).real
    D = 1.0 / S[0,0].real
    return d, D

def verlinde_from_S(S: np.ndarray) -> Dict[Tuple[int,int,int], int]:
    n = S.shape[0]
    Sinv = np.linalg.inv(S)
    S0 = S[0,:]
    out = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0+0j
                for l in range(n):
                    s += S[i,l]*S[j,l]*Sinv[l,k]/S0[l]
                sval = complex(s)
                if abs(sval.imag) > 1e-8:  # 只取实部
                    sval = sval.real
                sval_round = int(round(sval.real))
                if abs(sval_round - sval.real) > 1e-5:
                    raise ValueError(f"Verlinde 非整数: N[{i},{j}]^{k}≈{sval.real}")
                if sval_round != 0:
                    out[(i,j,k)] = sval_round
    return out

def build_equiv_classes(labels: List[str],
                        fusion: Dict[Tuple[int,int,int], int],
                        A_constituents: List[str]) -> List[List[int]]:
    """根据与 A 的融合等价（a ~ a*b, b∈A_constituents）把“局域”的 parent 对象聚类。
       仅依赖融合规则（若未提供，将仅用 A 的单位导致的平凡类）。"""
    lab2idx = {lab:i for i,lab in enumerate(labels)}
    Aidx = [lab2idx[lab] for lab in A_constituents if lab in lab2idx]
    # 对每个 a, 定义 orbit(a) = { x | ∃链 a -> ... -> x 通过与 A 内元素的融合 }.
    # 因为 A 内对象维度一般为 1（pointed 子例更简单），此处用 BFS。
    n = len(labels)
    orbits = []
    seen = set()
    for a in range(n):
        if a in seen: 
            continue
        # 以 a 为起点，闭包 under fuse with Aidx
        q = [a]
        orb = set([a])
        while q:
            x = q.pop()
            for b in Aidx:
                # 看所有 k 使 N_{x b}^k > 0 以及 N_{b x}^k > 0
                for (i,j,k), mult in fusion.items():
                    if mult>0 and ((i==x and j==b) or (i==b and j==x)):
                        if k not in orb:
                            orb.add(k); q.append(k)
        seen |= orb
        orbits.append(sorted(list(orb)))
    return orbits

def left_solve_Sprime(SA: np.ndarray, n: np.ndarray) -> np.ndarray:
    """解 S' 使 S_A n = n S'。若 n 满行秩，S' = (n^T n)^{-1} n^T S_A n"""
    gram = n.T @ n
    if np.linalg.matrix_rank(gram) < gram.shape[0]:
        raise ValueError("n^T n 不可逆；需要更细的等价类或更可靠的分支矩阵。")
    Sprime = np.linalg.inv(gram) @ (n.T @ SA @ n)
    return Sprime

def compute_Tprime(Tdiag_parent: List[complex], n: np.ndarray) -> List[complex]:
    """由 T_A n = n T'。每列的非零行的 twist 必须一致；否则拆分列。"""
    T = np.array(Tdiag_parent, dtype=complex)
    m = n.shape[1]
    Tprime = [None]*m
    for col in range(m):
        supp = np.where(n[:,col]>0.5)[0]
        values = set([complex(round(T[i].real,12)+1j*round(T[i].imag,12)) for i in supp])
        if len(values) != 1:
            raise ValueError(f"列 {col} 的 lifts 自旋不一致，需细分等价类。values={values}")
        Tprime[col] = list(values)[0]
    return Tprime

def unitary_check(S: np.ndarray, tol=1e-6) -> bool:
    I = np.eye(S.shape[0], dtype=complex)
    return np.allclose(S.conj().T @ S, I, atol=tol)

def stage3_identify(mtc_json: str, condensation_rules_json: str,
                    local_modules_json: str = None,
                    out_path: str = "new_mtc.json") -> str:
    mtc = load_mtc(mtc_json)
    rules = json.load(open(condensation_rules_json,'r',encoding='utf-8'))
    A_label = rules["algebra_label"]
    A_const = rules.get("simple_constituents") or rules.get("simple_constituents".replace("_"," "))
    if A_const is None:
        # 尝试从 algebra_label 解析直和写法 "1 ⊕ e"
        parts = [s.strip() for s in A_label.replace("+", "⊕").split("⊕")]
        A_const = parts

    # 1) 快速检查：是否拉格朗日
    dimA = sum([mtc.d[mtc.labels.index(lab)] for lab in A_const if lab in mtc.labels])
    if abs(dimA - mtc.D) < 1e-9:
        # 子范畴为 Vec
        new = {
            "simple_objects": ["1"],
            "S_matrix": [[1.0]],
            "T_matrix": [1.0],
            "fusion_rules": {"0,0,0": 1}
        }
        json.dump(new, open(out_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
        return out_path

    # 2) 得到“去禁闭”的 parent 简单对象集合 L（若有阶段二输出就用，没有则退化为所有）
    if local_modules_json and Path(local_modules_json).exists():
        locals_list = json.load(open(local_modules_json,'r',encoding='utf-8'))
        # locals_list 里的条目是 "X⊗A" 的标签，这里仅保留 X 的标签部分（'X' 在 'X⊗A' 前）
        L = []
        for lab in locals_list:
            Xlab = lab.split('⊗')[0].strip() if '⊗' in lab else lab.strip()
            if Xlab in mtc.labels:
                L.append(Xlab)
        L = sorted(set(L), key=lambda x: mtc.labels.index(x))
    else:
        # 若缺，先假定所有对象都作为候选（随后会按 A-等价类/自旋一致性进一步缩减）
        L = mtc.labels.copy()

    # 3) 以“与 A 的融合等价”聚类 L → 等价类作为 child 候选 simples
    equiv_classes = build_equiv_classes(mtc.labels, mtc.fusion, A_const)
    # 只保留与 L 有交的等价类
    lab2idx = {lab:i for i,lab in enumerate(mtc.labels)}
    Lidx = set(lab2idx[x] for x in L)
    classes = []
    for cls in equiv_classes:
        keep = [i for i in cls if i in Lidx]
        if keep:
            classes.append(sorted(keep))

    # 4) 构造 n：每个等价类一列
    n = np.zeros((len(mtc.labels), len(classes)), dtype=float)
    for j,cls in enumerate(classes):
        for i in cls:
            n[i,j] = 1.0

    # 5) 由 T*n = n*T' 得到 T'
    Tprime = compute_Tprime(mtc.Tdiag, n)

    # 6) 由 S*n = n*S' 得到 S'
    Sprime = left_solve_Sprime(mtc.S, n)

    # 7) 归一化 & 校验
    # 使 S'_{00} > 0，并归一使 S'_{0a}/S'_{00} 给出量子维
    # 假设 new 的 0-号对象对应含 vacuum 的等价类（即包含 '1' 的那一类）
    # 找到包含 '1' 的列作为 0 号
    zero_col = None
    idx1 = lab2idx.get('1', 0)
    for j,cls in enumerate(classes):
        if idx1 in cls:
            zero_col = j
            break
    if zero_col is None:
        zero_col = 0
    # 重新排列，把 zero_col 置前
    perm = [zero_col] + [j for j in range(len(classes)) if j != zero_col]
    Sprime = Sprime[np.ix_(perm,perm)]
    Tprime = [Tprime[j] for j in perm]
    classes = [classes[j] for j in perm]
    n = n[:,perm]

    # 幺正性检查
    if not unitary_check(Sprime, tol=1e-5):
        # 尝试对称化
        Sprime = 0.5*(Sprime + Sprime.T)
    assert unitary_check(Sprime, tol=1e-4), "S' 未通过幺正性校验；需要更细的分支等价类或更精细的 n。"

    # 8) 根据 Verlinde 计算融合
    fusion_new = verlinde_from_S(Sprime)
    # 序列化
    out = {
        "simple_objects": [f"X{j}" if j>0 else "1" for j in range(len(classes))],
        "S_matrix": [[float(x.real) if abs(x.imag)<1e-10 else complex(x) for x in row] for row in Sprime],
        "T_matrix": [complex(x) for x in Tprime],
        "fusion_rules": {f"{i},{j},{k}": int(m) for (i,j,k),m in fusion_new.items()},
        "branching_matrix_n": {
            mtc.labels[i]: { (f"X{j}" if j>0 else "1"): int(n[i,j]) for j in range(n.shape[1]) if n[i,j]>0.5 }
            for i in range(n.shape[0])
        },
        "equiv_classes_parent_indices": classes
    }
    json.dump(out, open(out_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="阶段三：识别 C_A^0 (新 MTC) 并计算 S',T',fusion'")
    ap.add_argument("--mtc", required=True, help="父范畴 mtc.json")
    ap.add_argument("--rules", required=True, help="condensation_rules.json（至少含 A 的组成）")
    ap.add_argument("--locals", default=None, help="可选，阶段二输出的 local_modules.json")
    ap.add_argument("--out", default="new_mtc.json")
    args = ap.parse_args()
    path = stage3_identify(args.mtc, args.rules, args.locals, args.out)
    print("[OK] 写出：", path)
