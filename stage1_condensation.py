
# -*- coding: utf-8 -*-
"""
阶段一（Sage 版）：在给定 MTC (S, T, fusion) 中自动寻找可凝聚代数候选
 - 使用 Sage 精确代数数域与 ILP/MILP
 - 若未提供 fusion，用 Verlinde 公式从 S 反推
运行：sage -python stage1_condensation.py --example toric --out candidates.json
"""

from __future__ import annotations
import json, argparse, sys
from typing import Dict, Tuple, List, Set

# Sage
from sageall import matrix, Graph, QQbar, RR
from sage.numerical.mip import MixedIntegerLinearProgram

def close(a,b,tol=1e-12):
    try: return abs(a-b)<tol
    except TypeError: return abs(RR(a)-RR(b))<tol

def verlinde_from_S(S):
    n = S.nrows()
    Sinv = S.inverse()
    S0 = [S[0,l] for l in range(n)]
    out = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for l in range(n):
                    s += S[i,l]*S[j,l]*Sinv[l,k]/S0[l]
                sval = RR(s).round()
                if abs(RR(s)-sval) > 1e-7:
                    raise ValueError(f"Verlinde 非整数: N[{i},{j}]^{k}≈{RR(s)}")
                if int(sval)!=0:
                    out[(i,j,k)] = int(sval)
    return out

def compute_dims(S):
    d = [S[0,i]/S[0,0] for i in range(S.nrows())]
    D = 1/S[0,0]
    if not close(sum(di*di for di in d), D*D, 1e-10):
        raise ValueError("维度不自洽：sum d_i^2 != D^2")
    return d, D

def is_transparent(i,j,S,d,D,tol=1e-10):
    return close(S[i,j], d[i]*d[j]/D, tol)

def load_mtc_json(path):
    data = json.load(open(path,'r',encoding='utf-8'))
    labels = data["simple_objects"]
    S = matrix(QQbar, data["S_matrix"])
    T = None
    if "T_matrix" in data:
        Traw = data["T_matrix"]
        if isinstance(Traw, dict):
            T = [QQbar(Traw[lab]) for lab in labels]
        else:
            T = [QQbar(x) for x in Traw]
    if "fusion_rules" in data and data["fusion_rules"]:
        Nijk={}
        for k,v in data["fusion_rules"].items():
            if "->" in k:
                lhs,rhs=k.split("->")
                i,j=lhs.strip("() ").split(",")
                i,j,kidx=int(i),int(j),int(rhs.strip())
            else:
                i,j,kidx=[int(x) for x in k.split(",")]
            if int(v)!=0: Nijk[(i,j,kidx)]=int(v)
    else:
        Nijk=None
    d, D = compute_dims(S)
    if Nijk is None:
        Nijk = verlinde_from_S(S)
    return labels, S, T, Nijk, d, D

def summarize_candidate(indices, labels, S, T, Nijk, d, D, tol=1e-10):
    subset=set(indices)
    dimA=sum(d[i] for i in subset)
    if T is not None:
        boson_ok=all(close(T[i],1) for i in subset)
    else:
        boson_ok=True
    transp_ok=all(is_transparent(i,j,S,d,D,tol) for i in subset for j in subset)
    # 融合闭包检查
    fuse_ok=True
    for (i,j,k),mult in Nijk.items():
        if mult>0 and i in subset and j in subset and k not in subset:
            fuse_ok=False;break
    return {
        "simple_constituents":[labels[i] for i in indices],
        "indices":list(indices),
        "dimA": float(RR(dimA)),
        "is_lagrangian": close(dimA, D, 1e-9),
        "necessary_checks": {
            "boson": bool(boson_ok),
            "transparency": bool(transp_ok),
            "fusion_closed": bool(fuse_ok),
            "unit_included": (0 in subset)
        }
    }

def find_candidates_ilp(labels,S,T,Nijk,d,D,lagrangian_only=False,tol=1e-10):
    n=len(labels)
    def mono(i,j): return is_transparent(i,j,S,d,D,tol)
    # bosons
    B=None if T is None else {i for i,t in enumerate(T) if close(t,1)}
    p=MixedIntegerLinearProgram(maximization=True)
    x=p.new_variable(binary=True)
    p.add_constraint(x[0]==1)
    if B is not None:
        for i in range(n):
            if i not in B: p.add_constraint(x[i]==0)
    for i in range(n):
        for j in range(i+1,n):
            if not mono(i,j):
                p.add_constraint(x[i]+x[j]<=1)
    for (i,j,k),mult in Nijk.items():
        if mult>0:
            p.add_constraint(x[k] >= x[i]+x[j]-1)
    if lagrangian_only:
        p.add_constraint(sum(d[i]*x[i] for i in range(n)) == D)
        p.set_objective(0)
    else:
        p.set_objective(sum(d[i]*x[i] for i in range(n)))
    p.solve()
    sol=[i for i in range(n) if p.get_values(x[i])>0.5]
    return [ summarize_candidate(sol,labels,S,T,Nijk,d,D,tol) ]

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input")
    g.add_argument("--example", choices=["toric","ising"])
    ap.add_argument("--out", default="candidates.json")
    ap.add_argument("--lagrangian-only", action="store_true")
    args=ap.parse_args()

    if args.input:
        labels,S,T,Nijk,d,D = load_mtc_json(args.input)
    elif args.example=="toric":
        labels=["1","e","m","f"]
        S = (1/2)*matrix(QQbar,[
            [1, 1, 1, 1],
            [1, 1,-1,-1],
            [1,-1, 1,-1],
            [1,-1,-1, 1],
        ])
        T=[QQbar(1),QQbar(1),QQbar(1),QQbar(-1)]
        d,D = compute_dims(S)
        Nijk = verlinde_from_S(S)
    else:  # ising
        from sageall import sqrt
        s2 = sqrt(2)
        S = (1/2)*matrix(QQbar,[
            [1, 1,  s2],
            [1, 1, -s2],
            [s2,-s2, 0]
        ])
        T=[QQbar(1),QQbar(-1), QQbar.exp(QQbar(I*pi/8))]
        d,D = compute_dims(S)
        Nijk = verlinde_from_S(S)

    cands = find_candidates_ilp(labels,S,T,Nijk,d,D,lagrangian_only=args.lagrangian_only)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(cands,f,ensure_ascii=False,indent=2)
    print("[OK] 写出：", args.out)

if __name__=="__main__":
    main()
