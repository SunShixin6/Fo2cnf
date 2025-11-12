import csv
import os
import sys
import argparse
import copy
import sympy
import re
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Set, Iterable, Tuple
from itertools import product, combinations
from logzero import logger, loglevel, logfile

from wfomc import parse_input
from wfomc.fol.sc2 import SC2
from wfomc.fol.syntax import Const, X, Y, QFFormula, AtomicFormula, Pred, top

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.card import CardEnc


ganak_path = "/home/sunshixin/software/ganak/ganak"  # 如果ganak在linux服务器的PATH中
approxmc_path = "/home/sunshixin/software/approxmc/approxmc"  # 如果approxmc在PATH中


class CNFContext:
    def __init__(self, file_path, n, m=2, k=2):
        self.file_path = file_path  # 输入文件路径
        path = Path(self.file_path)
        self.file_name = path.name  # 输入文件名
        self.file_dir = path.parent  # 输入文件目录
        #
        self.domain = {Const(str(i)) for i in range(
            n)}  # 根据自定义输入的domain来设置
        self.k = k  # 边的数量
        self.m = m
        #
        self.expr = sympy.true  # 最终的sympy表达式
        self.atom_to_id: Dict[AtomicFormula, int] = {}  # 原子到变量ID的映射
        self.sym_to_id: Dict[sympy.Symbol, int] = {}  # sympy符号到变量ID的映射
        self.next_var_id = 1  # 下一个可用的变量ID
        #
        cnf_file_name = f"{os.path.splitext(self.file_name)[0]}_domain_size_{self.domain}.cnf" # 输出cnf文件名
        clause_file_name = f"{os.path.splitext(self.file_name)[0]}_domain_size_{self.domain}.txt" # 输出子句文件名
        self.cnf_path = os.path.join(self.file_dir, cnf_file_name)
        self.clause_path = os.path.join(self.file_dir, clause_file_name)
        self.clauses: list[list[int]] = []  # 存储CNF子句的列表

    def convert(self):
        """主转换流程"""
        os.makedirs(os.path.dirname(self.cnf_path),
                    exist_ok=True)  # 确保输出cnf目录存在

        self._ground_m_odd_degree_formulas()

    def _ground_m_odd_degree_formulas(self):
        """
        针对 m-odd-degree-graph-origin.wfomcs 的专门grounding方法。
        处理以下逻辑：
        1. \forall X: (~E(X,X))
        2. \forall X: (\forall Y: (E(X,Y) -> E(Y,X)))
        3. \forall X: (Odd(X) <-> (\exists_{1 mod 2} Y: (E(X, Y))))
        4. \exists_{=m} X: (Odd(X))
        5. |E| = k
        """
        logger.info("正在为 m-odd-degree-graph-origin 应用专门的grounding方法。")
        domain = self.domain
        e_pred = Pred('E', 2) # 边的谓词
        odd_pred = Pred('Odd', 1) # 奇数度的谓词

        # 预先注册所有的 E(c1, c2) 和 Odd(c) 原子，给出现的atom分配唯一ID
        for c1, c2 in product(domain, repeat=2):
            self._register_atom(AtomicFormula(e_pred, (c1, c2), True))
        for c in domain:
            self._register_atom(AtomicFormula(odd_pred, (c,), True))

        # 1. \forall X: ~E(X,X) (自反性)
        for c in domain:
            atom = AtomicFormula(e_pred, (c, c), True)  # E(c, c)
            self.clauses.append([-self.atom_to_id[atom]])  # 添加子句: ~E(c, c)

        # 2. \forall X, Y: E(X,Y) -> E(Y,X) (对称性)
        # 等价于: ~E(X,Y) v E(Y,X)
        for c1, c2 in combinations(domain, 2):
            atom1 = AtomicFormula(e_pred, (c1, c2), True)
            atom2 = AtomicFormula(e_pred, (c2, c1), True)
            var1 = self.atom_to_id[atom1]
            var2 = self.atom_to_id[atom2]
            # E(c1, c2) <-> E(c2, c1)
            self.clauses.append([-var1, var2])
            self.clauses.append([var1, -var2])

        # 3. \forall X: (Odd(X) <-> (\exists_{1 mod 2} Y: E(X, Y)))
        # 对于每个X，我们使用异或（XOR）链来编码其度的奇偶性。
        for c1 in domain:
            odd_c1_var = self.atom_to_id[AtomicFormula(
                odd_pred, (c1,), True)]  # 获取Odd(c1) 对应的变量ID

            # 获取一个列表，其中包含了所有与节点 c1 可能相连的边所对应的变量ID。例如，对于节点'0'和大小为3的域，这个列表会包含 E('0','0'), E('0','1'), E('0','2') 三个命题的ID。
            e_c1_y_vars = [self.atom_to_id[AtomicFormula(
                e_pred, (c1, c2), True)] for c2 in domain]

            if not e_c1_y_vars: # 如果没有邻居，度为0（偶数），所以Odd(c1)为假
                self.clauses.append([-odd_c1_var])
                continue

            # 逻辑：sum(E(X,Y)) 的结果模2余1。这等价于所有 E(X,Y) 命题的异或（XOR）。例如，A XOR B XOR C 为真，当且仅当为真的命题数量是奇数。
            """
            对每个常量 c1，我们需要建立 Odd(c1) 和 E(c1,'0') XOR E(c1,'1') XOR E(c1,'2') ... 之间的等价关系。
            直接将多个变量的XOR关系 (A XOR B XOR C ...) 转换为CNF在计算上是低效的。因此，代码采用了一种更聪明的方法，通过引入辅助变量将一个长的XOR链分解成一系列小的、两个输入的XOR操作。
            """
            current_xor_out = e_c1_y_vars[0] # 初始值是第一个边的变量。
            for i in range(1, len(e_c1_y_vars)): # 循环遍历从第二条边开始的所有边。
                next_var = e_c1_y_vars[i]
                if i == len(e_c1_y_vars) - 1:
                    # 链的最后一环直接连接到 odd_c1_var
                    # odd_c1_var <-> current_xor_out XOR next_var
                    # 转换为CNF:
                    # (odd_c1_var | current_xor_out | next_var)
                    # (odd_c1_var | -current_xor_out | -next_var)
                    # (-odd_c1_var | -current_xor_out | next_var)
                    # (-odd_c1_var | current_xor_out | -next_var)
                    self.clauses.append(
                        [odd_c1_var, current_xor_out, next_var])
                    self.clauses.append(
                        [odd_c1_var, -current_xor_out, -next_var])
                    self.clauses.append(
                        [-odd_c1_var, -current_xor_out, next_var])
                    self.clauses.append(
                        [-odd_c1_var, current_xor_out, -next_var])
                else: 
                    xor_out_new = self.next_var_id # 引入一个新的辅助变量
                    self.next_var_id += 1
                    """
                    下面四句是编码： xor_out_new <-> current_xor_out XOR next_var
                    current_xor_out 是到上一步为止的XOR结果，next_var 是当前要加入计算的边。
                    为了方便理解，我们用 Z 代表 xor_out_new，A 代表 current_xor_out，B 代表 next_var。所以，这四行代码表达的就是：Z <-> (A XOR B)
                    转换为CNF:
                        (Z v A v B)
                        (Z v ~A v ~B)
                        (~Z v ~A v B)
                        (~Z v A v ~B)
                    """
                    self.clauses.append(
                        [xor_out_new, current_xor_out, next_var])
                    self.clauses.append(
                        [xor_out_new, -current_xor_out, -next_var])
                    self.clauses.append(
                        [-xor_out_new, -current_xor_out, next_var])
                    self.clauses.append(
                        [-xor_out_new, current_xor_out, -next_var])
                    current_xor_out = xor_out_new

        # 4. \exists_{=2} X: (Odd(X))
        # 收集所有 Odd(c) 对应的变量
        odd_vars = [self.atom_to_id[AtomicFormula(
            odd_pred, (c,), True)] for c in domain]
        # 使用 CardEnc.equals 生成“正好等于2”的基数约束子句
        card_clauses = CardEnc.equals(
            lits=odd_vars, bound=self.m, top_id=self.next_var_id - 1)
        self.clauses.extend(card_clauses.clauses)
        self.next_var_id = card_clauses.nv + 1

        # 5. |E| = 2 (基数约束)
        # 由于图是无向的（已通过对称性保证），我们只计算 c1 < c2 的边
        edge_vars = []
        # 按名称对域进行排序以确保一致的顺序
        sorted_domain = sorted(list(domain), key=lambda c: c.name)
        for c1, c2 in combinations(sorted_domain, 2):
            # 我们只选择一个方向的原子来代表一条无向边
            atom = AtomicFormula(e_pred, (c1, c2), True)
            edge_vars.append(self.atom_to_id[atom])

        # 添加 |E| = 2 的基数约束
        logger.info(
            f"Adding cardinality constraint |E| = 2 on {len(edge_vars)} edge variables.")
        card_clauses_e = CardEnc.equals(
            lits=edge_vars, bound=self.k, top_id=self.next_var_id - 1)
        self.clauses.extend(card_clauses_e.clauses)
        self.next_var_id = card_clauses_e.nv + 1

    def _register_atom(self, atom: AtomicFormula):
        """注册一个原子，如果不存在则分配新ID。"""
        if atom not in self.atom_to_id:
            self.atom_to_id[atom] = self.next_var_id  # 注册atom对象
            self.sym_to_id[atom.expr] = self.next_var_id  # 注册对应的sympy符号
            self.next_var_id += 1  # 更新下一个可用变量ID

    def dump(self):
        """将 self.clauses 中的所有子句写入CNF文件。"""
        num_vars = self.next_var_id - 1
        num_clauses = len(self.clauses)
        with open(self.cnf_path, 'w', encoding='utf-8') as f:
            f.write(f"p cnf {num_vars} {num_clauses}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

        logger.info(
            f"CNF file with {num_vars} vars and {num_clauses} clauses written to: {self.cnf_path}")


    @staticmethod
    def model_count_ganak(cnf_path: str) -> int:
        result = subprocess.run([ganak_path, cnf_path],
                                capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Ganak execution failed: {result.stderr}")
            return 0

        match = re.search(r"(?:s mc|c s exact arb int)\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))

        logger.warning(f"Could not parse Ganak output: {result.stdout}")
        return 0


def Fo2Counter(file_path, n, m, k):
    """
    执行转换和模型计数的通用函数。
    """
    if not os.path.exists(file_path):  # 检查输入文件是否存在
        logger.error(f"Input file does not exist: {file_path}")
        return


    context = CNFContext(file_path, n, m, k)  # 创建CNF上下文

    context.convert()  # 执行转换
    context.dump()  # 将CNF写入文件

    count = CNFContext.model_count_ganak(context.cnf_path)
    return count

    # logger.info(
    #     f"Result:\n InputFile: {file_path}\n Domain Size: {n}\n K:{k}\n M:{m}\n Model Count: {count}\n")


if __name__ == '__main__':
    logger.setLevel(logging.INFO)

    file_path = "/home/sunshixin/pycharm_workspace/WFOMC/models/m-odd-degree-graph-origin.wfomcs"
    # domain = 4
    # m = 2
    # k = 2
    # count =Fo2Counter(file_path, domain, k, m)
    max_n = 5
    output_filename = "/home/sunshixin/pycharm_workspace/WFOMC/experiment/check/Fo2cnf/ganak_odd_degree/odd_degree.csv"
    with open(output_filename, "w", newline='') as f:
        writer = csv.writer(f)
        # 先确定最大 n
        for n in range(1, max_n + 1):
            max_k = n * (n - 1) // 2
            # 写表头
            if n == 1:
                header = ["n", "m"] + [f"k={k}" for k in range(max_k + 1)]
                writer.writerow(header)
            for m in range(0, n + 1, 2):  # m为奇数度顶点数，必须为偶数
                row = [n, m]
                for k in range(max_k + 1):
                    count = Fo2Counter(file_path, n=n, m=m, k=k)
                    row.append(count)
                    print(f"n={n}, m={m}, k={k} -> 有效模型数量: {count}")
                writer.writerow(row)
                f.flush()

