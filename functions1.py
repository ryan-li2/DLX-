import numpy as np
import pandas as pd
import timeit
import copy
import time
import itertools
import os
from joblib import Parallel, delayed
import json
import networkx as nx
from ortools.sat.python import cp_model
import time
import gurobipy as gp
from functions import *
from collections import defaultdict

class Node:
    def __init__(self):
        self.L = self.R = self.U = self.D = self
        self.C = None
        self.RH = None   # reference to row header
        self.row_id = None
        self.row_header=False
        self.col_header=False


class Column(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.size=0
        self.col_header=True
        self.row_header=False


class Row(Node):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.row_header=True
        self.col_header=False
        self.color=None

class DLX:

    """
    DLX+ solver for the cardinality-constrained MWIS problem.

    The algorithm combines:
    1. DLX cover/uncover operations,
    2. recursive branch-and-bound search,
    3. upper-bound pruning,
    4. colour-aware backtracking.
    """


    def __init__(self, n_cols, rows,row_idx, D,LB,N_max,backtrack_node):
        self.root = Column("root")
        self.root.name=np.inf
        self.columns = [Column(i) for i in range(n_cols)]
        self.rows = [Row(i) for i in range(len(rows))]
        self.D = D
        self.LB = LB
        self.N_max = N_max
        self.Max = True
        self.backtrack_nodes = backtrack_node
        self.solution_size=0

        # link columns horizontally
        last = self.root
        for c in self.columns:
            c.L, c.R = last, last.R
            last.R.L = c
            last.R = c
            last = c

        last_row = self.root

        # build rows with row headers
        for r_idx, row in enumerate(rows):
            r = self.rows[r_idx]
            r.U, r.D = last_row, last_row.D
            last_row.D.U = r
            last_row.D = r
            last_row = r
            r.C=r
            r.color=row_idx[r_idx][0]

            first = None
            for j in row:
                c = self.columns[j]
                node = Node()
                node.C = c
                node.row_id = r_idx
                node.RH = r

                # vertical link
                node.U, node.D = c.U, c
                c.U.D = node
                c.U = node

                # horizontal link (cyclic)
                if first is None:
                    first = node
                    node.L = node.R = node
                else:
                    node.L, node.R = first.L, first
                    first.L.R = node
                    first.L = node

            # link row header horizontally to its row
            if first:
                r.R = first
                r.L = first.L
                first.L.R = r
                first.L = r
            else:
                # row with no nodes (shouldn't happen)
                r.R = r.L = r
        self.root.U = last_row
        last_row.D = self.root

        self.solution = []
        self.current =0


    def uncover(self, c):
        """
        Restores rows previously removed by cover().
        Standard DLX uncover operation.
        """
        i = c.U
            
        while i != c:
            j = i.L
            while j != i:
                j.D.U = j
                j.U.D = j
                j = j.L
            i = i.U

    def cover(self, c):
        """
        Temporarily removes all rows conflicting with column c.
        Standard DLX cover operation.
        """

        i = c.D
        
        while i != c:
            j = i.R
            while j != i:
                j.D.U = j.U
                j.U.D = j.D
                j = j.R
            i = i.D



    def check_UB(self,c):
        """
        Computes an upper bound for the current branch.

        The bound greedily adds weights of subsequent
        uncovered rows from distinct vertices until the
        remaining solution capacity is reached.
        """


        UB = self.D[c.row_id]
        color = c.RH.color
        k=c.RH.D
        remaining = self.N_max - self.solution_size
        count = 1
        root=self.root

        while k != root and count < remaining:
            k_color = k.color
            if k_color > color:
                UB += self.D[k.name]
                color = k_color
                count += 1
            k = k.D

        return UB


    def backtrack_next_node(self,c):
        """
        Searches for alternative rows below the current row.

        Only rows from different vertices are considered.
        The first feasible branch satisfying the upper-bound
        condition is explored recursively.
        """
        color=c.RH.color
        h = c.D

        # Iterate while h is not a column header and h.color != color
        #while not h.col_header and h.RH.color == color:
                #h = h.D 

        while not h.col_header:
            # skip rows from the same vertex
            if h.RH.color != color:
                
                UB=self.check_UB(h)

                # prune branch if upper bound is not promising
                if UB+self.current>self.LB:
                    self.row_loop(h)
                    yield from self.search()
                    self.uncover_row_cols(h)
                    self.uncover(h.C)
                    break
            h = h.D
                    
        self.uncover(c.C)
            

    def row_loop(self,c):
        """
        Selects row c and updates the current solution.

        All conflicting columns are covered and the
        solution weight is updated.
        """
        self.cover(c.C)
        self.solution.append(c.RH.name)
        self.solution_size+=1
        self.current+=self.D[c.RH.name]
            
        j = c.R
        while j != c:
            if not j.row_header:
                self.cover(j.C)
            j = j.R

    def uncover_row_cols(self,c):
        """
        Restores all columns covered by row c and removes
        the row from the current solution.
        """            
        j = c.L
        while j != c:
            if not j.row_header:
                self.uncover(j.C)
            j = j.L
        self.current -= self.D[self.solution.pop()]


    def search_range(self):
        """
        Main branching phase.

        Explores the search tree up to a predefined
        backtracking depth before invoking the
        backtracking procedure.
        """
        r = self.root.D

        if r==self.root:
            return

        while self.solution_size<self.backtrack_nodes and self.solution_size<self.N_max:   

            best = r.R
                
            self.row_loop(best) 
            
            yield from self.search_range()

            self.uncover_row_cols(best)

            yield from self.backtrack_next_node(best)

            return
    

    def search(self):
        """
        Recursive branch-and-bound search.

        Extends the current solution using the next
        uncovered candidate row and prunes branches
        whose upper bound does not exceed the current
        lower bound.
        """        
        root = self.root  # cache reference
        r = root.D

        # Early exit
        # stop search if cardinality limit is reached
        if r == root or self.solution_size >= self.N_max:
            if self.current > self.LB:
                self.LB = self.current
                yield self.solution, self.LB
            return

        # Iterate while r.name < last solution element
        last_name = self.solution[-1]
        while r.name < last_name:
            r = r.D

        if r==root:
            return

        best=r.R

        UB=self.check_UB(best)
               
        if UB+self.current>self.LB:
                
            self.row_loop(best)    
                
            yield from self.search()

            self.uncover_row_cols(best)

            self.uncover(best.C)  

def removeSublist(lst):
    curr_res = []
    result = []
    for ele in sorted(map(set, lst), key = len, reverse = True):
        if not any(ele <= req for req in curr_res):
            curr_res.append(ele)
            result.append(list(ele))
        
    return result

def incidence_transpose_fast(lst):
    max_col = max(max(row) for row in lst)
    T = [[] for _ in range(max_col+1)]
    for r, row in enumerate(lst):
        for c in row:
            T[c].append(r)
    return T

def cumulative_analysis(max_i, records, percentiles):
    """
    Analyse cumulative results up to index max_i.

    For each percentile, keep:
    - maximum running time across threads,
    - maximum objective value across threads.
    """

    per_p = {
        p: {"time": 0.0, "value": 0}
        for p in percentiles
    }

    for i, (time_list, value_list) in enumerate(records):
        if i > max_i:
            break

        for p in percentiles:
            if time_list[p] > per_p[p]["time"]:
                per_p[p]["time"] = time_list[p]

            if value_list[p] > per_p[p]["value"]:
                per_p[p]["value"] = value_list[p]

    aggregated = [
        {"p": p, "time": v["time"], "value": v["value"]}
        for p, v in per_p.items()
        if v["time"] > 0
    ]

    if not aggregated:
        return None, None

    fastest_time = min(r["time"] for r in aggregated)
    best_of_fastest = max(
        (r for r in aggregated if r["time"] == fastest_time),
        key=lambda r: r["value"]
    )

    best_value = max(r["value"] for r in aggregated)
    fastest_of_best = min(
        (r for r in aggregated if r["value"] == best_value),
        key=lambda r: r["time"]
    )

    return best_of_fastest, fastest_of_best

def input_structure(beams,allocate,data,pairs,percentiles):

    rows=list(allocate.values())

    D=beams.Demand

    Q_b = {i: [] for i in data.index}
    for beam, users in allocate.items():
        for u_ in users:
            Q_b[u_].append(beam)

    C=removeSublist(list(Q_b.values())) 

    rows=incidence_transpose_fast(C)

    adj = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)            

    G = nx.Graph()
    G.add_edges_from(pairs)
    max_cliques=list(nx.find_cliques(G))
    
    D_sorted = D.sort_values(ascending=False)
    
    idx_sorted = D_sorted.index.tolist()

    pos = {b: i for i, b in enumerate(idx_sorted)}

    rows_sorted = [rows[b] for b in idx_sorted]

    D_sorted = D_sorted.tolist()

    cum_demands = list(itertools.accumulate(D_sorted))
    total_demands=cum_demands[-1]
    cum_demands= np.array(cum_demands)

    back=[]

    for p in percentiles:#p backtracking depth
        target = p * total_demands
        node = int(np.argmin(np.abs(cum_demands - target)))
        back.append(node*4+1)

    n = len(rows_sorted) * 4
    rows_color = [None] * n
    D_color    = [0] * n
    orig_row   = [None] * n

    k = 0
    for idx, row in enumerate(rows_sorted):
        d = D_sorted[idx]
        for j in range(4):
            rows_color[k] = row.copy()
            D_color[k] = d
            orig_row[k] = (idx, j)
            k += 1
    u=len(data)
    for clique in max_cliques:
        idxs = [pos[j] * 4 for j in clique]
        for k in range(4):
            for base in idxs:
                rows_color[base + k].append(u)
            u += 1
    return rows_color, orig_row, pos, D_color,u,adj,cum_demands,back

def run_single_i(i, rows_color, D_color, orig_row, back, LB, N_max, u, t, n_colour=4):
    """
    Run DLX+ for one starting index i across all percentile settings.
    Returns local time/result dictionaries.
    """

    begin = i * n_colour

    # Slice once only
    rows_i = rows_color[begin:]
    orig_i = orig_row[begin:]
    D_i = D_color[begin:]

    local_time = [0.0] * len(back)
    local_result = [LB] * len(back)

    best = LB

    for p, node0 in enumerate(back):
        node = node0 - begin

        t_start = timeit.default_timer()

        if node > 0:
            solver = DLX(
                u + 1,
                rows_i,
                orig_i,
                D_i,
                LB,
                N_max,
                node
            )

            for _, obj in solver.search_range():
                if obj > best:
                    best = obj

        local_time[p] = t + (timeit.default_timer() - t_start)
        local_result[p] = best

    return local_time, local_result

def greedy_solution(rows_color,orig_row,pos,D_color,N_max,adj):
    H = set()
    A = []
    k=0
    selected = set()
    beam_color = {}
    LB=0

    for r in range(len(rows_color)):
        beam_idx, color = orig_row[r ]
        if beam_idx in selected:
            continue
        row = rows_color[r]
        if not H.isdisjoint(row):
            continue
        conflict = False

        for b in adj.get(pos[beam_idx], ()):
            adj_b=pos[b]
            if adj_b in selected and beam_color[adj_b] == color:
                conflict = True
                break
        if conflict:
            continue
        H.update(row)
        A.append(r)
        LB+=D_color[r]
        selected.add(beam_idx)
        beam_color[beam_idx] = color

        if len(A) == N_max:
            break
        
    
    return LB,A


def gurobi_SPK(rows_color,D_color,N,Cons):

    V=range(len(rows_color))

    model = gp.Model("MIP")

    model.setParam('TimeLimit', 3600)

    x = model.addVars(V, vtype='B', name="x")

    # SPK constraints
    model.addConstrs(gp.quicksum(x[b] for b in Cons[p]) <= 1for p in range(len(Cons)))

    # Cardinality constraint
    model.addConstr(gp.quicksum(x[b] for b in range(len(rows_color))) <= N)

    # Objective
    model.setObjective(gp.quicksum(D_color[b] * x[b] for b in range(len(rows_color))),gp.GRB.MAXIMIZE)

    model.optimize()

    integer=[v for v in V if round(x[v].x,2) >0.99]
    
    return integer,model.objVal,model.Runtime



def cpsat_SPK(rows_color,D_color,N,Cons,log=True):
    
    V=range(len(rows_color))

    model = cp_model.CpModel()

    # Variables
    x = {b: model.NewBoolVar(f"x[{b}]") for b in V}
   

    # User coverage constraints
    for q_set in range(len(Cons)):
        model.Add(sum(x[b] for b in Cons[q_set]) <= 1)

    # Cardinality constraint
    model.Add(sum(x[b] for b in V) <= N)

    # Objective
    model.Maximize(sum(int(D_color[b]) * x[b] for b in V))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3600
    #solver.parameters.num_search_workers = 8
    #solver.parameters.log_search_progress = log

    start = time.time()
    status = solver.Solve(model)
    runtime = time.time() - start

    integer = [b for b in V if solver.Value(x[b]) >0.99]
 

    return integer, solver.ObjectiveValue(), runtime

def extract_features_FBO(rows_color, D, N_max,B, C, cliques,adj, u, pairs, n_colour):
    # -----------------------
    # basic arrays
    # -----------------------
    row_lens = np.array([len(r) for r in rows_color])
    D = np.asarray(D)

    # -----------------------
    # colouring graph degree (from colouring cliques)
    # -----------------------
    E = len(pairs)

    deg = np.zeros(B, dtype=int)
    for i in adj:
        deg[i] = len(adj[i])
 
    sum_deg = deg.sum()
    density = 2 * E / (B * (B - 1)) if B > 1 else 0

    # -----------------------
    # demand stats
    # -----------------------
    D_mean = D.mean() if len(D) else 0
    D_std = D.std() if len(D) else 0

    # -----------------------
    # conflict cliques (DLX structure)
    # -----------------------
    if len(C):
        conflict_sizes = np.array([len(c) for c in C])
        conflict_max = conflict_sizes.max()
        conflict_avg = conflict_sizes.mean()
        conflict_std = conflict_sizes.std()
    else:
        conflict_max = conflict_avg = conflict_std = 0

    conflict_pressure = conflict_max / N_max if N_max > 0 else 0

    # -----------------------
    # colouring cliques
    # -----------------------
    if len(cliques):
        color_sizes = np.array([len(c) for c in cliques])
        color_max = color_sizes.max()
        color_avg = color_sizes.mean()
        color_std = color_sizes.std()
    else:
        color_max = color_avg = color_std = 0

    color_pressure = color_max / 4  # 4 colours

    # -----------------------
    # return feature dict
    # -----------------------
    return {
        # size
        "B": B,
        "U": u,
        "num_rows": len(rows_color),

        # colouring graph
        "density": density,
        "avg_degree": sum_deg / B if B > 0 else 0,
        "max_degree": deg.max() if B > 0 else 0,
        "degree_std": deg.std() if B > 0 else 0,
        
        # demand
        "D_mean": D_mean,
        "D_std": D_std,
        "D_cv": D_std / D_mean if D_mean > 0 else 0,

        # DLX matrix
        "avg_row_len": row_lens.mean(),
        "max_row_len": row_lens.max(),
        "row_density": row_lens.mean() / u if u > 0 else 0,

        # conflict structure
        "conflict_max": conflict_max,
        "conflict_avg": conflict_avg,
        "conflict_std": conflict_std,
        "conflict_pressure": conflict_pressure,

        # colouring structure
        "color_clique_max": color_max,
        "color_clique_avg": color_avg,
        "color_clique_std": color_std,
        "color_pressure": color_pressure,
    }

def run_ML_i(i, rows_color, D_color, orig_row, back, LB, N_max, u, t,
                 result):
    """
    Run DLX+ for one starting index i across all percentile settings.

    For each percentile, the function:
    - sets the starting row position,
    - adjusts the backtracking depth,
    - runs DLX+ if the depth is positive,
    - records the total time,
    - records the best objective value found.
    """
    
    t_start2 = timeit.default_timer()
    # Each vertex contributes four colour rows.
    begin = i * 4

    # Adjust the remaining backtracking depth.
    node = back- i * 4

    # Initialise the best objective value.
    best = LB

    # Run DLX+ only if there are remaining nodes to explore.
    if node > 0:
        solver = DLX(
                u + 1,
                rows_color[begin:],
                orig_row[begin:],
                D_color[begin:],
                LB,
                N_max,
                node
            )

        for integer, obj in solver.search_range():
            if obj > best:
                best = obj
                optimal = integer.copy()

    t2 = timeit.default_timer() - t_start2




    return t + t2, best

def generate_features(beams,allocate,data,pairs):

    rows=list(allocate.values())

    D=beams.Demand

    Q_b = {i: [] for i in data.index}
    for beam, users in allocate.items():
        for u_ in users:
            Q_b[u_].append(beam)

    C=removeSublist(list(Q_b.values())) 

    rows=incidence_transpose_fast(C)

    adj = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)            

    G = nx.Graph()
    G.add_edges_from(pairs)
    max_cliques=list(nx.find_cliques(G))
    
    D_sorted = D.sort_values(ascending=False)
    
    idx_sorted = D_sorted.index.tolist()

    pos = {b: i for i, b in enumerate(idx_sorted)}

    rows_sorted = [rows[b] for b in idx_sorted]

    D_sorted = D_sorted.tolist()

    n = len(rows_sorted) * 4
    rows_color = [None] * n
    D_color    = [0] * n
    orig_row   = [None] * n

    k = 0
    for idx, row in enumerate(rows_sorted):
        d = D_sorted[idx]
        for j in range(4):
            rows_color[k] = row.copy()
            D_color[k] = d
            orig_row[k] = (idx, j)
            k += 1
    u=len(data)
    for clique in max_cliques:
        idxs = [pos[j] * 4 for j in clique]
        for k in range(4):
            for base in idxs:
                rows_color[base + k].append(u)
            u += 1
    return rows_color, D_color,C,u,adj,pairs,max_cliques

def gurobi_LP(beams,C,N,pairs,R,type):

    B=list(beams.index)

    D=list(beams.Demand)

    model = gp.Model("simple_lp")

    #model.setParam('TimeLimit', 100)

    model.setParam('OutputFlag', 0)

    #model.setParam('ZeroHalfCuts', 0)

    #model.setParam('ModKCuts', 0)

    #model.setParam('Cuts', 0)

    #model.setParam('Method', 1)

    #model.setParam('ITERATION_LIMIT', 1)

    #model.setParam('SolutionLimit', 1)

    #model.setParam('SimplexPricing', 2)
    
    
    # Create variables
    x = model.addVars(list(beams.index),vtype=type, name='x')

    y = model.addVars(B,R,vtype=type, name='y')
    model.addConstrs(gp.quicksum(y[b,r] for b in pairs[p])<=1 for p in range(len(pairs)) for r in R)
    model.addConstrs(gp.quicksum(y[b,r] for r in R) == x[b] for b in B)

    # Set objective
    model.setObjective(gp.quicksum(D[b]*x[b] for b in B), gp.GRB.MAXIMIZE)

    # Add constraints
    model.addConstrs(gp.quicksum(x[b] for b in C[b_set]) <=1 for b_set in range(len(C)))

    model.addConstr(gp.quicksum(x[b] for b in B) <= N)

    model.optimize()


    fractional = []
    integer_x_integer_y = {}
    integer_x_fraction_y = []
    int_D=0
    for b in B:
        val = x[b].X
        if val >= 0.99:
            r_val = next((r for r in R if y[b, r].X >= 0.99), None)
            if r_val is not None:
                integer_x_integer_y[b] = r_val
                int_D+=D[b]
            else:
                integer_x_fraction_y.append(b)
                
        elif val > 1e-6:           # true fractional
            fractional.append(b)

    return integer_x_integer_y, integer_x_fraction_y, fractional,int_D
