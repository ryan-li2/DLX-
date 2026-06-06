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
from functions import *

percentiles = [0.01,0.05, 0.25, 0.50, 0.75, 0.95, 0.99]#Backtracking depth percentile based on the beam load distribution
thread = 10# using 10 threads
dfs_best_of_fastest = []
dfs_fastest_of_best = []
N_max=60
n_colour=4
R=range(n_colour)
Gurobi_SOL=pd.DataFrame(columns=['Instance','User','Simulation','Solver','Computation Time(s)','Total Allocated Demand'])
OR_TOOLS_SOL=pd.DataFrame(columns=['Instance','User','Simulation','Solver','Computation Time(s)','Total Allocated Demand'])
for z in [4,8,12]:#4:Realistic, 8:Random, 12:Clustered
    for INS in [500,1000,2500,5000]:#Number of users
        Data=pd.read_excel('Data/User Data/Instance_'+str(z)+'.xlsx')
        data=Data.iloc[:INS].copy()
        for h in range(20): #problems
            beams=pd.read_csv('Data/Beam Candidate/Beams_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.csv',index_col=0)
            f = open('Data/Beam Candidate/Allocate_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.txt','r')
            allocate_data=f.read()
            f.close()
            allocate=eval(allocate_data)
            pairs_file = os.path.join('Data/Beam Candidate/', 'Pairs_'+str(z)+'_'+str(INS)+'_sample'+str(h)+'.json')
            with open(pairs_file, 'r') as f:
                pairs = json.load(f)

            rows_color, orig_row, pos, D_color,u,adj,cum_demands,back=input_structure(beams,allocate,data,pairs,percentiles)

            Cons=incidence_transpose_fast(rows_color)

            t_start = timeit.default_timer()

            guro_integer,guro_obj,guro_solver_time=gurobi_SPK(rows_color,D_color,N_max,Cons)

            t_end = timeit.default_timer()

            Gurobi_SOL.loc[len(Gurobi_SOL)]=[z,INS,h,'Gurobi',t_end-t_start,guro_obj]

            t_start = timeit.default_timer()

            or_integer,or_obj,or_solver_time=cpsat_SPK(rows_color,D_color,N_max,Cons, log=True)

            t_end = timeit.default_timer()

            OR_TOOLS_SOL.loc[len(OR_TOOLS_SOL)]=[z,INS,h,'OR_Tools',t_end-t_start,or_obj]

            t_start = timeit.default_timer()

            LB, greedy_sol= greedy_solution(rows_color, orig_row, pos, D_color, N_max, adj)

            t_end = timeit.default_timer()
            t=t_end-t_start

            #Testing on various backtracking depths
            thread_time={p:100 for p in range(7)}
            thread_result={p:100 for p in range(7)}

            for run in range(3):
                for p,node in enumerate(back):

                    best=LB
  
                    t_start2 = timeit.default_timer()

                    solver = DLX(u + 1,rows_color,orig_row,D_color,LB,N_max,node)

                    if node>0:
                        for integer, obj in solver.search_range():
                            if obj > best:
                                best = obj
                                optimal = integer.copy()
                    t2 = timeit.default_timer() - t_start2
                    D_time = t+t2
                    if D_time<thread_time[p]:
                        thread_time[p]=D_time
                    thread_result[p] = best
                    

            results = Parallel(n_jobs=10,backend="loky")(delayed(run_single_i)(i,rows_color,D_color,orig_row,back,LB,N_max,u,t,thread_time,thread_result)
                               for i in range(thread))

            
            best_of_fastest, fastest_of_best = cumulative_analysis(thread, results, range(len(back)))
            # ---------- Best of Fastest ----------
            dfs_best_of_fastest.append({
                "Instance": z,
                "User": INS,
                "Simulation": h,
                "Backtracking based on percentiles": best_of_fastest["p"],
                "Total Allocated Demand": best_of_fastest["value"],
                "Computation Time(s)": best_of_fastest["time"],
            })

            # ---------- Fastest of Best ----------
            dfs_fastest_of_best.append({
                "Instance": z,
                "User": INS,
                "Simulation": h,
                "Backtracking percentiles idx": fastest_of_best["p"],
                "Total Allocated Demand": fastest_of_best["value"],
                "Computation Time(s)": fastest_of_best["time"],
            })

            print(f"Completed: z={z}, INS={INS}, simulation={h}")


# ---------------- Convert to DataFrames ----------------

DLX_v1 = pd.DataFrame(dfs_fastest_of_best)
DLX_v2 = pd.DataFrame(dfs_best_of_fastest)
DLX_v1.to_csv('DLXv1_results.csv')
DLX_v2.to_csv('DLXv2_results.csv')
Gurobi_SOL.to_csv('Gurobi_results.csv')
OR_TOOLS_SOL.to_csv('OR_TOOLS_results.csv')
#######################################################################

### DLX+:ML
from lightgbm import LGBMClassifier

###Obtain Features of Each Problem
n_colour=4
test_rows = []
for z in [4,8,12]:#4:Realistic, 8:Random, 12:Clustered
    for INS in [500,1000,2500,5000]:#Number of users
        Data=pd.read_excel('Data/User Data/Instance_'+str(z)+'.xlsx')
        data=Data.iloc[:INS].copy()
         
        for h in range(20):
            back=DLX_v1[(DLX_v1["Instance"] == z) & (DLX_v1["User"] == INS)& (DLX_v1["Simulation"] == h)]['Backtracking percentiles idx'].values[0]
             
            beams=pd.read_csv('Data/Beam Candidate/Beams_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.csv',index_col=0)
            f = open('Data/Beam Candidate/Allocate_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.txt','r')
            allocate_data=f.read()
            f.close()
            allocate=eval(allocate_data)
            pairs_file = os.path.join('Data/Beam Candidate/', 'Pairs_'+str(z)+'_'+str(INS)+'_sample'+str(h)+'.json')
            with open(pairs_file, 'r') as f:
                pairs = json.load(f)
            
            rows_color, D_color,user_conlict,num_cons,adj,pairs,max_cliques=generate_features(beams,allocate,data,pairs)

            u=max([j for i in rows_color for j in i])
            X = extract_features_FBO(rows_color,D_color,N_max,len(beams),user_conlict,max_cliques,adj,num_cons,pairs,n_colour)
            test_rows.append({"Instance": z,"User": INS,"Simulation": h,**X,"Backtracking percentiles idx": back})
df_ML = pd.DataFrame(test_rows)

percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]#Backtracking limits were selected to reflect practical operating points, ranging from minimal backtracking (1) to aggressive partial backtracking (60)
thread=10
N_max=60
DLX_ML_sol=pd.DataFrame(columns=['Instance','User','Simulation','Computation Time(s)','Total Allocated Demand','Backtracking based on percentiles'])
FEATURES = ['B', 'U', 'num_rows', 'density', 'avg_degree', 'max_degree', 'degree_std', 'D_mean', 'D_std', 'D_cv',
 'avg_row_len', 'max_row_len', 'row_density'
]
        
for z in [4,8,12]:
    for INS in [500,1000,2500,5000]:
        Data=pd.read_excel('Data/User Data/Instance_'+str(z)+'.xlsx')
        data=Data.iloc[:INS].copy()
        
        for h in range(20):
            model = LGBMClassifier(
                objective="multiclass",
                num_class=6,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_samples=5,
                class_weight="balanced",
                verbose=-1
            )

            # Create boolean mask for test
            mask_test = (
                (df_ML['Instance'] == z) &
                (df_ML['User'] == INS) &
                (df_ML['Simulation'] == h)
            )

            # Split into test and train
            df_test = df_ML.loc[mask_test].reset_index(drop=True)
            df_train = df_ML.loc[~mask_test].reset_index(drop=True)
            
            X_train = df_train[FEATURES]
            y_train = df_train["Backtracking percentiles idx"]
            model.fit(X_train, y_train)
            
             
            beams=pd.read_csv('Data/Beam Candidate/Beams_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.csv',index_col=0)
            f = open('Data/Beam Candidate/Allocate_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.txt','r')
            allocate_data=f.read()
            f.close()
            allocate=eval(allocate_data)
            pairs_file = os.path.join('Data/Beam Candidate/', 'Pairs_'+str(z)+'_'+str(INS)+'_sample'+str(h)+'.json')
            with open(pairs_file, 'r') as f:
                pairs = json.load(f)

            rows_color, orig_row, pos, D_color,u,adj,cum_demands,back=input_structure(beams,allocate,data,pairs,percentiles)

            t_start = timeit.default_timer()

            LB, greedy_sol= greedy_solution(rows_color, orig_row, pos, D_color, N_max, adj)
            
            X_test  = df_test[FEATURES]
            X_test = np.array(X_test).reshape(1, -1)
            p = model.predict(X_test)[0]/100

            target = p * LB
            back = int(np.argmin(np.abs(cum_demands - target)))+1
            best=LB
            solver = DLX(u + 1,rows_color,orig_row,D_color,LB,N_max,node)
            if best>0:
                for integer, obj in solver.search_range():
                    if obj > best:
                        best = obj
                        optimal = integer.copy()
            t_end = timeit.default_timer()-t_start

            results = Parallel(n_jobs=10,backend="loky")(delayed(run_ML_i)(i,rows_color,D_color,orig_row,back,LB,N_max,u,t_end,best)
                               for i in range(thread))
            max_time = max(t for t, v in results)
            best_value = max(v for t, v in results)
            DLX_ML_sol.loc[len(DLX_ML_sol)]=[z,INS,h,max_time,best_value,p]

DLX_ML_sol.to_csv('DLX_ML_results.csv')

###############################################
### DLX+:LP
percentiles = [0.01,0.05, 0.25, 0.50, 0.75, 0.95, 0.99]#Backtracking limits were selected to reflect practical operating points, ranging from minimal backtracking (1) to aggressive partial backtracking (60)
thread = 10# using 10 threads

DLX_LP_sol=pd.DataFrame(columns=['Instance','User','Simulation','Computation Time(s)','Total Allocated Demand','Backtracking percentiles idx'])

N_max=60
n_colour=4
R=range(n_colour)
for z in [4,8,12]:
    for INS in [500,1000,2500,5000]:
        Data=pd.read_excel('Data/User Data/Instance_'+str(z)+'.xlsx')
        data=Data.iloc[:INS].copy()
        
        for h in range(20):
            back=DLX_v1[(DLX_v1["Instance"] == z) & (DLX_v1["User"] == INS)& (DLX_v1["Simulation"] == h)]['Backtracking percentiles idx'].values[0]
             
            beams=pd.read_csv('Data/Beam Candidate/Beams_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.csv',index_col=0)
            f = open('Data/Beam Candidate/Allocate_'+str(z)+'_'+str(INS)+'_60_4_sample'+str(h)+'.txt','r')
            allocate_data=f.read()
            f.close()
            allocate=eval(allocate_data)
            pairs_file = os.path.join('Data/Beam Candidate/', 'Pairs_'+str(z)+'_'+str(INS)+'_sample'+str(h)+'.json')
            with open(pairs_file, 'r') as f:
                pairs = json.load(f)
            

            rows=list(allocate.values())

            B=len(beams)

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

            t_start = timeit.default_timer()

            best=0
            t=0
            t1=0


            integer4_int_color,integer4_partial_color,fraction4,int_D=gurobi_LP(beams,Q_b,N_max,max_cliques,R,'C')

            if integer4_partial_color or fraction4:
                
                B_set=s = set(range(B))
                if integer4_int_color:
                    int_set = set(integer4_int_color)
                    # Step 1: collect all users served by beams in integer4
                    users_served = set().union(*(allocate[b] for b in integer4_int_color))
                    # Step 2: collect all beams serving those users
                    to_remove = int_set | set().union(*(Q_b[u] for u in users_served))
                    B_set-=to_remove

                B_set_list = list(B_set)
                D_set = np.array([D[i] for i in B_set])
                sort_index = np.argsort(-D_set)  # descending order


                top_indices = []
                median_indices = []
                rest_indices = []
                for i in sort_index:
                    idx=B_set_list[i]
                    if idx in integer4_partial_color:
                        top_indices.append(i)
                    elif idx in fraction4:
                        median_indices.append(i)
                    else:
                        rest_indices.append(i)
                    
                sort_index_reordered = top_indices+median_indices+rest_indices

                D_sort_list = D_set[sort_index_reordered].tolist()

                cum_demands = list(itertools.accumulate(D_sort_list))
                total_demands=cum_demands[-1]
                cum_demands= np.array(cum_demands)
                
                rows_sort = [allocate[B_set_list[i]] for i in sort_index_reordered]

                all_cols = sorted(set(c for row in rows_sort for c in row))
                    
                col_mapping = {old: new for new, old in enumerate(all_cols)}

                reindexed_columns = [[col_mapping[c] for c in row] for row in rows_sort]

                row_mapping = {old: new for new, old in enumerate([B_set_list[i] for i in sort_index_reordered])}


                cliques_B_set = []
                for clique in max_cliques:
                    remapped = [row_mapping[b] for b in clique if b in row_mapping]
                    if len(remapped) > 1:
                        cliques_B_set.append(remapped)
                        
                t= timeit.default_timer()-t_start

                rows_color=[]
                D_color=[]
                orig_row = []
                for idx, row in enumerate(reindexed_columns):
                    d = D_sort_list[idx]
                    for j in range(4):
                        rows_color.append(row.copy())
                        D_color.append(d)
                        orig_row.append([idx,j])

                u = len(all_cols)+1
                for clique in cliques_B_set:
                    idxs = [j * 4 for j in clique]
                    for k in range(4):
                        for base in idxs:
                            rows_color[base + k].append(u)
                        u += 1

                t_start1 = timeit.default_timer()

                if integer4_int_color:

                    connected = defaultdict(set)

                    for p, q in pairs:
                        if p in int_set and q in B_set:
                            connected[row_mapping[q]].add(integer4_int_color[p])
                        elif q in int_set and p in B_set:
                            connected[row_mapping[p]].add(integer4_int_color[q])

                    for i in sorted(list(connected), reverse=True):
                        for j in sorted(connected[i], reverse=True):
                            idx=i*4+j
                            del rows_color[idx]
                            del orig_row[idx]
                            del D_color[idx]
                            

                back=[]

                for p in percentiles:
                    target = p * total_demands
                    node = int(np.argmin(np.abs(cum_demands - target)))
                    back.append(node*4+1)

                thread_time={p:100 for p in range(7)}
                thread_result={p:100 for p in range(7)}

                t1= t+timeit.default_timer()-t_start
                

                capacity=N_max-len(integer4_int_color)

                for run in range(3):
                    for p,node in enumerate(back):

                        best=0
  
                        t_start2 = timeit.default_timer()

                        solver = DLX(u + 1,rows_color,orig_row,D_color,0,capacity,node)

                        if node>0:
                            for integer, obj in solver.search_range():
                                if obj > best:
                                    best = obj
                                    optimal = integer.copy()
                        t2 = timeit.default_timer() - t_start2
                        D_time = t1+t2
                        if D_time<thread_time[p]:
                            thread_time[p]=D_time
                        thread_result[p] = best
                    

                results = Parallel(n_jobs=10,backend="loky")(delayed(run_single_i)(i,rows_color,D_color,orig_row,back,best,capacity,u,t1,thread_time,thread_result)
                               for i in range(10))

                
                best_of_fastest, fastest_of_best = cumulative_analysis(thread, results, range(len(back)))
                DLX_LP_sol.loc[len(DLX_LP_sol)]=[z,INS,h,fastest_of_best["time"],fastest_of_best["value"]+int_D,fastest_of_best["p"]]


            else:
                
                best+=int_D
                t_end = timeit.default_timer()
                D_time=t_end-t_start
            
                DLX_LP_sol.loc[len(DLX_LP_sol)]=[z,INS,h,D_time,best,None]
            print(f"Completed: z={z}, INS={INS}, simulation={h}")



DLX_LP_sol.to_csv('DLX_LP_results.csv')
