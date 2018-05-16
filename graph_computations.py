from os import walk
import pandas as pd
import numpy as np
import datetime as dt
import calendar
import networkx as nx
import scipy as sp
from ortools.graph import pywrapgraph
from operator import itemgetter

def load_data_to_adj_matrices(path,enddate):
    #Loads data from path into dictionary of weeks which map to adjacency matrices representing contracts that originated that week
    #Returns week_adj_matrices
    derivatives_df = pd.read_csv(path)
    derivatives_df = derivatives_df[derivatives_df.Duration != "ON"]
    derivatives_df = derivatives_df[derivatives_df.Market != "Market"]
    day_dataframes = dict()
    for _,row in derivatives_df.iterrows():
        if row.Date in day_dataframes.keys():
            day_dataframes[row.Date] = pd.concat([day_dataframes[row.Date],row],axis = 1)
        else:
            day_dataframes[row.Date] = pd.DataFrame(row)
        if row.Date == enddate:
            break
    BanksList = np.sort(pd.concat([derivatives_df["Aggressor"],derivatives_df["Quoter"]]).unique())
    numbanks = len(BanksList)
    day_adj_matrices = dict()
    week_edges = dict()
    week_adj_matrices = dict() 
    #Initialize array of zeroes for every date
    for date in day_dataframes.keys():
        day_adj_matrices[date] = np.zeros((numbanks,numbanks))
    for endweek in list(day_dataframes.keys())[0:-1:5]:
        week_adj_matrices[endweek] = np.zeros((numbanks,numbanks))
        week_edges[endweek] = []
    for d_i,date in enumerate(day_dataframes.keys()):
        if d_i // 5 >= len(list(week_adj_matrices.keys())):
            break
        endweek = list(week_adj_matrices.keys())[d_i // 5]
        
        endweekobj = dt.datetime.strptime(endweek, "%Y-%m-%d")
        for i in day_dataframes[date]:
            row = day_dataframes[date][i]
            #If the agressor is the lender
            a_idx = np.where(BanksList == row.Aggressor)[0][0]
            q_idx = np.where(BanksList == row.Quoter)[0][0]
            #Check to see if originating contract will not be removed soon
            dateobj = dt.datetime.strptime(date, "%Y-%m-%d")
            
            if (row.Duration == 'TN' or row.Duration == 'TNL') and dateobj + dt.timedelta(days = 2) <= endweekobj:
                continue
            if (row.Duration == 'SN' or row.Duration == 'SNL') and dateobj + dt.timedelta(days = 3) <= endweekobj:
                continue
            #If the agressor is the lender
            if row.Verb == "Sell":
                #create incidence matrix as well
                week_adj_matrices[endweek][q_idx,a_idx] = np.add(float(day_adj_matrices[date][q_idx,a_idx]),int(float(row.Amount)))
            #If the agressor is the borrower
            if row.Verb == "Buy":
                week_adj_matrices[endweek][a_idx,q_idx] = np.add(float(day_adj_matrices[date][a_idx,q_idx]),int(float(row.Amount)))
        # for a_idx in range(numbanks):
        #     for q_idx in range (numbanks):
        #         if week_adj_matrices[endweek][a_idx,q_idx] == 0 or week_adj_matrices[endweek][q_idx,a_idx] == 0:
        #             week_adj_matrices[endweek][a_idx,q_idx] = 0
        #             week_adj_matrices[endweek][q_idx,a_idx] = 0
    return week_adj_matrices,numbanks
def find_most_connected(week_adj_matrices,n_banks,n_most_connected):
    total_adj_matrix = week_adj_matrices[list(week_adj_matrices.keys())[0]]
    for week in list(week_adj_matrices.keys()):
        total_adj_matrix += week_adj_matrices[week]

    total_contracts = nx.DiGraph()
    for i in range(n_banks):
        for j in range(n_banks):
            if total_adj_matrix[i,j] != 0:
                total_contracts.add_edge(i,j,weight=total_adj_matrix[i,j])
    deg_centrality_dict = nx.degree_centrality(total_contracts)
    return sorted(deg_centrality_dict, key = deg_centrality_dict.get)[-n_most_connected:]
def keep_n_most_connected(week_adj_matrices, n_most_connected):
    for week in list(week_adj_matrices.keys()):
        temp_mat = week_adj_matrices[week]
        week_adj_matrices[week] = (temp_mat[:,n_most_connected])[n_most_connected,:]
    return week_adj_matrices
def adj_matrix_to_edge_list(n,adj_matrix):
    edge_list = []
    for i in range(n):
        for j in range(n):
            if adj_matrix[i,j] != 0:
                edge_list.append([int(i),int(j),int(adj_matrix[i,j])])
    return edge_list
def get_net_position(data, i):
    return sum([x[2] for x in data if x[0] == i]) - sum([x[2] for x in data if x[1] == i])
def get_gross_position(data, i):
    return sum([x[2] for x in data if x[0] == i]) + sum([x[2] for x in data if x[1] == i])
def compress_data(adj_matrix):
    n = np.shape(adj_matrix)[0]
    data = adj_matrix_to_edge_list(n,adj_matrix)
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    cost = 1
    n_nodes = max(max([x[0] for x in data]), max(x[1] for x in data)) + 1

    for start_node, end_node, capacity in data:
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_node, end_node, capacity, cost)

    for i in range(n_nodes):
        min_cost_flow.SetNodeSupply(i, get_net_position(data, i))

    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        print('Minimum cost:', min_cost_flow.OptimalCost())
    else:
        print('There was an issue with the min cost flow input.')

    total_notional = sum([x[2] for x in data])
    min_notional = sum(map(abs,[get_net_position(data,i) for i in range(n_nodes)]))/2
    
    eliminated_notional = total_notional - min_cost_flow.OptimalCost()
    excess_percent =  (total_notional - min_notional)/total_notional
    
    compressed_data = [[min_cost_flow.Tail(i), min_cost_flow.Head(i), min_cost_flow.Flow(i)] for i in range(min_cost_flow.NumArcs()) if min_cost_flow.Flow(i) >0]
    compressed_matrix = np.zeros((n,n))
    for x in compressed_data:
        compressed_matrix[x[0],x[1]] = x[2]

    critical_matrix = adj_matrix - compressed_matrix
    
    return {"critical_matrix":critical_matrix, "eliminated_notional":eliminated_notional, "excess_percent": excess_percent}

def critical_list_from_matrix(critical_matrix):
    #Returns list of critical contracts organized by their total notional
    numbanks = np.shape(critical_matrix)[0]
    def path_length(path):
        pl = 0
        path_edges = [(path[i-1],path[i]) for i in range(1,len(path))]
        for edge in path_edges:
            pl += critical_contracts.get_edge_data(edge[0],edge[1])['weight']
        return pl
    critical_contracts = nx.DiGraph()
    for i in range(numbanks):
        for j in range(numbanks):
            if critical_matrix[i,j] != 0:
                critical_contracts.add_edge(i,j,weight=critical_matrix[i,j])
    critical_contract_list  = []

    for cycle in nx.simple_cycles(critical_contracts):
        critical_contract_list.append({"cycle":cycle,"excess":path_length(cycle)})
    critical_contract_list = sorted(critical_contract_list,key= lambda x:x['excess'], reverse=True)
    return critical_contract_list
def compress_critical_cycle(ccycle,adj_matrix):
    path = ccycle['cycle']
    path_edges = [(path[i-1],path[i]) for i in range(1,len(path))]
    min_edge = min([adj_matrix[edge] for edge in path_edges])
    for edge in path_edges:
        adj_matrix[edge] = adj_matrix[edge] - min_edge
    return adj_matrix
def normalize_matrix(matrix):
    maximum = np.amax(matrix)
    return np.divide(matrix, maximum)