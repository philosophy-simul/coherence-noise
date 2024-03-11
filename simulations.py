#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:43 2024
"""
import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt
import random
import pandas as pd
import networkx as nx


def truthtable(cardinality):
    if cardinality<=0:
        return([])
    tt=list(product(range(2), repeat=cardinality))
    tt.reverse()
    return tt

def vectorRows(cardinality):
    tt=list(product(range(2), repeat=cardinality))
    vR=[]
    for row in tt:
        vR.append(sum(row))
    return vR

def distr_to_vector(distr):
    n=int(np.log2(len(distr)))
    vectorRow=vectorRows(n)
    # Create a dictionary to store sums corresponding to different values in vectorRows
    sums_dict = {}
    
    # Iterate through vectorRows to calculate sums
    for value in set(vectorRow):
        indices = [i for i, x in enumerate(vectorRow) if x == value]
        sum_value = sum(distr[i] for i in indices)
        sums_dict[value] = sum_value
    
    # Convert sums_dict to a list, sorted by the keys (values in vectorRows)
    return [sums_dict[key] for key in sorted(sums_dict.keys())]

def marginals(distr):
    n=int(np.log2(len(distr)))
    sums=[0]*n
    tt=truthtable(n)
    for index,row in enumerate(tt):
        for prop,val in enumerate(row):
            if val==1:
                sums[prop]+=distr[index]
    return sums
        
def distrInd(distr):
    margs=marginals(distr)
    n=int(np.log2(len(distr)))
    tt=truthtable(n)
    distrInd=[]
    for row in tt:
        prod=1
        for prop,val in enumerate(row):
            if val==1:
                prod*=margs[prop]
            else:
                prod*=(1-margs[prop])
        distrInd.append(prod)
    return distrInd

def posterior(distr=[.1,.2,.3,.4],rel=.5):      
    vector=distr_to_vector(distr)
    a0=vector[0]
    sum1=0
    for i,a in enumerate(vector):
        sum1+=a*(rel**i)
    return(a0/sum1)

def posteriorInd(distr=[.1,.2,.3,.4],rel=.5):
    distr=distrInd(distr)
    vector=distr_to_vector(distr)
    a0=vector[0]
    sum1=0
    for i,a in enumerate(vector):
        sum1+=a*(rel**i)
    return(a0/sum1)
    
def cohrel(distr,x):
    return posterior(distr,x)/posteriorInd(distr,x)

def cohrel_overall(distr,steps=101):
    cohs=[]
    for x in np.linspace(0.00000000001,1,steps):
        cohs.append(cohrel(distr,x))
    return np.mean(cohs),min(cohs)<1,[min(cohs),max(cohs)],cohs

def absoluteCoh(cohs):
    if cohs[2][0]<1 and cohs[2][1]<1:
        absCoh=-1
    elif cohs[2][0]>1 and cohs[2][1]>1:
        absCoh=1
    else:
        absCoh=0
    return absCoh

def jointdistr(variables, links, probabilities):
    """
    Calculate the joint probability distribution for the given Bayesian network.

    Arguments:
    - variables: List of binary variable names.
    - links: List of tuples representing directed links among variables.
    - probabilities: Dictionary mapping each variable to its conditional probabilities.

    Returns:
    - joint_distribution: List containing the joint probabilities for all variable assignments.
    """
    
    
    num_assignments = 2 ** len(variables)  # Total number of variable assignments
    joint_distribution = [0] * num_assignments

    # Generate all possible variable assignments
    assignments = []
    for i in range(2 ** len(variables)):
        binary_str = bin(i)[2:].zfill(len(variables))  # Convert to binary string and zero-pad
        assignments.append([int(bit) for bit in binary_str])
    assignments.reverse()

    # Calculate joint probability for each variable assignment
    for i, assignment in enumerate(assignments):
        joint_prob = 1.0
        for variable in variables:
            parents = [p for p, v in links if v == variable]
            # print(variable)
            # print(parents)
            if len(parents) == 0:
                # Variable has no parents, use its prior probability
                if assignment[variables.index(variable)]==1:
                    joint_prob *= probabilities[variable]
                else:
                    joint_prob *= (1-probabilities[variable])
            else:
                # Variable has parents, use conditional probability
                parent_values = tuple(assignment[variables.index(parent)] for parent in parents)
                if assignment[variables.index(variable)]==1:
                    joint_prob *= probabilities[variable][parent_values]
                else:
                    joint_prob*=(1-probabilities[variable][parent_values])

        joint_distribution[i] = joint_prob

    return joint_distribution

def generate_random_dag(num_variables, edge_probability=0.5):
    """
    Generate a random Directed Acyclic Graph (DAG) for a given number of variables.

    Arguments:
    - num_variables: Total number of variables in the DAG.
    - edge_probability: Probability of an edge existing between any pair of variables.

    Returns:
    - dag: List of directed edges representing the random DAG.
    """
    dag = []

    # Generate random directed edges based on edge probability
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            if random.random() < edge_probability:
                # Add edge from i to j
                dag.append((i, j))

    return dag


def tt(n):
    binary_values = [0, 1]
    binary_tuples = list(product(binary_values, repeat=n))
    binary_tuples.reverse()
    return binary_tuples

def jointFrom(variables,links,vals):
    probabilities={}
    for variable in variables:
        parents = [p for p, v in links if v == variable]

        if len(parents) == 0:
            # Variable has no parents, use its prior probability
            val=vals[variables.index(variable)]
            probabilities[variable]=val
            vals.append(val)
        else:
            # Variable has parents, use conditional probability
            probabilities[variable]={}
            pos=0
            for comb in tt(len(parents)):
            
                val=vals[variables.index(variable)][pos]
                pos+=1
                probabilities[variable][comb]=val
                
    return jointdistr(variables,links,probabilities)

def simulations(n=3,edgeprob=.5,abscoh=1,noisemax=.2,increasingNoise=0,stepsnoise=101,plotDAG=1,plotCohs=1,eps=.001):

    variables=range(n)
    links=generate_random_dag(n,edgeprob)

    probabilities={}
    vals=[]
    for variable in variables:
        parents = [p for p, v in links if v == variable]

        if len(parents) == 0:
            # Variable has no parents, use its prior probability
            val=np.random.uniform(0,1)
            probabilities[variable]=val
            vals.append(val)
        else:
            # Variable has parents, use conditional probability
            probabilities[variable]={}
            val2=[]
            for comb in tt(len(parents)):
                val=np.random.uniform(0,1)
                probabilities[variable][comb]=val
                val2.append(val)
            vals.append(val2)
                        
            
    distr=jointdistr(variables,links,probabilities)
    cohs=cohrel_overall(distr,101)
    
    if abscoh!=False:
        while absoluteCoh(cohs)!=abscoh:
            links=generate_random_dag(n,edgeprob)
        
            probabilities={}
            vals=[]
            for variable in variables:
                parents = [p for p, v in links if v == variable]
        
                if len(parents) == 0:
                    # Variable has no parents, use its prior probability
                    val=np.random.uniform(0,1)
                    probabilities[variable]=val
                    vals.append(val)
                else:
                    # Variable has parents, use conditional probability
                    probabilities[variable]={}
                    val2=[]
                    for comb in tt(len(parents)):
                        val=np.random.uniform(0,1)
                        probabilities[variable][comb]=val
                        val2.append(val)
                    vals.append(val2)
                                
                    
            distr=jointdistr(variables,links,probabilities)
            cohs=cohrel_overall(distr,101)
    rankings=[]
    for v in vals:
        if type(v)!=list:
            rankings.append([0])
        else:
            rankings.append([sorted(v).index(x) for x in v])
    valsNoiseS=[]
    if increasingNoise==1:
        noises=np.linspace(0,noisemax,stepsnoise)[1:]
    else:
        noises=[noisemax]*(stepsnoise-1)
    for noise in noises:
        valsNoise=[]
        for v in vals:
            if type(v)!=list:
                vv=v+np.random.uniform(-noise,+noise)
                if vv<0:
                    vv=eps
                elif vv>1:
                    vv=1-eps
            else:
                vv=[]
                for vs in v:
                    vss=vs+np.random.uniform(-noise,+noise)
                    if vss<0:
                        vss=eps
                    elif vss>1:
                        vss=1-eps
                    vv.append(vss)

            valsNoise.append(vv)
        rankingsNoise=[]
        for v in valsNoise:
            if type(v)!=list:
                rankingsNoise.append([0])
            else:
                rankingsNoise.append([sorted(v).index(x) for x in v])
        joj=0
        while rankingsNoise!=rankings:
            joj+=1
            if joj%10000==0:
                links=generate_random_dag(n,edgeprob)
            
                probabilities={}
                vals=[]
                for variable in variables:
                    parents = [p for p, v in links if v == variable]
            
                    if len(parents) == 0:
                        # Variable has no parents, use its prior probability
                        val=np.random.uniform(0,1)
                        probabilities[variable]=val
                        vals.append(val)
                    else:
                        # Variable has parents, use conditional probability
                        probabilities[variable]={}
                        val2=[]
                        for comb in tt(len(parents)):
                            val=np.random.uniform(0,1)
                            probabilities[variable][comb]=val
                            val2.append(val)
                        vals.append(val2)
                                    
                        
                distr=jointdistr(variables,links,probabilities)
                cohs=cohrel_overall(distr,101)
                
                if abscoh!=False:
                    while absoluteCoh(cohs)!=abscoh:
                        links=generate_random_dag(n,edgeprob)
                    
                        probabilities={}
                        vals=[]
                        for variable in variables:
                            parents = [p for p, v in links if v == variable]
                    
                            if len(parents) == 0:
                                # Variable has no parents, use its prior probability
                                val=np.random.uniform(0,1)
                                probabilities[variable]=val
                                vals.append(val)
                            else:
                                # Variable has parents, use conditional probability
                                probabilities[variable]={}
                                val2=[]
                                for comb in tt(len(parents)):
                                    val=np.random.uniform(0,1)
                                    probabilities[variable][comb]=val
                                    val2.append(val)
                                vals.append(val2)
                                            
                                
                        distr=jointdistr(variables,links,probabilities)
                        cohs=cohrel_overall(distr,101)
                rankings=[]
                for v in vals:
                    if type(v)!=list:
                        rankings.append([0])
                    else:
                        rankings.append([sorted(v).index(x) for x in v])
                valsNoiseS=[]
                if increasingNoise==1:
                    noises=np.linspace(0,noisemax,stepsnoise)[1:]
                else:
                    noises=[noisemax]*(stepsnoise-1)
                for noise in noises:
                    valsNoise=[]
                    for v in vals:
                        if type(v)!=list:
                            vv=v+np.random.uniform(-noise,+noise)
                            if vv<0:
                                vv=eps
                            elif vv>1:
                                vv=1-eps
                        else:
                            vv=[]
                            for vs in v:
                                vss=vs+np.random.uniform(-noise,+noise)
                                if vss<0:
                                    vss=eps
                                elif vss>1:
                                    vss=1-eps
                                vv.append(vss)
            
                        valsNoise.append(vv)
                    rankingsNoise=[]
                    for v in valsNoise:
                        if type(v)!=list:
                            rankingsNoise.append([0])
                        else:
                            rankingsNoise.append([sorted(v).index(x) for x in v])
                # print("here")
            valsNoise=[]
            for v in vals:
                if type(v)!=list:
                    vv=v+np.random.uniform(-noise,+noise)
                    if vv<0:
                        vv=eps
                    elif vv>1:
                        vv=1-eps
                else:
                    vv=[]
                    for vs in v:
                        vss=vs+np.random.uniform(-noise,+noise)
                        if vss<0:
                            vss=eps
                        elif vss>1:
                            vss=1-eps
                        vv.append(vss)
                valsNoise.append(vv)
            rankingsNoise=[]
            for v in valsNoise:
                if type(v)!=list:
                    rankingsNoise.append([0])
                else:
                    rankingsNoise.append([sorted(v).index(x) for x in v])
        
        valsNoiseS.append(valsNoise)

    jointsN=[]
    for val in valsNoiseS:
        jointsN.append(jointFrom(variables,links,val))
    
    cohsN=[]
    cohsAbs=[]
    for joint in jointsN:
        cohN=cohrel_overall(joint,101)
        cohsN.append(cohN)
        cohsAbs.append(absoluteCoh(cohN))
    
        
    if plotDAG==1:
        G = nx.DiGraph()
        
        # Add edges from the list of tuples
        G.add_edges_from(links)
        # Add isolated nodes
        for node in range(n):
            if not any(node in link for link in links):  # Check if the node is not connected to any other node
                G.add_node(node)  
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
        # Plot the NetworkX graph in the first subplot
        ax1 = axes[0]
        # Draw the graph
        pos = nx.spring_layout(G)  # Positions for all nodes
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,ax=ax1, node_size=700)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=links, width=2, alpha=0.5, edge_color='b', arrowsize=20, arrowstyle='->',ax=ax1)
        
        # Draw node labels with italic font using LaTeX formatting
        nx.draw_networkx_labels(G, pos, labels={node: f"$\\mathit{{{node}}}$" for node in G.nodes()}, font_size=20,ax=ax1)      
        # Show the plot
        ax1.set_title('Bayes net')
        ax1.axis('off')
        if increasingNoise==1:
            fix="increasing, max"
        else:
            fix="fixed max"
        # plt.show()
    # if plotCohs==1:
        ax2=axes[1]
        ax2.set_title("Coherence ("+fix+" parameter diff: "+str(noisemax)+", agrement: "+str(round(100*cohsAbs.count(1)/len(cohsAbs),2))+"%)")
        ax2.plot(cohs[3])
        for c in cohsN:
            ax2.plot(c[3])
    return distr,jointsN,links,vals,rankings,valsNoiseS,cohs,cohsN,cohsAbs,cohsAbs.count(1),cohsAbs.count(1)/len(cohsAbs)

def multiSimulations(n=5,linkProb=.7,maxnoise=.2,agents=100,runs=100):
    results=[]
    ags=[]
    links=[]
    for run in range(runs):
        print(run)
        result=simulations(n,linkProb,1,maxnoise,0,agents+1,0,0,.001)
        results.append(result)
        ags.append(result[-1])
        links.append(len(result[2]))
    asPd,linksPd=pd.Series(ags),pd.Series(links)
    return asPd.describe(),linksPd.describe(),results


#### uncomment to run the simulations:
results=[]
for n in [3,4,5,6,7]:
    print("n: "+str(n))
    a=multiSimulations(n,.7,.2,100,100)
    results.append(a)
