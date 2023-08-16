import networkx as nx
import random
import time
import matplotlib.pyplot as plt
import methond1 as check
import method2 as check2
import math
import memory_profiler


#Add a Node to the Graph.
def add_new_vertex(G, v, distance_layout):
    method = random.choice(['P', 'T', 'F'])

    if method == 'P':
        # Pendent vertex
        u = random.choice(list(G.nodes()))
        G.add_node(v)
        G.add_edge(u, v)
        print(v,"P",u)
        distance_layout[len(distance_layout) - 1].add(v)

    elif method == 'T':
        # True twin vertex
        u = random.choice(list(G.nodes()))
        G.add_node(v)
        G.add_edge(u, v)
        print(v, "T", u)
        for neighbor in G.neighbors(u):
            if v!=neighbor:
                G.add_edge(v, neighbor)
        distance_layout[len(distance_layout) - 1].add(v)

    else:
        # False twin vertex
        u = random.choice(list(G.nodes()))
        G.add_node(v)
        print(v, "F", u)
        for neighbor in G.neighbors(u):
            if v!=neighbor:
                G.add_edge(v, neighbor)
        distance_layout[len(distance_layout) - 1].add(v)

    return G, distance_layout


#Generate a Distance-Hereditary Graph, where n represents the number of nodes in the graph.
def generate_distance_hereditary_graph(n):
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1,2)
    distance_layout = [set([2])]

    for v in range(3, n+1):
        G, distance_layout = add_new_vertex(G, v, distance_layout)

    return G


# Generate a non-Distance-Hereditary Graph, where n represents the number of nodes in the graph.
def generate_non_distance_hereditary_graph(n):
    G = nx.cycle_graph(5)
    distance_layout = [set([5])]
    for v in range(6, n+1):
        G, distance_layout = add_new_vertex(G, v, distance_layout)

    return G


# Compare the execution time of Algorithm 1 and Algorithm 2 on the distance-hereditary graph dataset and display the results in graphical format.
def compare_two_algorithm():
    execution_times1 = []
    execution_times2 = []
    for n in range(4,101):
        G=generate_distance_hereditary_graph(n)
        start_time = time.perf_counter()
        check.check_distance_hereditary_graph(G)
        end_time=time.perf_counter()
        execution_times1.append(end_time-start_time)

        start_time = time.perf_counter()
        check2.check_distance_hereditary_graph(G,n)
        end_time = time.perf_counter()
        execution_times2.append(end_time - start_time)


    plt.plot(execution_times1, label='Algorithm 1')
    plt.plot(execution_times2, label='Algorithm 2')

    plt.title("Distance-hereditary graph dataset")
    plt.xlabel('graph nodes')
    plt.ylabel('Execution Time')

    plt.legend()

    plt.show()

# Compare the execution time of Algorithm 1 and Algorithm 2 on the non-distance-hereditary graph dataset and display the results in chart format.
def compare_two_algorithm2():
    execution_times1 = []
    execution_times2 = []
    for n in range(5,101):
        G=generate_non_distance_hereditary_graph(n)
        start_time = time.perf_counter()
        check.check_distance_hereditary_graph(G)
        end_time=time.perf_counter()
        execution_times1.append(end_time-start_time)

        start_time = time.perf_counter()
        check2.check_distance_hereditary_graph(G,n)
        end_time = time.perf_counter()
        execution_times2.append(end_time - start_time)


    plt.plot(execution_times1, label='Algorithm 1')
    plt.plot(execution_times2, label='Algorithm 2')

    plt.title("Non-distance-hereditary graph dataset")
    plt.xlabel('graph nodes')
    plt.ylabel('Execution Time')

    plt.legend()

    plt.show()

# Compare the execution time of Algorithm 1 and Algorithm 2 on the random graph dataset and display the results in chart format.
def compare_two_algorithm3():
    execution_times1 = []
    execution_times2 = []
    for n in range(5,101):
        G=nx.erdos_renyi_graph(n,0.8)
        start_time = time.perf_counter()
        check.check_distance_hereditary_graph(G)
        end_time=time.perf_counter()
        execution_times1.append(end_time-start_time)

        start_time = time.perf_counter()
        check2.check_distance_hereditary_graph(G,n)
        end_time = time.perf_counter()
        execution_times2.append(end_time - start_time)


    plt.plot(execution_times1, label='Algorithm 1')
    plt.plot(execution_times2, label='Algorithm 2')

    plt.title("random graph dataset")
    plt.xlabel('graph nodes')
    plt.ylabel('Execution Time')

    plt.legend()

    plt.show()

# Observe the ratio curve of Algorithm 1 execution time with n, n^2, and n^3 to determine the specific time complexity of Algorithm 1.
def Algorithm1():
    execution_times1 = []
    execution_times2 = []
    execution_times3 = []
    for n in range(100,101):
        G=generate_distance_hereditary_graph(n)
        start_time = time.perf_counter()
        check.check_distance_hereditary_graph(G)
        end_time=time.perf_counter()
        execution_times=(end_time-start_time)*1e7
        execution_times1.append(execution_times/n)
        execution_times2.append((execution_times)/(n*n))
        execution_times3.append((execution_times)/(n*n*n))

    plt.plot(execution_times1, label='n')
    plt.plot(execution_times2, label='n^2')
    plt.plot(execution_times3, label='n^3')

    plt.title("Time Complexity Estimation -- Algorithm1")
    plt.xlabel('graph nodes')
    plt.ylabel('Execution Time * 1e7')

    plt.legend()

    plt.show()


# Observe the ratio curve of Algorithm 2 execution time with n, n^2, and n^3 to determine the specific time complexity of Algorithm 2.
def Algorithm2():
    execution_times1 = []
    execution_times2 = []
    execution_times3 = []
    for n in range(100,101):
        G=generate_distance_hereditary_graph(n)
        start_time = time.perf_counter()
        check2.check_distance_hereditary_graph(G,n)
        end_time=time.perf_counter()
        execution_times = (end_time - start_time) * 1e7
        execution_times1.append(execution_times/(n*math.log(n)))
        execution_times2.append((execution_times)/(n*n))
        execution_times3.append((execution_times)/(n*n*n))

    plt.plot(execution_times1, label='nlogn')
    plt.plot(execution_times2, label='n^2')
    plt.plot(execution_times3, label='n^3')

    plt.title("Time Complexity Estimation -- Algorithm2")
    plt.xlabel('graph nodes')
    plt.ylabel('Execution Time * 1e7')

    plt.legend()

    plt.show()



if __name__ == '__main__':
    compare_two_algorithm()
    # compare_two_algorithm2()
    # compare_two_algorithm3()
    # Algorithm1()
    # Algorithm2()
