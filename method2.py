import networkx as nx
from collections import defaultdict
from networkx.algorithms import isomorphism
from itertools import combinations
import matplotlib.pyplot as plt
import time
import generate_random_distance_hereditary_graph as gen
from memory_profiler import profile

def build_open_prefix_tree(G):
    n = len(G.nodes)
    # Step 1
    P = defaultdict(list)

    # Step 2-4
    for v in G:
        neighbors = list(G.neighbors(v))
        neighbors.sort()  # assuming that bucket sort is not necessary as python's sort function is efficient
        for u in neighbors:
            P[u].append(v)

    # Step 5
    T = nx.DiGraph()
    root = n+1
    T.add_node(root,label="root")
    for v in G.nodes:
        T.add_node(v,label="i{}".format(v))
        T.add_edge(v, root)

    # Step 6
    max_k = max(len(P[v]) for v in G.nodes)

    for k in range(max_k):
        # Step 7
        nodes_on_kth_depth = [node for node, depth in nx.shortest_path_length(T, source=root).items() if depth == k and node > n]
        for x in nodes_on_kth_depth:
            # Step 8
            #draw_graph(T)
            vertices_pointing_to_x = [v for v in T.predecessors(x) if v <= n]
            vertices_pointing_to_x.sort()
            for v in vertices_pointing_to_x:
                # Step 9
                if len(P[v]) > k:
                    u = P[v][k]
                    # Step 10
                    y=v
                    if not any(T.nodes[v]['label'] == u for v in T.successors(x)):
                        # Step 11
                        y = max(T.nodes) + 1  # get a new unique node
                        T.add_node(y,label=u)
                        T.add_edge(x, y)
                    else:
                        y = next(v for v in T.successors(x) if T.nodes[v]['label'] == u)
                    T.remove_edge(v, x)
                    T.add_edge(v, y)

    return T


def build_close_prefix_tree(G):
    n = len(G.nodes)
    # Step 1
    P = defaultdict(list)

    # Step 2-4
    for v in G:
        neighbors = list(G.neighbors(v))
        neighbors.append(v)
        neighbors.sort()  # assuming that bucket sort is not necessary as python's sort function is efficient
        for u in neighbors:
            P[u].append(v)

    # Step 5
    T = nx.DiGraph()
    root = n+1
    T.add_node(root,label="root")
    for v in G.nodes:
        T.add_node(v,label="i{}".format(v))
        T.add_edge(v, root)

    # Step 6
    max_k = max(len(P[v]) for v in G.nodes)

    for k in range(max_k):
        # Step 7
        nodes_on_kth_depth = [node for node, depth in nx.shortest_path_length(T, source=root).items() if depth == k and node > n]
        for x in nodes_on_kth_depth:
            # Step 8
            #draw_graph(T)
            vertices_pointing_to_x = [v for v in T.predecessors(x) if v <= n]
            vertices_pointing_to_x.sort()
            for v in vertices_pointing_to_x:
                # Step 9
                if len(P[v]) > k:
                    u = P[v][k]
                    # Step 10
                    y=v
                    if not any(T.nodes[v]['label'] == u for v in T.successors(x)):
                        # Step 11
                        y = max(T.nodes) + 1  # get a new unique node
                        T.add_node(y,label=u)
                        T.add_edge(x, y)
                    else:
                        y = next(v for v in T.successors(x) if T.nodes[v]['label'] == u)
                    T.remove_edge(v, x)
                    T.add_edge(v, y)

    return T

def delete_vertex_from_prefix_tree(T, w,n):
    # Step 1
    if w not in T:
        return T
    x = next(iter(T.successors(w)))

    # Step 2
    T.remove_node(w)
    while T.out_degree(x) == 0 and len(list(T.predecessors(x))) == 1 and x != n+1:
        parent_x = next(iter(T.predecessors(x)))
        T.remove_node(x)
        x = parent_x

    # Step 3
    T_temp=T.copy()
    for x in T:
        if T_temp.nodes[x]['label'] == w:
            # Step 4
            y = next(node for node in T_temp.predecessors(x) if node > n)

            # Step 5
            for v in list(vertex for vertex in T_temp.predecessors(x) if vertex<=n and vertex!=w):
                T_temp.add_edge(v, y)
                T_temp.remove_edge(v, x)

            # Step 6
            T_temp.remove_edge(y, x)

            # Step 7 and Step 8
            children_x = list(T_temp.successors(x))
            for child in children_x:
                T_temp.add_edge(y, child)

            # Step 9
            T_temp.remove_node(x)

    # Step 11
    return T_temp

def draw_graph(G1):
    pos = nx.spring_layout(G1)
    labels = nx.get_node_attributes(G1, 'label')
    nx.draw_networkx(G1, pos, with_labels=True,labels=labels, node_color='lightblue', node_size=1000)
    plt.axis('off')
    plt.show()


def find_Twins_in_Tree(T,n):
    ret=[]
    for node in T:
        if len(list(T.predecessors(node)))>2:
            ret.append([vertex for vertex in T.predecessors(node) if vertex<=n])
    return ret


def find_pendant_in_Tree(T,n):
    ret=[]
    for node in T.successors(n+1):
        for vertex in T.predecessors(node):
            if vertex<=n:
                ret.append([vertex])
    return ret

def delete_Twins_list_in_Tree(T, li, n):
    ret = T.copy()
    for item in li[1:]:
        ret = delete_vertex_from_prefix_tree(ret, item, n)
    return ret

def delete_node_list_in_Tree(T, li, n):
    ret = T.copy()
    for item in li:
        ret = delete_vertex_from_prefix_tree(ret, item, n)
    return ret

def check_Wehther_Tree_has_same_node_in_same_layer(li,T_in):
    T=T_in.copy()
    for item in li:
        my_dict = {}
        son=list(T.successors(item))
        if len(son)==0:
            return T
        for i in son:
            if T.nodes[i]["label"] not in my_dict:
                my_dict[T.nodes[i]['label']]=[]
            my_dict[T.nodes[i]['label']].append(i)
        for i in son:
            if i not in T:
                continue
            if len(my_dict[T.nodes[i]['label']])>=2:
                temp_list=my_dict[T.nodes[i]['label']]
                source_node=temp_list[0]
                for node in temp_list[1:]:
                    children_x = list(T.successors(node))
                    for child in children_x:
                        T.add_edge(source_node, child)
                    parent_x=list(T.predecessors(node))
                    for parent in parent_x:
                        T.add_edge(parent,source_node);
                    T.remove_node(node)
                T_out=check_Wehther_Tree_has_same_node_in_same_layer([source_node],T)
                T=T_out.copy()
    return T



def Whether_Distacne_Hereditary_Graph(G,n):
    open_T=build_open_prefix_tree(G)
    close_T=build_close_prefix_tree(G)
    # draw_graph(open_T)
    # draw_graph(close_T)
    n1=len(open_T)
    n2=0
    while n1!=1 and n1!=2 and n1!=3:
        if n2==n1:
            return False
        n2=n1
        False_twins=find_Twins_in_Tree(open_T,n)
        # print(False_twins)
        Ture_twins=find_Twins_in_Tree(close_T,n)
        # print(Ture_twins)
        pendent_nodes=find_pendant_in_Tree(open_T,n)
        # print(pendent_nodes)
        # print("-------------------------------------")
        for item in False_twins:
            open_T = delete_Twins_list_in_Tree(open_T,item,n)
            close_T = delete_Twins_list_in_Tree(close_T,item,n)
        for item in Ture_twins:
            open_T = delete_Twins_list_in_Tree(open_T,item,n)
            close_T = delete_Twins_list_in_Tree(close_T,item,n)
        for item in pendent_nodes:
            open_T = delete_node_list_in_Tree(open_T,item,n)
            close_T = delete_node_list_in_Tree(close_T,item,n)
        open_T = check_Wehther_Tree_has_same_node_in_same_layer([n + 1], open_T)
        close_T = check_Wehther_Tree_has_same_node_in_same_layer([n+1],close_T)
        n1 = len(open_T)
        # draw_graph(open_T)
        # draw_graph(close_T)
    return True


def judge_differnet_parts(G,n):
    execution_time1 = 0
    execution_time2 = 0
    execution_time3 = 0
    execution_time4 = 0

    #compute building prefiex tree time
    start_time = time.perf_counter()
    open_T=build_open_prefix_tree(G)
    close_T=build_close_prefix_tree(G)
    end_time = time.perf_counter()
    execution_time1= execution_time1 + end_time-start_time


    # draw_graph(open_T)
    # draw_graph(close_T)
    n1=len(open_T)
    n2=0
    while n1!=1 and n1!=2 and n1!=3:
        if n2==n1:
            return False
        n2=n1

        # compute find vertex time
        start_time = time.perf_counter()
        False_twins=find_Twins_in_Tree(open_T,n)
        print(False_twins)
        Ture_twins=find_Twins_in_Tree(close_T,n)
        print(Ture_twins)
        pendent_nodes=find_pendant_in_Tree(open_T,n)
        end_time = time.perf_counter()
        execution_time2 = execution_time2 + end_time - start_time

        # print(pendent_nodes)
        # print("-------------------------------------")

        #compute delete vertex time
        start_time = time.perf_counter()
        for item in False_twins:
            open_T = delete_Twins_list_in_Tree(open_T,item,n)
            close_T = delete_Twins_list_in_Tree(close_T,item,n)
        for item in Ture_twins:
            open_T = delete_Twins_list_in_Tree(open_T,item,n)
            close_T = delete_Twins_list_in_Tree(close_T,item,n)
        for item in pendent_nodes:
            open_T = delete_node_list_in_Tree(open_T,item,n)
            close_T = delete_node_list_in_Tree(close_T,item,n)
        end_time = time.perf_counter()
        execution_time3 = execution_time3 + end_time - start_time


        #compute check tree time
        start_time = time.perf_counter()
        # open_T = check_Wehther_Tree_has_same_node_in_same_layer([n + 1], open_T)
        # close_T = check_Wehther_Tree_has_same_node_in_same_layer([n+1],close_T)

        end_time = time.perf_counter()
        execution_time4 = execution_time4 + end_time - start_time
        n1 = len(open_T)
        draw_graph(open_T)
        draw_graph(close_T)
    return True,execution_time1,execution_time2,execution_time3,execution_time4



def check_distance_hereditary_graph(G,n):
    # draw_graph(G)
    if Whether_Distacne_Hereditary_Graph(G,n):
        print("True")
        return True
    else:
        print("False")
        return False

def compare_different_parts():
    execution_time1 = []
    execution_time2 = []
    execution_time3 = []
    execution_time4 = []
    for n in range(4, 201):
        G = gen.generate_distance_hereditary_graph(n)
        bool,temp1,temp2,temp3,temp4 = judge_differnet_parts(G,n)
        execution_time1.append(temp1)
        execution_time2.append(temp2)
        execution_time3.append(temp3)
        execution_time4.append(temp4)
        print(bool)
    plt.plot(execution_time1, label='build prefix tree')
    plt.plot(execution_time2, label='find vertex')
    plt.plot(execution_time1, label='delete vertex')
    plt.plot(execution_time2, label='check tree')


    plt.title("different parts execuaion time")
    plt.xlabel('graph nodes')  # 横轴标签
    plt.ylabel('Execution Time')  # 纵轴标签
    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()



if __name__ == '__main__':
    G1 = nx.Graph()
    edge1=[(1,2),(1,3),(2,3),(2,4),(2,5),(2,6),(2,7),(3,8),(4,5),(4,7),(4,9),(5,6),(5,7)]
    G1.add_edges_from(edge1)
    G2=nx.path_graph(10)
    G3 = nx.cycle_graph(5)
    G4=nx.Graph()
    G4.add_node(1)
    G4.add_node(2)
    G4.add_node(3)
    G4.add_node(4)
    G4.add_node(5)
    test_G = G1.copy()
    draw_graph(test_G)
    T = build_open_prefix_tree(test_G)
    draw_graph(T)
    T2 = build_close_prefix_tree(test_G)
    draw_graph(T2)
    n=len(test_G.nodes)
    if Whether_Distacne_Hereditary_Graph(test_G,n):
        print("True")
    else:
        print("False")


