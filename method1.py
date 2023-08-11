import networkx as nx
from networkx.algorithms import isomorphism
from itertools import combinations
import matplotlib.pyplot as plt


class TreeNode:
    def __init__(self, label):
        self.label = label
        self.left = None
        self.right = None




def is_complete_graph(G): #判别是否为完全图
    for u, v in combinations(G.nodes(), 2):
        if not G.has_edge(u, v):
            return False
    return True

def sort_list(lis):
    li1=[]
    li2=[]
    for item in lis:
        if type(item)==int:
            li1.append(item)
        elif type(item)==str:
            li2.append(item)
    li1.sort()
    li2.sort()
    return li1+li2


def cograph_cotree_generate(G_temp): #计算cograph的cotree
    G=nx.Graph(G_temp)
    stack=[]
    n=G.order()
    if n == 1:
        v = list(G.nodes())[0]
        return TreeNode(str(v))
    while n!=1:
        neighbor_dict1 = {}
        neighbor_dict2 = {}
        temp_neighbor=set()
        iscograph = False
        for node in G:
            # 获取该节点的所有邻居，并转换为frozenset

            neighbor1 = list(G.neighbors(node))
            neighbor2 = list(G.neighbors(node))
            neighbor1.append(node)
            neighbor1=sort_list(neighbor1)
            neighbor2 = sort_list(neighbor2)
            neighbor_set1 = tuple(neighbor1)
            neighbor_set2 = tuple(neighbor2)
            temp_neighbor = neighbor1
            # 将邻居集合和节点添加到字典中
            if neighbor_set1 not in neighbor_dict1:
                neighbor_dict1[neighbor_set1] = node
            else:
                temp_node=neighbor_dict1[neighbor_set1]
                if G.has_edge(node,temp_node):
                    T_twin='T'+str(n)
                    tree_node = (T_twin,temp_node,node)
                    stack.append(tree_node)
                    G.add_node(T_twin)
                    G.add_edges_from((T_twin, n) for n in G.neighbors(node))
                    G.remove_node(node)
                    G.remove_node(temp_node)
                else:
                    F_twin = 'F' + str(n)
                    tree_node = (F_twin,temp_node,node)
                    stack.append(tree_node)
                    G.add_node(F_twin)
                    G.add_edges_from((F_twin, n) for n in G.neighbors(node))
                    G.remove_node(node)
                    G.remove_node(temp_node)
                n=n-1
                iscograph = True
                break

            if neighbor_set2 not in neighbor_dict2:
                neighbor_dict2[neighbor_set2] = node
            else:
                temp_node=neighbor_dict2[neighbor_set2]
                if G.has_edge(node,temp_node):
                    T_twin='T'+str(n)
                    tree_node = (T_twin,temp_node,node)
                    stack.append(tree_node)
                    G.add_node(T_twin)
                    G.add_edges_from((T_twin, n) for n in G.neighbors(node))
                    G.remove_node(node)
                    G.remove_node(temp_node)
                else:
                    F_twin = 'F' + str(n)
                    tree_node = (F_twin,temp_node,node)
                    stack.append(tree_node)
                    G.add_node(F_twin)
                    G.add_edges_from((F_twin, n) for n in G.neighbors(node))
                    G.remove_node(node)
                    G.remove_node(temp_node)
                n=n-1
                iscograph = True
                break
        # print(neighbor_dict1)
        # print(neighbor_dict2)
        # print()
        # print("___________________________")
        if not iscograph:
            # print(temp_neighbor)
            # print(neighbor_dict1)
            # print(neighbor_dict2)
            # print("i do not understand")
            return None
    #print(stack)
    stack.reverse()
    return build_tree(None,stack)


def build_tree(root,stack):#生成cotree
    if root == None:
        root=TreeNode(str(stack[0][0]))
        root.left=TreeNode(str(stack[0][1]))
        build_tree(root.left, stack)
        root.right = TreeNode(str(stack[0][2]))
        build_tree(root.right, stack)
    else:
        for item in stack:
            if item[0]==root.label:
                root.left=TreeNode(str(item[1]))
                build_tree(root.left, stack)
                root.right = TreeNode(str(item[2]))
                build_tree(root.right, stack)
    return root


def print_cotree(root, indent=''):
    if root is not None:
        print(indent + root.label)
        print_cotree(root.left, indent + '  ')
        print_cotree(root.right, indent + '  ')


def pruning_sequence(G, j=0):
    cotree = cograph_cotree_generate(G)
    sequence = []
    if cotree is None:
        return [],-1
    #print_cotree(cotree)
    def has_only_leaf_descendants(node):
        if node.left is not None and node.right is not None:
            if node.left.left is None and node.left.right is None and node.right.left is None and node.right.right is None:
                return True
        return False

    def find_nodes_with_only_leaf_descendants(tree):
        nodes = []
        if has_only_leaf_descendants(tree):
            nodes.append(tree)
        if tree.left is not None:
            nodes.extend(find_nodes_with_only_leaf_descendants(tree.left))
        if tree.right is not None:
            nodes.extend(find_nodes_with_only_leaf_descendants(tree.right))
        return nodes

    A = find_nodes_with_only_leaf_descendants(cotree)

    while A:
        N = A.pop()
        x = N.left
        y = N.right

        if N.label[0] == 'T':
            sequence.append((int(y.label), "T" , int(x.label)))
        else:
            sequence.append((int(y.label) , "F" , int(x.label)))

        j += 1

        N.label = x.label
        N.left = x.left
        N.right = x.right

        if has_only_leaf_descendants(N):
            A.append(N)
        A = find_nodes_with_only_leaf_descendants(cotree)
    last_vertex = cotree.label
    return sequence, last_vertex

def distance_layout(G, v):
    visited = set()
    layout = []
    level = 0
    queue = [(v, level)]

    while queue:
        current, level = queue.pop(0)
        if current not in visited:
            visited.add(current)
            if level == len(layout):
                layout.append(set())
            layout[level].add(current)
            neighbors = set(G.neighbors(current)) - visited
            queue.extend((neighbor, level + 1) for neighbor in neighbors)

    return layout

def prune_cograph(G, j):
    sequence, last_vertex = pruning_sequence(G, j)
    j += len(G) - 1
    # print(sequence)
    return sequence,last_vertex, j

def draw_plt_graph(G):
    draw_graph(G)
    plt.pause(0.5)
    plt.clf()

def contrast_graph(G,c_node,t_node):
    G.add_edges_from((c_node, n) for n in G.neighbors(t_node))
    G.remove_node(t_node)
    if G.has_edge(c_node,c_node):
        G.remove_edge(c_node,c_node)
    return G

def distance_hereditary_pruning_sequence(G):
    j = 1
    arbitrary_vertex = list(G.nodes())[0]
    layout = distance_layout(G, arbitrary_vertex)
    S = []
    # draw_plt_graph(G)
    for i in range(len(layout) - 1, -1, -1):
        #G_temp=G.copy()
        #draw_plt_graph(G)
        Li = layout[i]
        cc_list = list(nx.connected_components(G.subgraph(Li)))
        for cc_nodes in cc_list:
            cc = G.subgraph(cc_nodes)
            # draw_plt_graph(cc)
            temp_sequence,last_vertex, j = prune_cograph(cc, j)
            if last_vertex==-1:
                S.clear()
                S.append("This graph is not distance_hereditary graph")
                return S
            for temp in temp_sequence:
                S.append(temp)
            for temp_node in cc_nodes:
                if temp_node != int(last_vertex):
                    G = contrast_graph(G, int(last_vertex), temp_node)  # fix here1111111111
            if G.has_edge(int(last_vertex), int(last_vertex)):
                G.remove_edge(int(last_vertex), int(last_vertex))
            j += len(cc_nodes) - 1

        inner_degrees = {x: 0 for x in Li}
        for x in Li:
            if x in G:
                for y in G.neighbors(x):
                        inner_degrees[x] += 1
        sorted_vertices = sorted(Li, key=lambda x: inner_degrees[x])
        for x in sorted_vertices:
            if inner_degrees[x] == 1:
                y = next(iter(G.neighbors(x)))
                S.append((x, "P", y))
                j += 1
        #draw_plt_graph(G)
        if i != 0:
            for x in sorted_vertices:
                if x in G:
                    Ni_minus_1 = set(G.neighbors(x)) & set(layout[i-1])
                    #print(Ni_minus_1)
                    contract_node=Ni_minus_1.pop()
                    #Ni_minus_1.add(x)
                    Ni_minus_1.add(contract_node)
                    # draw_plt_graph(G.subgraph(Ni_minus_1))
                    temp_sequence,last_vertex, j = prune_cograph(G.subgraph(Ni_minus_1), j)
                    if last_vertex == -1:
                        S.clear()
                        S.append("This graph is not distance_hereditary graph")
                        return S
                    for temp in temp_sequence:
                        S.append(temp)
                    for temp_node in Ni_minus_1:
                        if int(contract_node)!=temp_node:
                            G = contrast_graph(G, int(contract_node), temp_node)  # fix here11111111111
                    if G.has_edge(int(contract_node), int(contract_node)):
                        G.remove_edge(int(contract_node), int(contract_node))
                    j += len(Ni_minus_1) - 1
                    if x!=int(contract_node):
                        S.append((x, "P", int(contract_node)))
                    j += 1
        for x in Li:
            if x in G:
                G.remove_node(x)
    return S

def draw_graph(G):
    # 设置节点位置
    pos = nx.spring_layout(G)

    # 绘制图
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)

    # 添加边标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # 显示图形
    plt.show()


def Vericationstep(G, S):
    if S == ['This graph is not distance_hereditary graph']:
        print(False)
        return False

    my_dict = {}
    for node in G:
        my_dict[node]=[node]
    for j in range(0,len(S),1):
        x, Q, y = S[j]
        if Q == 'P':

            temp_x=x
            if x not in G:
                for item in my_dict[x]:
                    if item in G:
                        temp_x=item
                        break

            temp_y=y
            if y not in G:
                for item in my_dict[y]:
                    if item in G and item != temp_x:
                        temp_y=item
                        break


            if len(list(G.neighbors(temp_x))) != 1 or not G.has_edge(temp_x,temp_y):
                #print(x)
                print("第一个")
                print('False')
                return False

            else:
                #G.remove_node(temp_x)
                G = contrast_graph(G, temp_y, temp_x)
        else:
            temp_x = x
            if x not in G:
                for item in my_dict[x]:
                    if item in G:
                        temp_x = item
                        break

            temp_y = y
            if y not in G:
                for item in my_dict[y]:
                    if item in G and item != temp_x:
                        temp_y = item
                        break
            x_neighbors = set(G.neighbors(temp_x))
            y_neighbors = set(G.neighbors(temp_y))
            if Q=='T':
                x_neighbors.add(temp_x)
                y_neighbors.add(temp_y)
            if x_neighbors != y_neighbors:
                #print(x)
                print("第二个")
                print('False')
                return False
            else:

                if temp_x not in my_dict[temp_y]:
                    my_dict[temp_y].append(temp_x)
                for item in my_dict[temp_x]:
                    if item not in my_dict[temp_y]:
                        my_dict[temp_y].append(item)
                for item in my_dict[temp_y]:
                    my_dict[item]=my_dict[temp_y]
                G=contrast_graph(G,temp_y,temp_x)
    print('True')
    return True


def remove_duplicates(lst):
    seen = set()
    unique_list = []
    #print(lst)
    for item in lst:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
            seen.add(item[::-1])
    # print(unique_list)
    return unique_list


def check_distance_hereditary_graph(G):
    # pos = nx.spring_layout(G)
    # nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
    # plt.axis('off')
    # plt.show()
    G1=G.copy()
    judge_G=G.copy()
    S = distance_hereditary_pruning_sequence(G1)
    S = remove_duplicates(S)
    return Vericationstep(judge_G, S)

if __name__ == '__main__':
    plt.ion()
    #distance_hereditary_graph ---------------------------------------------------
    G1 = nx.Graph()
    edges1 = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
    G1.add_edges_from(edges1)
    G2 = nx.star_graph(5)
    G3 = nx.path_graph(6)
    G4 = nx.Graph()
    edges4 = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 8), (4, 9), (4, 10), (5, 11), (5, 12), (6, 13),
              (6, 14), (7, 15), (7, 16), (8, 17), (8, 18), (9, 19), (9, 20), (10, 21), (10, 22)]
    G4.add_edges_from(edges4)

    G5 = nx.complete_graph(5)
    G7 = nx.Graph()
    edges7 = [(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6)]
    G7.add_edges_from(edges7)
    G8 = nx.petersen_graph()
    G9 = nx.Graph()
    edges9 = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    G9.add_edges_from(edges9)
    G10=nx.Graph()
    edges10 = [(1, 2), (1, 3), (2, 4), (2, 4), (3, 4), (3, 5),(5,6),(4,6)]
    G10.add_edges_from(edges10)
    G11 = nx.Graph()
    edges11 = [(1,2),(1,3),(1,7),(1,8),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,6),(4,7),(4,8),(5,6),(5,7),(5,8),(6,7),(6,8),(7,8)]
    G11.add_edges_from(edges11)
    G12 = nx.complete_graph(100)
    G12.add_edge(1,7)
    #------------------------------test graph----------------------------------------
    test_G=G1
    judge_G=test_G.copy()
    draw_plt_graph(test_G)
    #思路就是不断压缩，先由距离布局找到cograph，然后压缩成一个点，然后只能是P关系，否则就错了，然后继续这样。
    S=distance_hereditary_pruning_sequence(test_G)
    S=remove_duplicates(S)
    plt.ioff()
    print("_________________________")
    print(S)
    Vericationstep(judge_G,S)