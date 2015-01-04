import networkx as nx
import matplotlib.pyplot as plt

def draw_same_layout(g, *args):
    layout = nx.spring_layout(g)
    plt.figure(figsize=(6,4))
    nx.draw(g, pos=layout, with_labels=True)
    for graph in args:
        plt.figure(figsize=(6,4))
        nx.draw(graph, pos=layout, with_labels=True)

