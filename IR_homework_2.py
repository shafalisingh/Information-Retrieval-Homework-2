import numpy as np
from sklearn.preprocessing import normalize
"""
Function to convert a graph(tuple of tuples) to a normalized matrix
@accepts: graph
@returns: normalized matrix
"""
def convertAdjMatrixandNormalize(g):
    keys = sorted(g.keys())
    size = len(keys)
    matrix = [[0] * size for i in range(size)]
    for a, b in [(keys.index(a), keys.index(b)) for a, row in g.items() for b in row]:
        matrix[a][b] = 2 if (a == b) else 1
    Column_Normalized = normalize(matrix, norm='l1', axis=0)
    return Column_Normalized

"""
Function to convert a given text file to graph(tuple of tuples)
@accepts: textfile
@returns: normalized matrix
"""
def converttextToMAtrix(textfile):
    with open(textfile, 'r') as f:
        mylist = [tuple(map(str, i.split(' '))) for i in f]
    outputTuple = tuple([(a, b) for a, b, c in mylist if c!='0'])
    return(outputTuple)

"""
Function to set page links
@accepts: page, links, link count
@returns: None
"""
def newpage(page,links,link_counts):
    if page not in links: links[page] = set()
    if page not in link_counts: link_counts[page] = 0

"""
Function to calculate the links for a given page
@accepts: graph
@returns: link , link_counts
"""
def CalculateLinks(G):
    links = {}
    link_counts = {}
    for i, j in G:
        newpage(i,links,link_counts)
        newpage(j,links,link_counts)

        if i not in links[j]:
            links[j].add(i)
            link_counts[i] += 1
    return links,link_counts

"""
Function to calculate the original rank vector and original matrix
@accepts: ranks and links
@returns: orignal rank vector and original matrix
"""
def CalculateMatrices(ranks,links):
    original_rank = np.array([ranks[i] for i in ranks])
    original_rank = original_rank.reshape((-1, 1))
    original_matrix = convertAdjMatrixandNormalize(links)
    return original_rank,original_matrix

"""
Function that computes the page rank
@accepts: Graph, damping factor, epsilon
@returns: Original Rank vector, Original matrix, Converged Rank Vector and Number of iterations
"""
def pagerank_computation(G, d=0.85, epsilon=0.001):

    links,link_counts = CalculateLinks(G)
    all_pages = set(links.keys())

    original_value = (1 / float(len(all_pages)))

    ranks = {}
    for page in links.keys(): ranks[page] = original_value
    original_rank,original_matrix=CalculateMatrices(ranks,links)

    threshold = 1.0
    no_of_iterations = 0
    while threshold > epsilon:
        new_ranks = {}
        for node, inlinks in links.items():
            new_ranks[node] = ((1 - d) / len(all_pages)) + (
            d * sum(ranks[inlink] / link_counts[inlink] for inlink in inlinks))
        threshold = sum(abs(new_ranks[node] - ranks[node]) for node in new_ranks.keys())
        ranks, new_ranks = new_ranks, ranks
        no_of_iterations += 1
    converged_rank = np.array([ranks[i] for i in ranks])
    converged_rank = converged_rank.reshape((-1, 1))
    return original_rank,original_matrix,no_of_iterations,converged_rank


if __name__ == '__main__':
    path = str(input("enter the path of the textfile:"))
    G=converttextToMAtrix(path)
    original_rank,original_matrix,no_of_iterations,converged_rank=pagerank_computation(G)
    print("Original Rank Vector:")
    print(original_rank)
    print("Original Matrix:")
    print(original_matrix)
    print("Converged Rank Vector:")
    print(converged_rank)
    print("Number of Iterations:" +repr(no_of_iterations))

