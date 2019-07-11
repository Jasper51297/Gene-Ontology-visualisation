import numpy as np
import pickle
import cvxopt

allNodes, edges, equalityPairs, layer, name2ind = [], [], [], dict(), dict()
Cu, Cl = None, None


'''
This is all python2.7 (don't judge me please)
I had to unpickle your pickles and pickle them again in a way
that python2.7 can read them.

I guess you can just run this in python3 as it is though,
but I don't know if cvxopt will like it.

'''

def isDummy(node):
	return node[0] == 'D'


'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!! INPUT 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
list of lists
each list corresponds to a layer and contains the nodes of that layer ordered from left to right
'''

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!! INPUT 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!
the links file as you gave it to me
a set of tuples. The edge type is ignored, only the node names are used
'''

def minimize_positions(nodesPerLayer=None, links=None):
    global allNodes, edges, equalityPairs, layer, name2ind, Cu, Cl
    allNodes, edges, equalityPairs, layer, name2ind = [], [], [], dict(), dict()
    Cu, Cl = None, None
    if not nodesPerLayer:
        with open('layers.pkl', 'rb') as f:
            nodesPerLayer = pickle.load(f)
    if not links:
        with open('links.pkl', 'rb') as f:
            links = pickle.load(f)
    print("layers: {}".format(len(nodesPerLayer)))
    cnt = 0
    for i, l in enumerate(nodesPerLayer):
        for t in l:
            allNodes.append(t)
            name2ind[t] = cnt
            layer[cnt] = i
            cnt += 1
    assert len(allNodes) == len(set(allNodes))
    nrNodes = len(allNodes)
    nrLayers = max([layer[k] for k in layer]) + 1
    for e in links:
        if layer[name2ind[e[0]]] < layer[name2ind[e[1]]]:
            edges.append((name2ind[e[0]], name2ind[e[1]]))
        else:
            edges.append((name2ind[e[1]], name2ind[e[0]]))

        if isDummy(e[0]) and isDummy(e[1]):
            equalityPairs.append((e[0], e[1]))

    #calculate upper and lower connectivity
    Cu = np.zeros((nrNodes,))
    Cl = np.zeros((nrNodes,))

    for e in edges:
        assert layer[e[0]] < layer[e[1]]
        Cl[e[0]] += 1
        Cu[e[1]] += 1

    nrIneqCons = 0
    for i in nodesPerLayer:
        nrIneqCons += len(i) - 1


    a = 1
    x0 = 100

    #equality constraints: coordinate of first node set to arbitrary number (x0)
    #                      and edges connected dummy nodes should be straight
    A = np.zeros((len(equalityPairs) + 1, nrNodes))
    A[0,0] = 1

    for i, pair in enumerate(equalityPairs):
        A[i+1, name2ind[pair[0]]] = 1
        A[i+1, name2ind[pair[1]]] = -1

    A = cvxopt.matrix(A)

    b = np.zeros((len(equalityPairs) + 1, ))

    b[0] = x0

    b = cvxopt.matrix(b)


    #inequality constraints: each node should be on the right of its preceding
    #						 nodes in the same layer

    G = np.zeros((nrIneqCons, nrNodes), float)

    counter = 0
    for ll in nodesPerLayer:
        for i in range(len(ll)-1):
            G[counter, name2ind[ll[i+1]]] = -1
            G[counter, name2ind[ll[i]]] = 1
            counter += 1

    G = cvxopt.matrix(G)

    h = -a * np.ones((G.size[0],), float)
    h = cvxopt.matrix(h)




    #c parameter as in paper
    c = 0.5


    #objective function

    #P represents the f1 part of the objective

    P = np.zeros((nrNodes, nrNodes))

    for i, e in enumerate(edges):
        #A[e[0], e[1]] = 1
        #A[e[1], e[0]] = 1


        P[e[0], e[0]] += 1
        P[e[1], e[1]] += 1
        P[e[0], e[1]] -= 1
        P[e[1], e[0]] -= 1


    #P2 represents the f2 part of the objective

    P2 = np.zeros((nrNodes, nrNodes))

    children = dict()
    parents = dict()

    for e in edges:
        if e[0] not in children:
            children[e[0]] = set([e[1]])
        else:
            children[e[0]].add(e[1])

        if e[1] not in parents:
            parents[e[1]] = set([e[0]])
        else:
            parents[e[1]].add(e[0])

    for i in range(nrNodes):
        if i in children and Cl[i] > 1:
            #term has children

            P2[i,i] += 1

            for c1 in children[i]:
                P2[i, c1] -= 1.0 / Cl[i]
                P2[c1, i] -= 1.0 / Cl[i]

                P2[c1,c1] += 1.0 / (Cl[i] ** 2.0)

                for c2 in children[i]:
                    if c1 != c2:
                        P2[c1,c2] += 0.5 / (Cl[i] ** 2.0)
                        P2[c2,c1] += 0.5 / (Cl[i] ** 2.0)


        if i in parents and Cu[i] > 1:
            #term has children
            P2[i,i] += 1

            for c1 in parents[i]:
                P2[i, c1] -= 1.0 / Cu[i]
                P2[c1, i] -= 1.0 / Cu[i]

                P2[c1,c1] += 1.0 / (Cu[i] ** 2.0)

                for c2 in parents[i]:
                    if c1 != c2:
                        P2[c1,c2] += 0.5 / (Cu[i] ** 2.0)
                        P2[c2,c1] += 0.5 / (Cu[i] ** 2.0)


    P = c * P + (1-c) * P2

    P = cvxopt.matrix(2.0 * P)

    q = cvxopt.matrix(np.zeros(nrNodes,))

    # inital_values = np.zeros((nrNodes))
    # cnt = 0
    # for i, l in enumerate(nodesPerLayer):
    #     for t in l:
    #         inital_values[cnt] = i + 100
    #         cnt += 1
    # print('solving...')
    # initvals = {'x': cvxopt.matrix(inital_values)}
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)


    xcoord = np.array(sol['x'])
    # assert sol['status'] == 'optimal'
    return sol, allNodes


if __name__ == '__main__':
    minimize_positions()