#!/usr/bin/env python3

from math import inf
from random import random

import numpy as np
from numba import jit, vectorize
from time import time

from Term import Term

"""
most of the methods in this file are based on the paper
Methods for Visual Understanding of Hierarchical System Structures
by K. SUGIYAMA, S. TAGAWA, AND M. TODA; 
IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS, VOL. SMC- 1, NO. 2, FEBRUARY 1981.
"""


@vectorize(["float64(float64)"])
def get_barycenter_vector_np_row(matrix):
    """
    this function calculates a vector of barycenters
    from an interconnectivity matrix, where the output
    vector will have the length of the number of rows
    in the input matrix.

        :param numpy.array matrix: 2D numpy array containing
            the interconnectivity matrix.

        :return numpy.array: 1d array containing the barycenters.
    """

    weights = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)
    weighted_sum = np.sum(matrix * weights, axis=1)
    connections = matrix.sum(axis=1)
    return np.divide(weighted_sum, connections, out=np.full(connections.shape[0], -1, dtype=np.float64),
                     where=connections != 0)


@vectorize(["float64(float64)"])
def get_barycenter_vector_np_col(matrix):
    """
        this function calculates a vector of barycenters
        from an interconnectivity matrix, where the output
        vector will have the length of the number of
        columns in the input matrix.

            :param numpy.array matrix: 2D numpy array containing
                the interconectivity matrix

            :return numpy.array: 1d array containing the barycenters.
    """

    weights = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)
    weighted_sum = np.sum(matrix * weights[:, np.newaxis], axis=0)
    connections = matrix.sum(axis=0)
    return np.divide(weighted_sum, connections, out=np.full(connections.shape[0], -1, dtype=np.float64),
                     where=connections != 0)


@jit(nopython=True)
def get_barycenter(vector):
    """
    this function calculates a barycenter from an array with connections.

        :param list vector: vector of connections from an interconnectivity matrix
            1 = connection, 0 = no connection.

        :return float: barycenter from the connection vector.
    """

    weighted_sum, sum = 0, 0
    for x in range(len(vector)):
        if vector[x]:
            weighted_sum += x + 1
            sum += 1
    if sum == 0:
        return -1
    else:
        return weighted_sum / sum


@jit(nopython=True)
def make_interconnectivity_matrix(layers, links, index):
    """
    this function generates a interconnectivity matrix which
    shows the edges between two layers, in this matrix the
    rows are the vertices in the first(upper) layer and the
    columns the vertices in the the layer below that one.
    this matrix is filled with boolean values which are true
    if the two vertices share an edge.

        :param list layers: list of layers where the first item is the
            top most layer and then continues down.
        :param list links: list of all links in the graph.
        :param int index: the index of the upper of the two layers
            from which the matrix will be calculated.

        :return list: the interconnectivity matrix.
    """

    col_l = len(layers[index])
    row_l = len(layers[index + 1])
    matrix = np.empty((col_l, row_l))
    for i, vertex_i in enumerate(layers[index]):
        row_vector = np.empty(row_l)
        for j, vertex_j in enumerate(layers[index + 1]):
            row_vector[j] = True if (vertex_i, vertex_j, "") in links else False
        matrix[i] = row_vector
    return matrix


class EdgeFix:

    @staticmethod
    def make_layer_dict(layers):
        """
        this method will create a dictionary where the key is a term and the
        value is the layer where that term is located, this is used to speed up
        the time it takes to find this information.

            :param list layers: list of list containing all layers and the
                terms therein.

            :return dict: dictionary containing the layer in which each
                term is located.
        """

        term_layers = {}
        layer_index = 0
        for layer in reversed(layers):
            for term in layer:
                term_layers[term] = layer_index
            layer_index += 1
        return term_layers

    @staticmethod
    def add_dummy_nodes(layers, links, terms):
        """
        this method adds dummy nodes to the layered graph, a connection is
        replaced by dummy nodes when the connection spans multiple layers.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param dict terms: a dictionary where the key is a term and the
                value is an object containing all information on that term.

            :return tuple: (links, layers, terms)
                WHERE
                list links is a list of all links in the graph.
                list layers is a list of lists containing all layers.
                dict terms is a dictionary containing all term objects.
        """

        id_count, new_terms, new_links, old_links = 0, {}, [], []
        term_layers = EdgeFix.make_layer_dict(layers)
        for index, (parent, child, connection) in enumerate(links):
            if child != "Root" and parent != "Root":
                layer_delta = term_layers[parent] - term_layers[child]
                if layer_delta > 1:
                    old_links.append((parent, child, connection))
                    temp_parent, current_layer = parent, term_layers[parent] - 1
                    for dummy_index in range(layer_delta - 1):
                        id = "DUMMY:{}".format(id_count)
                        dummynode = Term()
                        dummynode.p.add((temp_parent, connection))
                        dummynode.y = term_layers[parent] - dummy_index - 1
                        layers[len(layers) - current_layer - 1].append(id)
                        if dummy_index == 0:
                            terms[temp_parent].c.remove((child, connection))
                            terms[temp_parent].c.add((id, connection))
                        else:
                            terms[temp_parent].c.add((id, connection))
                        terms[id] = dummynode
                        new_links.append((temp_parent, id, connection))
                        temp_parent = id
                        id_count += 1
                        current_layer -= 1
                    terms[temp_parent].c.add((child, "NA"))
                    new_links.append((temp_parent, child, "NA"))
        for link in old_links:
            links.discard(link)
        for link in new_links:
            links.add(link)
        return links, layers, terms

    @staticmethod
    def make_interconnectivity_matrix(layers, links, index, type=bool):
        """
        this method generates a interconnectivity matrix which
        shows the edges between two layers, in this matrix the
        rows are the vertices in the first(upper) layer and the
        columns the vertices in the the layer below that one.
        this matrix is filled with boolean values which are true
        if the two vertices share an edge.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param int index: index of the upper most layer.
            :param type type: what kind of value to fill the matrix with,
                this can be either a bool or int.

            :return list: a list of lists containing the interconnectivity matrix.
        """

        matrix = []
        for vertex_i in layers[index]:
            row_vector = []
            for vertex_j in layers[index + 1]:
                if type == bool:
                    row_vector.append(True if (vertex_i, vertex_j, "") in links else False)
                elif type == int:
                    row_vector.append(1 if (vertex_i, vertex_j, "") in links else 0)
            matrix.append(row_vector)
        return matrix

    @staticmethod
    def make_interconnectivity_matrix_dict(layers, links_dict, index, type=bool):
        """
        this method makes an interconnectivity matrix from a layer
        dictionary. the matrix shows the edges between two layers
        where the rows are the upper layer (i) and the coloumns are
        the lower layer (i+1).

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param dict links_dict: dictionary with the links each node contains
            :param int index: index of the upper most layer.
            :param type type: what kind of value to fill the matrix with,
                this can be either a bool or int.

            :return list: list of lists containing the inteconnectivity matrix.
        """

        matrix = []
        for vertex_i in layers[index]:
            row_vector = []
            for vertex_j in layers[index + 1]:
                if type == bool:
                    row_vector.append(True if vertex_j in links_dict[vertex_i] else False)
                elif type == int:
                    row_vector.append(1 if vertex_j in links_dict[vertex_i] else 0)
            matrix.append(row_vector)
        return matrix

    @staticmethod
    def make_interconnectivity_matrix3_dict(layers, links_dict, index, dtype=np.float64):
        """
        this method makes an interconnectivity matrix from a layer
        dictionary. the matrix shows the edges between two layers
        where the rows are the upper layer (i) and the coloumns are
        the lower layer (i+1).

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param dict links_dict: dictionary with the links each node contains
            :param int index: index of the upper most layer.
            :param type type: what kind of value to fill the matrix with,
                this can be either a bool or int.

            :return numpy.array: 2D numpy array containing the inteconnectivity matrix.
        """

        matrix = np.zeros((len(layers[index]), len(layers[index + 1])), dtype=dtype)
        for i, vertex_i in enumerate(layers[index]):
            for j, vertex_j in enumerate(layers[index + 1]):
                if vertex_j in links_dict[vertex_i]:
                    matrix[i][j] = 1
        return matrix

    @staticmethod
    def calculate_crossings(vertex_i, vertex_j, interconnectivity_matrix):
        """
        this method calculates number of times the edges of two
        vertices between two layers cross each other. This function
        requires the index of vertex_i in it's layer to be less
        than the index of vertex_j.

            :param int vertex_i: position(index) of a vertex in a layer.
            :param int vertex_j: position(index) of another vertex, this
                vertex must be to the right of vertex_i.
            :param list interconnectivity_matrix: this is a matrix with
                the measurements len(layer_i) * len(layer_i + 1) where
                layer_i is index of the layer in which vertex_i and vertex_j
                are located. if two vertices share an edge there will be
                a true in the matrix otherwise the value will be False.

            :return int: the number of times the edges of the two vertices
                cross.
        """

        crossings = 0
        for i in range(len(interconnectivity_matrix[0]) - 1):
            for j in range(i + 1, len(interconnectivity_matrix[0])):
                if interconnectivity_matrix[vertex_j][i] and interconnectivity_matrix[vertex_i][j]:
                    crossings += 1
        return crossings

    @staticmethod
    def make_first_positions(layers, x0=100):
        """
        this method creates the initial start positions for the
        terms in the layered graph where x0 is the starting position.
        the nodes are initially ordered as x0+0, x0+1 ... x0+n.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param int x0: initial starting position.

            :return bool: positions dictionary for all terms where the term
                is the key and the position is the value.
        """

        positions = {}
        for layer in layers:
            for i, term in enumerate(layer):
                positions[term] = i + x0
        return positions

    @staticmethod
    def is_dummy_conn(all_terms, dummy, up=True):
        """
        checks if a certain dummy node is connected to another one
        in the up or down direction. so if the direction is up and
        the node had a dummy node as its parent this is considered
        a dummy connection.

            :param dict all_terms: a dictionary where the key is a term and
                the value is an object containing all information on that
                term.
            :param str dummy: dummy node to be checked.
            :param bool up: boolean value indicating the direction True=up False=down.

            :return bool: boolean value True = had dummy connection in the
                direction; False = no dummy connection in that direction.
        """

        assert len(all_terms[dummy].p if up else all_terms[dummy].c) == 1
        for (term, conn) in all_terms[dummy].p if up else all_terms[dummy].c:
            if term[0] != "D":
                return False
        return True

    @staticmethod
    def get_layer_ind(layers):
        """
        this method generates a dictionary where the key is
        a term and the value is the index of the position of the
        term in it's layer.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.

            :return dict: dictionary containing layer indexes for every
                term.
        """

        indexes = {}
        for layer in layers:
            for i, term in enumerate(layer):
                indexes[term] = i
        return indexes

    @staticmethod
    def longest_layer_index(layers):
        """
        this method searches for the index of the layer that contains
        the most terms

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.

            :return int: index of the layer with the most terms.
        """

        max, max_i = -1, -1
        for i, layer in enumerate(layers):
            if len(layer) > max:
                max = len(layer)
                max_i = i
        return max_i

    @staticmethod
    def reqursive_xpositions(layers, links, all_terms, order=[True, False]):
        """
        this method is the start and end for the method that assigns
        x positions to terms recursively. first it creates dictionaries
        so later methods have a quick way access certain information,
        also the initial starting positions of all terms are created.

        after this it will loop through the tree in the order provided,
        when it loops down the tree (dir=True) it loops through the
        layers 1 ... n-1. when it moves up the tree it loops through
        the layers n-1 ... 1, these will just be the indexes from which
        an interconnectivity matrix is calculated and do not represent
        the layers to be improved.

        after this it will loop through the layered graph one more time
        from the longest layer in the opposite direction it looped though
        last. after this the positions are returned

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param dict all_terms: a dictionary where the key is a term and
                the value is an object containing all information on that
                term.
            :param list order: order the method will go through the tree
                True = down, False = up.

            :return dict: dictionary where the key is a term and the value
                is the x position of that term.
        """

        link_dict = EdgeFix.make_link_dict(links)
        positions = EdgeFix.make_first_positions(layers)
        indexes = EdgeFix.get_layer_ind(layers)
        for dir in order:
            for i in range(len(layers))[:-1] if dir else list(range(len(layers)))[-2::-1]:
                EdgeFix.calc_layer_x_positions(link_dict, layers, all_terms, positions, indexes, dir, i)
        for i in range(len(layers))[EdgeFix.longest_layer_index(layers):-1]:
            EdgeFix.calc_layer_x_positions(link_dict, layers, all_terms, positions, indexes, not dir, i)
        for layer in layers:
            layer_positions = [positions[term] for term in layer]
            assert len(layer_positions) == len(set(layer_positions))
        return positions

    @staticmethod
    def calc_layer_x_positions(link_dict, layers, all_terms, positions, indexes, dir, i):
        """
        this task of this method is to make a priority queue from
        the layer it is given and call a recursive function to
        calculate x positions for the terms.

        first every term is assigned a priority, dummy nodes
        that have a connection to another dummy node in the
        direction the method is looking are assigned a priority
        of infinity then the priorities of the other terms is the
        same as the number of connections in the direction the
        function is looking. then a recursive method is called.

            :param dict link_dict: dictionary with the links each node contains.
            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param dict all_terms: a dictionary where the key is a term and
                the value is an object containing all information on that
                term.
            :param dict positions: positions dictionary for all terms where the term
                is the key and the x position is the value.
            :param dict indexes: dictionary containing index of the term in it's layer
            :param bool dir: direction the algorithm is moving down the layers.
                True = down, False = up.
            :param int i: index from which the interconnectivity matrix is made.

            :return None: None
        """

        queue, terms, connectivity, layer = [], [], [], layers[i + 1 if dir else i].copy()
        matrix = EdgeFix.make_interconnectivity_matrix3_dict(layers, link_dict, i)
        connectivity_list = matrix.sum(axis=0 if dir else 1)
        dummys = []
        for j, term in enumerate(layers[i + 1 if dir else i]):
            terms.append(term)
            if term[0] == "D" and EdgeFix.is_dummy_conn(all_terms, term, up=dir):
                connectivity.append(inf)
                dummys.append(term)
            else:
                connectivity.append(connectivity_list[j])
        pos, prev_term = -inf, ""
        for dummy in dummys:
            cons = all_terms[dummy].p if dir else all_terms[dummy].c
            connections = [positions[term] for (term, conn) in cons if term in positions]
            assert len(connections) == 1
            if connections[0] < pos:
                print("smaller! term: {}, pos: {}, term pos: {}".format(dummy, pos, connections[0]))
            assert connections[0] > pos
            pos = connections[0]
            prev_term = dummy
        queue = [x for _, x in sorted(zip(connectivity, terms), key=lambda x: (x[0], random()))]
        # queue = [x for _, x in sorted(zip(connectivity, terms), key=lambda x: x[0])]
        EdgeFix.reqursive_pr_method(queue, dir, positions, all_terms, indexes, -inf, inf, -1, len(layer))

    @staticmethod
    def reqursive_pr_method(queue, dir, positions, all_terms, indexes, s, e, s_index, e_index, gap=1):
        """
        this method is a recursice function that tries to place
        each term on the median position of either the parents
        or the children.

        first the median position of either the parents or the
        children is calculated and is dubbed the 'preferred
        position', in the case the node doesn't have any
        connections it's previous position is the preferred
        position. after this it is checked if the node can be
        placed on this preferred position, the bounds are the
        position of the last node placed in that direction +/-
        the nodes that still need to be placed in that direction
        +/- one space for the term itself. if the node is outside
        of the boundary it is placed on the boundary. after this
        the queue is split into two parts: terms that need to be
        placed to the left of the current term and terms that
        need to be placed to the right of this term. then this
        method is called for both the right and left side.

            :param list queue: queue with terms that still need to be placed.
                the node that needs to be placed next is on position -1.
            :param bool dir: direction the algorithm is moving down the layers.
                True = down, False = up.
            :param dict positions: positions dictionary for all terms where the
                term.
                is the key and the x position is the value
            :param dict all_terms: a dictionary where the key is a term and
                the value is an object containing all information on that
                term.
            :param dict indexes: dictionary containing index of the term in it's
                layer.
            :param int s: x position of the left most node that was already placed.
            :param int e: x position of the right most node that was already placed.
            :param int s_index: index in the layer list of the left most term that
                was already placed.
            :param int e_index: index in the layer list of the right most term that
                was already placed.
            :param int gap: gap between each term.

            :return None: None
        """

        if queue:
            node = queue.pop()
        else:
            return

        cons = all_terms[node].p if dir else all_terms[node].c
        connections = [positions[term] for (term, conn) in cons if term in positions]
        if connections:
            prefered_pos = round(np.median(connections))
        else:
            prefered_pos = positions[node]

        assert indexes[node] > s_index
        assert indexes[node] < e_index
        assert s < e
        assert s_index < e_index
        assert s + (indexes[node] - s_index) * gap <= e - (e_index - indexes[node]) * gap

        if prefered_pos < s + (indexes[node] - s_index) * gap:
            actual_pos = s + (indexes[node] - s_index) * gap
        elif prefered_pos > e - (e_index - indexes[node]) * gap:
            actual_pos = e - (e_index - indexes[node]) * gap
        else:
            actual_pos = prefered_pos
        positions[node] = actual_pos

        left_queue, right_queue = [], []
        for term in queue:
            if indexes[term] < indexes[node]:
                left_queue.append(term)
            else:
                right_queue.append(term)
        EdgeFix.reqursive_pr_method(left_queue, dir, positions, all_terms, indexes, s, actual_pos, s_index, indexes[node])
        EdgeFix.reqursive_pr_method(right_queue, dir, positions, all_terms, indexes, actual_pos, e, indexes[node], e_index)

    @staticmethod
    def find_term_index(layers, node):
        """
        finds the index of a term in it's layer

            :param list layers: list of layers where the first item is the
                    top most layer and then continues down.
            :param str node: node to be searched.

            :return int: index in the layer the term is located in.
        """

        for layer in layers:
            for i, term in enumerate(layer):
                if node == term:
                    return i
        assert False

    @staticmethod
    def get_barycenter(vector):
        """
        calculates the barycenter (or average 'position'
        of either the children or the parents, where the
        position is the index in the layer) for a node.

            :param list vector: connectivity vector of a term.
                in this vector a 1 is a connection and a 0 is
                no connection.

            :return float: barycenter of the connectivity vector.
        """

        weighted_sum, sum = 0, 0
        for x in range(len(vector)):
            if vector[x]:
                weighted_sum += x + 1
                sum += 1
        if sum == 0:
            return -1
        else:
            return weighted_sum / sum

    @staticmethod
    def get_barycenter_vector(matrix, row_alignment=True, mode=""):
        """
        makes a barycenter vector from a interconnectivity matrix,
        the vector is essentially all the barycenters for a certain
        layer.

            :param numpy.array matrix: interconnectivity matrix
                (layer i x layer i+1) each value is either a 1 or a 0
                where a 1 indicates a connection and 0 no connection.
            :param bool row_alignment: alignment from which the barycenter
                vector is calculated. True = barycenter vector of layer i,
                False = barycenter vector of layer i+1.
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.

            :return list: barycenter vector from the matrix.
        """

        barycenter_vector = []
        if row_alignment:
            for row in matrix:
                if mode == "jit":
                    if type(row) != np.ndarray:
                        row = np.array(row)
                    barycenter_vector.append(get_barycenter(row))
                else:
                    barycenter_vector.append(EdgeFix.get_barycenter(row))
        else:
            for i in range(len(matrix[0])):
                if mode == "jit":
                    barycenter_vector.append(get_barycenter([row[i] for row in matrix]))
                else:
                    barycenter_vector.append(EdgeFix.get_barycenter([row[i] for row in matrix]))
        return barycenter_vector

    @staticmethod
    def get_barycenter_vector_np(matrix, row_alignment=True):
        """
        makes a barycenter vector from a interconnectivity matrix,
        the vector is essentially all the barycenters for a certain
        layer.

        this method is a numpy implementation, it's faster than the
        standard python implementation because it uses the ufunc's
        from numpy.

            :param numpy.array matrix: interconnectivity matrix
                (layer i x layer i+1) each value is either a 1 or a 0
                where a 1 indicates a connection and 0 no connection.
            :param bool row_alignment: alignment from which the barycenter
                vector is calculated. True = barycenter vector of layer i,
                False = barycenter vector of layer i+1.

            :return numpy.array: barycenter vector from the matrix.
        """

        axis = 1 if row_alignment else 0
        weights = np.arange(1, matrix.shape[axis] + 1, dtype=np.float64)
        weighted_sum = np.sum(matrix * weights if axis == 1 else matrix * weights[:, np.newaxis], axis=axis)
        connections = matrix.sum(axis=axis)
        return np.divide(weighted_sum, connections, out=np.full(connections.shape[0], -1, dtype=np.float64), where=connections!=0)

    @staticmethod
    def get_layer_crossings(layers, links, layer_i, mode=""):
        """
        calculates the number of crossings between two layers, this
        is done by looping though all unique pairs of vertices
        between the two layers and calculating the crossing  count
        between them with the function calculate_crossings().

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param int layer_i: the index of the upper of the two layers.

            :return int: the amount of times the edges between the two
                layers cross.
        """

        crossings = 0
        link_dict = EdgeFix.make_link_dict(links)
        if mode == "jit":
            matrix = EdgeFix.make_interconnectivity_matrix_dict(layers, link_dict, layer_i, type=int)
        else:
            matrix = EdgeFix.make_interconnectivity_matrix_dict(layers, link_dict, layer_i)
        for i in range(len(layers[layer_i]) - 1):
            for j in range(i + 1, len(layers[layer_i])):
                crossings += EdgeFix.calculate_crossings(i, j, matrix)
        return crossings

    @staticmethod
    def get_graph_crossings(layers, links, mode=""):
        """
        prints the crossings between all layers in the graph.
        !!! this function will take very long on bigger graphs.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.

            :return int: number of crossings in the graph.
        """

        print("Layers: {}".format(len(layers)))
        crossings = 0
        for i in range(len(layers) - 1):
            print("Counting crossings for layer {} - {}... length: {}, {} ".format(i + 1, i + 2, len(layers[i]), len(layers[i + 1])), end="")
            cros = EdgeFix.get_layer_crossings(layers, links, i, mode=mode)
            crossings += cros
            print("crossings: {:,}; total: {:,}".format(cros, crossings))
        return crossings

    @staticmethod
    def make_penalty_graph(layers, links, layer_i):
        """
        part of the penalty minimization method.
        this method makes a directed graph of a layer, an edge represents the preferred order
        of a pair of points (for example (b, a) would suggest that if b comes before a the
        amount of crossings between these points would be optimal). the penalty associated
        with each edge is the amount of crossing that would be added if the order would be
        reversed.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param int layer_i: the index of the upper of the two layers.

            :return: a new edge list where each edge is a preferred ordering
                of vertices and a penalties list with the cost associated with
                reversing the order.
        """

        new_links = []
        penalties = {}
        conn_matrix = EdgeFix.make_interconnectivity_matrix(layers, links, layer_i)
        for i in range(len(conn_matrix[0]) - 1):
            for j in range(i + 1, len(conn_matrix[0])):
                penalty = EdgeFix.calculate_crossings(i, j, conn_matrix) - EdgeFix.calculate_crossings(j, i, conn_matrix)
                if penalty != 0:
                    if penalty < 0:
                        new_links.append((layers[layer_i][i], layers[layer_i][j]))
                        penalties[(layers[layer_i][i], layers[layer_i][j])] = abs(penalty)
                    else:
                        new_links.append((layers[layer_i][j], layers[layer_i][i]))
                        penalties[(layers[layer_i][j], layers[layer_i][i])] = abs(penalty)
        return new_links, penalties


    @staticmethod
    def print_matrix(matrix, layers, layer_i, mode="", namewidth=10):
        """
        print the interconnectivity matrix + barycenters for two
        layers.

            :param numpy.array matrix: interconnectivity matrix
                (layer i x layer i+1) each value is either a 1 or a 0
                where a 1 indicates a connection and 0 no connection.
            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param int layer_i: the index of the upper of the two layers.
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.
            :param int namewidth: space reserved for term names.

            :return None: None
        """

        b_rows = EdgeFix.get_barycenter_vector(matrix, True, mode=mode)
        b_columns = EdgeFix.get_barycenter_vector(matrix, False, mode=mode)
        print("{:^{nw}.{nw}}".format(" ", nw=namewidth) + "\t" + "\t".join(["{:^{nw}.{nw}}".format(x, nw=namewidth) for x in layers[layer_i + 1]]))
        for i, x in enumerate(layers[layer_i]):
            test = ["{:^{nw}.{nw}}".format("1", nw=namewidth) if x else "{:^{nw}.{nw}}".format("0", nw=namewidth) for x in matrix[i]]
            print("{:^{nw}.{nw}}".format(x, nw=namewidth) + "\t" + "\t".join(test) + "\t" + "{:^{nw}.{nw}}".format(str(round(b_rows[i], 1)) if type(b_rows[i]) == float else str(b_rows[i]), nw=namewidth))
        print("{:^{nw}.{nw}}".format(" ", nw=namewidth) + "\t" + "\t".join(["{:^{nw}.{nw}}".format(str(round(x, 1)) if type(x) == float else str(x), nw=namewidth) for x in b_columns]))
        print()

    @staticmethod
    def reverse_barycenters(layers, links, layer_i, isRow, mode=""):
        """
        reverses sets of terms in a layer that have the same barycenter`

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param int layer_i: the index of the upper of the two layers.
            :param bool isRow: indicates if the rows in the interconnectivity
                matrix need to be reversed or the columns. True = rows, False
                = columns.
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.

            :return bool: boolean value indicating if any terms have been
                reversed
        """

        layer_dict = EdgeFix.make_link_dict(links)
        matrix = EdgeFix.make_interconnectivity_matrix3_dict(layers, layer_dict, layer_i)
        vector = EdgeFix.get_barycenter_vector_np(matrix, isRow)
        current_barycenter, count, start_i, reversed = None, 0, 0, False
        index = layer_i if isRow else layer_i + 1
        for i, barycenter in enumerate(vector):
            if barycenter != current_barycenter:
                if barycenter != None and abs(i - start_i) > 1:
                    layers[index][start_i:i] = layers[index][start_i:i][::-1]
                    reversed = True
                current_barycenter = barycenter
                count += 1
                start_i = i
            else:
                count += 1
        return reversed

    @staticmethod
    def make_link_dict(links):
        """
        makes a dictionary where the key is a parent term and
        the value is a list of children.

            :param list links: list of all links in the graph.

            :return dict: dictionary where the key is a parent
                term and the value is a list of children.
        """

        links_dict = {}
        for (parent, child, type) in links:
            if parent not in links_dict:
                links_dict[parent] = [child]
            else:
                links_dict[parent].append(child)
            if child not in links_dict:
                links_dict[child] = []
        return links_dict

    @staticmethod
    def check_if_graph_ordered(layers, links, mode=""):
        """
        this function checks if all of the layers are sorted
        based on barycenters.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.

            :return bool: boolean value indicating if the layers are sorted
                by barycenter. True = sorted by barycenters, False = not
                sorted by barycenters.
        """

        for layer_i in range(len(layers) - 1):
            matrix = EdgeFix.make_interconnectivity_matrix3_dict(layers, links, layer_i)
            r_vector = [x for x in EdgeFix.get_barycenter_vector_np(matrix, True, mode=mode) if x != -1]
            if sorted(r_vector) != r_vector:
                return False
            c_vector = [x for x in EdgeFix.get_barycenter_vector_np(matrix, False, mode=mode) if x != -1]
            if sorted(c_vector) != c_vector:
                return False
        return True

    @staticmethod
    def sort_by_barycenter(vector, layer):
        """
        special method for sorting a layer based on barycenters
        the reason this cannot be done with a simple sorted zip is
        that nodes with a barycenter of 0 need to retain their
        original position. in this method the 0 barycenters are
        removed beforehand and then placed back after the sorting.

            :param list or numpy.array vector: vector containing the
                barycenters.
            :param list layer: list of terms in a layer.

            :return list: layer sorted by barycenters.
        """

        vector = list(vector)
        _layer, indexes, index_count = [], [], 0
        for i, x in enumerate(vector):
            if x == -1:
                vector.pop(i)
                _layer.append(layer.pop(i))
                indexes.append(i + index_count)
                index_count += 1
        layer = [x for _, x in sorted(zip(vector, layer))]
        for i, x in enumerate(_layer):
            layer.insert(indexes[i], x)
        return layer

    @staticmethod
    def barycentric_reordering_phase1_modules(layers, links, order, mode=""):
        """
        this method works it's way down or up the graph once, in the
        down phase (when it goes down the graph) it sorts all layers
        based on the layer above. in the up phase it sorts all layers
        based on the layer below(this is not represented in the indexes
        because of the way the interconnectivity matrix is made).

        during this process the time is printed to the screen.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param string order: direction the algorithm needs to go (either
                "up" or "down").
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.

            :return None: None
        """

        links_dict = EdgeFix.make_link_dict(links)
        if order == "down":
            for layer_i in range(len(layers) - 1):
                start = time()
                print("\t\tLayer {:>2}".format(layer_i + 1))
                # print("\t\t\tmaking matrix...")
                s_matrix = time()
                matrix = EdgeFix.make_interconnectivity_matrix3_dict(np.array(layers), links_dict, layer_i)
                print("\t\t\tMatrix time: {:>20.4f} Seconds".format(time() - s_matrix))
                s_bary = time()
                vector = EdgeFix.get_barycenter_vector_np(matrix, False)
                print("\t\t\t\tvector generation: {:>20.4f} Seconds".format(time() - s_bary))
                s_sorting = time()
                layers[layer_i + 1] = EdgeFix.sort_by_barycenter(vector, layers[layer_i + 1])
                print("\t\t\t\tsorting time:      {:>20.4f} Seconds".format(time() - s_sorting))
                print("\t\t\tBary time:   {:>20.4f} Seconds".format(time() - s_bary))
                print("\t\t\tTotal time:  {:>20.4f} Seconds".format(time() - start))
        else:
            for layer_i in range(len(layers) - 1, 0, -1):
                start = time()
                print("\t\tLayer {:>2}".format(layer_i + 1))
                s_matrix = time()
                matrix = EdgeFix.make_interconnectivity_matrix3_dict(np.array(layers), links_dict, layer_i - 1)
                print("\t\t\tMatrix time: {:>20.4f} Seconds".format(time() - s_matrix))
                s_bary = time()
                vector = EdgeFix.get_barycenter_vector_np(matrix, True)

                print("\t\t\t\tvector generation: {:>20.4f} Seconds".format(time() - s_bary))
                s_sorting = time()
                layers[layer_i - 1] = EdgeFix.sort_by_barycenter(vector, layers[layer_i - 1])
                print("\t\t\t\tsorting time:      {:>20.4f} Seconds".format(time() - s_sorting))
                print("\t\t\tBary time:   {:>20.4f} Seconds".format(time() - s_bary))
                print("\t\t\tTotal Time:  {:>20.4f} Seconds".format(time() - start))


    def barycentric_reordering_phase1(layers, links, order=["down", "up"], max=5, log="\t", mode=""):
        """
        =============================
        Phase 1 of barycentric method
        =============================
        this method keeps ordering all layers until the same configuration
        appears twice or the max amount of iterations is reached. to speed
        up the process and conserve memory the layers are hashed after each
        iteration

            :param list links: list of all links in the graph.
            :param string order: direction the algorithm needs to go (either
                "up" or "down").
            :param int max: maximum iterations
            :param srt log: certain amount of tabs it needs to log every print
                statement, (maybe not the most elegant but it works).
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.

            :return None: None
        """

        hashes = []
        for x in range(max):
            layer_hash = hash(tuple((tuple(x) for x in layers)))
            if layer_hash not in hashes:
                hashes.append(layer_hash)
                for dir in order:
                    print("{}Phase 1 - {}; i = {}".format(log, dir, x))
                    start = time()
                    EdgeFix.barycentric_reordering_phase1_modules(layers, links, dir, mode=mode)
                    print("Time: {} Seconds.".format(time() - start))
            else:
                return

    @staticmethod
    def barycentric_reordering_phase2(layers, links, mode="", max=5):
        """
        =============================
        Phase 2 of barycentric method
        =============================
        here all the barycenters with the same values are reversed (see:
        reverse_barycenters) and phase 1 is run again (see:
        barycentric_reordering_phase1).

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.
            :param int max: maximum iterations.

            :return None: None
        """

        # Down
        print("\tPhase 2 - down")
        reversed = False
        for layer_i in range(len(layers) - 1):
            rev = EdgeFix.reverse_barycenters(layers, links, layer_i, False, mode=mode)
            if not reversed and rev:
                reversed = True
        if reversed:
            print("\tbarycenters with same values reversed redoing Phase 1; order: down-up)")
            EdgeFix.barycentric_reordering_phase1(layers, links, order=["down", "up"], log="\t\t", mode=mode, max=max)

        # Up
        print("\tPhase 2 - up")
        reversed = False
        for layer_i in range(len(layers) - 1, 0, -1):
            _test = layers[layer_i - 1].copy()
            rev = EdgeFix.reverse_barycenters(layers, links, layer_i - 1, True, mode=mode)
            if not reversed and rev:
                reversed = True
        if reversed:
            print("\tbarycenters with same values reversed redoing Phase 1; order: up-down")
            EdgeFix.barycentric_reordering_phase1(layers, links, order=["up", "down"], log="\t\t", mode=mode, max=max)

    @staticmethod
    def n_level_barycentric_reordering(layers, links, max=5, mode=""):
        """
        initial function for the barycentric method. this method
        initiates both phase 1 and 2.

            :param list layers: list of layers where the first item is the
                top most layer and then continues down.
            :param list links: list of all links in the graph.
            :param int max: maximum iterations.
            :param str mode: type of function to be used. empty string
                = standard python, "jit" = just in time compiled function.

            :return None: None
        """

        start = time()
        for i, layer in enumerate(layers):
            print("Layer {:>2}: {} vertices".format(i+1, len(layer)))
        print("Phase 1")
        EdgeFix.barycentric_reordering_phase1(layers, links, mode=mode, max=max)
        print("Phase 2")
        EdgeFix.barycentric_reordering_phase2(layers, links, mode=mode, max=max)
        print("Total time: {:.4f}".format(time() - start))