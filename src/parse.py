#!/usr/bin/python
import time
import argparse
import pickle
import sys

import numpy as np
import pandas as pd

from MinimizePositions import minimize_positions
from GraphVizLib import EdgeFix
from random import random, shuffle

class Term:
    """
    This class represents a term.
    A list of the attributes and what they are an abbreviation of:
    id = GO id
    ns = Namespace
    df = def
    name = name
    acc1 = accuracy score model 1
    acc2 = accuracy score model 2
    imp = importance
    p = parents
    c = children
    x = x coördinate
    y = y coördinate
    """
    def __init__(self):
        self.id = "NA"
        self.ns = ["NA"]
        self.df = ["NA"]
        self.name = ["NA"]
        self.acc1 = "NA"
        self.acc2 = "NA"
        self.imp = "NA"
        self.p = set()
        self.c = set()
        self.x = "NA"
        self.y = "NA"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('obofile', help="A GO obo file")
parser.add_argument('--model1', help="The first model, this file should "
                                     "contain a column of GO terms and a column of accuracies")
parser.add_argument('--model2', help="The second model, this file should "
                                     "contain a column of GO terms and a column of accuracies")
parser.add_argument('--importance', help="A file containing the importance "
                                         "of each node ordered exactly like model1")
parser.add_argument('--storage_location', help="A filename with or with a path")
parser.add_argument('--pickled', help="file is pickled", action="store_true")
parser.add_argument('--cut_first', help="Cut before calculation of node coordinates",
                    action="store_true")
parser.add_argument('--cut_last', help="Cut after calculation of node coordinates",
                    action="store_true")
parser.add_argument('--save', help="Save node coordinates in a pickle file",
                    action="store_true")
args = parser.parse_args()

def get_user_info():
    """
    This function imports accuracies from both accuracy files
    and importanceout of the importance file, but only if the
    appropriate arguments are given when the parser is called.
    """
    if args.model1:
        with open(args.model1, "r") as model1:
            user_info = {}
            id_order = []
            for line in model1:
                goid, accuracy_model1 = line.split("\t")
                if goid not in user_info:
                    user_info[goid] = [float(accuracy_model1.rstrip('\n'))]
                    id_order.append(goid)
                else:
                    print("Duplicate accuracy found!\t", goid)
            if args.model2:
                with open(args.model2) as model2:
                    for line in model2:
                        goid, accuracy_model2 = line.split("\t")
                        if goid not in user_info:
                            print("Term lacks accuracy from first model:\t", goid)
                        else:
                            user_info[goid].append(float(accuracy_model2.rstrip('\n')))
            if args.importance:
                with open(args.importance, "r") as importance:
                    position = 0
                    for line in importance:
                        user_info[id_order[position]].append(float(line.rstrip('\n')))
                        position += 1
        return user_info
    else:
        return {}

def unpickle():
    """
    This function loads Terms from a pickled file and returns them
    as a dictionary.
    """
    all_terms = {}
    with open(args.storage_location, "rb") as infile:
        objectlist = pickle.load(infile)
        for stored_term in objectlist:
            new_term = Term()
            for id, value in stored_term:
                if id == "id":
                    new_term.id = value
                elif id == "ns":
                    new_term.ns = value
                elif id == "df":
                    new_term.df = value
                elif id == "name":
                    new_term.name = value
                elif id == "imp":
                    new_term.imp = value
                elif id == "p":
                    new_term.p = value
                elif id == "c":
                    new_term.c = value
                elif id == "x":
                    new_term.x = value
                elif id == "y":
                    new_term.y = value
    return all_terms

def getterm(stream):
    """
    This function travels through a given datastream and records all lines
    until it finds the start of a new term or the end of a useful section
    of the obofile.
    """
    block = []
    for line in stream:
        if line.strip() == "[Term]" or line.strip() == "[Typedef]":
            break
        else:
            if line.strip() != "":
                block.append(line.strip())
    return block

def parsetagvalue(term):
    """
    This function obtains all attributes of a term and puts it in
    a dictionary. Though it returns an empty string if function
    input is not an GO term.
    :param term:
    A list containing all information from one term obtained
    from the obofile.
    :return:
    """
    if not term or term[0][0:7] != 'id: GO:':
        return ''
    data = {}
    for line in term:
        tag = line.split(': ', 1)[0]
        value = line.split(': ', 1)[1]
        if tag not in data:
            data[tag] = []
        data[tag].append(value)
    return data

def fill_new_term(term, user_info):
    """
    This function fills a new Term object with attributes of a given term.
    The parents are labeled with their relational type between its child.
    User added input like accuracy and importance are only added if
    the first model accuracy has been given.
    """
    new_term = Term()
    new_term.name = term['name']
    new_term.id = term['id']
    new_term.ns = term['namespace']
    new_term.df = term['def']

    #add parents
    if "is_a" in term:
        for parent in term["is_a"]:
            new_term.p.add((parent.split(" ! ")[0], "is_a"))
    if "relationship" in term:
        for parent in term["relationship"]:
            relationship, listy = parent.split(' ', 1)
            new_term.p.add((listy.split(' ! ')[0], relationship))

    #add user info to node
    if args.model1:
        if new_term.id[0] in user_info:
            new_term.acc1 = user_info[new_term.id[0]][0]
            if not args.model2 and args.importance:
                new_term.imp = user_info[new_term.id[0]][1]
            elif args.model2 and not args.importance:
                new_term.acc2 = user_info[new_term.id[0]][1]
            elif args.model2 and args.importance:
                new_term.acc2 = user_info[new_term.id[0]][1]
                new_term.imp = user_info[new_term.id[0]][2]
            else:
                print("Something went wrong with obtaining user attributes!", new_term.id)
    return new_term

def precut(layers, links, all_terms, user_info):
    """
    This function cuts terms in layers if they do not exist inside
    the accuracy file of model 1.
    It also cuts all links if one of the terms inside does not exist
    inside the accuracy file of model 1.
    Finaly it cuts all terms taht do not exist inside the accuracy
    file of model 1.
    :return:
    Cut layers, links (edges) and terms are returned.
    """
    new_layers = []
    for layer in layers:
        new_layer = []
        for node in layer:
            if node in user_info:
                new_layer.append(node)
        if len(new_layer) != 0:
            new_layers.append(new_layer)
        else:
            new_layer = []
    new_links = set()
    for link in links:
        if link[0] in user_info and link[1] in user_info:
            new_links.add(link)
    new_all_terms = {}
    for term in all_terms:
        if term in user_info:
            new_all_terms[term] = all_terms[term]
    return new_layers, new_links, new_all_terms


def layer_assignment(terms):
    """
    This function implements a longest-path algorithm
    to assign nodes to layers. this algorithm is based on
    Healy,  P.,  &  Nikolov,  N.  S.  (2013).  Hierarchical
    Drawing  Algorithms.  Handbook  of  Graph  Drawing
    and  Visualization,  409–453
    :return:
    A list containing layers (lists).
    """
    terms_copy = {}
    for goid in terms:
        terms_copy[goid] = terms[goid].p
    final_layers = []
    all_layers = {}
    last_layer = {}
    while 0 != len(terms_copy):
        current_layer = []
        for term in last_layer:
            terms_copy.pop(term)
        for term in terms_copy:
            add_to_layer = True
            for child in terms_copy[term]:
                if child[0] not in last_layer and child[0] not in all_layers:
                    add_to_layer = False
            if add_to_layer:
                current_layer.append(term)
        for term in last_layer:
            all_layers[term] = 0
        if len(current_layer) != 0:
            final_layers.append(current_layer)
        last_layer = {}
        for item in current_layer:
            last_layer[item] = 0
    final_layers.reverse()
    return final_layers

def edges_and_links(all_terms):
    """
    This function assigns children of a node to the parent
    and defines links (edges) during this action.
    :return:
    a list of sets containing (parent, child, relationship type)
    """
    links = set()
    for term_id in all_terms:
        term = all_terms[term_id]
        for parent in term.p:
            links.add((parent[0], term_id, parent[1]))
            all_terms[parent[0]].c.add((term_id, parent[1]))
    return links

def store_obofile(all_terms):
    """
    This function stores a dictionary of Term objects into a file
    through the use of the pickle module.
    """
    with open(args.storage_location, "wb") as outfile:
        pickableterms = []
        for term in all_terms:
            term_list = []
            for i in all_terms[term].__dict__:
                term_list.append([i, all_terms[term].__getattribute__(i)])
            pickableterms.append(term_list)
            term_list = []
        gurkin = pickle.dumps(pickableterms)
        outfile.write(gurkin)


def fill_edge_dict(layers, links, all_terms, edge_dict):
    """
    This function fills the edge dataframe dictionary
    with appropriate content to the edge
    :param layers:
    A list of layers.
    :param links:
    A list of link sets.
    :param all_terms:
    A dictionary with the GO id's as keys and term objects as values.
    :param edge_dict:
    A dictionary resembeling a dataframe with the keys
    as column names and values as rows.
    """
    color_dict = {"is_a": "pink", "regulates": "blue", "part_of": "lilac",
                  "negatively_regulates": "red", "positively_regulates": "green", "NA": "black"}
    layer_dict = EdgeFix.make_layer_dict(layers)
    for (term1, term2, edgetype) in links:
        if "DUMMY" in term2:
            if "DUMMY" not in term1:
                child, start, cnt, xlist, ylist = term2, layer_dict[term2], 1, [], []
                edge_dict["Term1"].append(term1)
                edge_dict["Type"].append(edgetype)
                edge_dict["Color"].append(color_dict[edgetype])
                xlist.append(all_terms[term1].x)
                ylist.append(layer_dict[term1])
                while "DUMMY" in child:
                    xlist.append(all_terms[child].x)
                    ylist.append(layer_dict[child])
                    assert len(all_terms[child].c) == 1
                    child = list(all_terms[child].c)[0][0]
                xlist.append(all_terms[child].x)
                ylist.append(layer_dict[child])
                edge_dict["Term2"].append(child)
                edge_dict['x'].append(xlist)
                edge_dict['y'].append(ylist)
        elif "DUMMY" in term1:
            pass
        else:
            edge_dict["Term1"].append(term1)
            edge_dict["Term2"].append(term2)
            edge_dict["Type"].append(edgetype)
            edge_dict["Color"].append(color_dict[edgetype])
            edge_dict['x'].append([all_terms[term1].x, all_terms[term2].x])
            edge_dict['y'].append([layer_dict[term1], layer_dict[term2]])


def fill_node_dict(layers, links, all_terms, node_dict):
    """
    This function fills the node dataframe dictionary
    with appropriate content to the term.
    :param layers:
    A list of layers.
    :param links:
    A list of link sets
    :param all_terms:
    A dictionary with the GO idś as keys and term objects as values.
    :param node_dict:
    A dictionary resembeling a dataframe with the keys
    as column names and values as rows.
    """
    layer_dict = EdgeFix.make_layer_dict(layers)
    layer_len = len(layers)
    for term in all_terms:
        if "DUMMY" not in term:
            term_obj = all_terms[term]
            node_dict['Term'].append(str(term))
            node_dict['X'].append(term_obj.x)
            node_dict['Y'].append(layer_dict[term])
            node_dict['Score1'].append(term_obj.acc1)
            node_dict['Score2'].append(term_obj.acc2)
            node_dict['SearchDescription'].append(" ".join([term_obj.df[0], term_obj.ns[0], term_obj.name[0]]))
            node_dict['Description'].append(term_obj.df[0])
            node_dict['Namespace'].append(term_obj.ns[0])
            node_dict['Importance'].append(term_obj.imp)
            node_dict['Difference'].append(term_obj.acc1 - term_obj.acc2 if term_obj.acc1 != "NA" and term_obj.acc2 != "NA" else "NA")
            node_dict['Top'].append(layer_dict[term] + 0.3)
            node_dict['Bottom'].append(layer_dict[term] - 0.3)
            node_dict['Layer'].append(str(layer_len - layer_dict[term] - 1))
    index_dict = {}
    for i, term in enumerate(node_dict['Term']):
        index_dict[term] = i
    for term in node_dict['Term']:
        if "DUMMY" not in term:
            children, parents = [], []
            for child in all_terms[term].c:
                if "DUMMY" not in child[0] and child[0] in index_dict:
                    children.append(index_dict[child[0]])
            for parent in all_terms[term].p:
                if "DUMMY" not in parent[0] and parent[0] in index_dict:
                    parents.append(index_dict[parent[0]])
            node_dict['Children'].append(children)
            node_dict['Parents'].append(parents)


def make_tree_dataframe(layers, links, all_terms):
    """
    This function constructs two dictionaries which represents
    the term dataframe and edge dataframe.
    :param layers:
    List of layers
    :param links:
    List of link sets
    :param all_terms:
    A dictionary with the GO idś as keys and term objects as values.
    :return:
    edge dataframe, node dataframe
    """
    edge_dict = {"Term1": [], "Term2": [], "x": [], "y": [], "Type": [], "Color": []}
    node_dict = {"Term": [], "X": [], "Y": [], "Score1": [], "Score2": [], "SearchDescription": [],
                 "Description": [], "Namespace": [], "Importance": [], "Difference": [], "Top": [],
                 "Bottom": [], "Parents": [], "Children": [], "Layer": []}
    fill_edge_dict(layers, links, all_terms, edge_dict)
    fill_node_dict(layers, links, all_terms, node_dict)
    edges = pd.DataFrame.from_dict(edge_dict)
    nodes = pd.DataFrame.from_dict(node_dict)
    return edges, nodes


def main():
    #set timer
    t0 = time.time()
    #allow more recursion in system (increases RAM usage)
    sys.setrecursionlimit(50000)
    #check if the obo file is pickled
    if args.pickled:
        user_info = get_user_info()
        all_terms = unpickle()
    else:
        #obtain terms from obofile and fill objects
        all_terms = {}
        user_info = get_user_info()
        with open(args.obofile) as obofile:
            getterm(obofile)
            while 1:
                term = parsetagvalue(getterm(obofile))
                if len(term) != 0:
                    if term['id'][0] not in all_terms and "is_obsolete" not in term:
                        all_terms[term['id'][0]] = fill_new_term(term, user_info)
                else:
                    break
        #find links and add children
        links = edges_and_links(all_terms)
        #assign terms to layers
        layers = layer_assignment(all_terms)[::-1]
    if args.cut_first:
        #cut all terms not in the model 1 accuracy file from the
        #layers, links and all_terms dictionary
        layers, links, all_terms = precut(layers, links, all_terms, user_info)
    #get current time
    t1 = time.time()
    #calculate total seconds
    total = t1-t0
    #print time taken by parser
    print('Time parser:', str(total))
    print("making dummys!")
    #print layers with their length
    for i, layer in enumerate(layers):
        print("Layer {:>2}: {} vertices".format(i + 1, len(layer)))
    #add dummy nodes
    EdgeFix.add_dummy_nodes(layers, links, all_terms)
    #fix term order
    EdgeFix.n_level_barycentric_reordering(layers, links, mode="jit")
    #find x positions
    xpositions = EdgeFix.reqursive_xpositions(layers, links, all_terms)
    #store x positions
    for term in xpositions:
        all_terms[term].x = xpositions[term]
    if args.save:
        #save term objects
        store_obofile(all_terms)
    #for term in all_terms:
    #     all_terms[term].acc1 = random()
    #     all_terms[term].acc2 = random()
    #     all_terms[term].imp = random() * 4
    #produce dataframes
    edges, terms = make_tree_dataframe(layers, links, all_terms)
    #write dataframes pickles
    with open("layers.pkl", "wb") as file:
        pickle.dump(layers, file)
    with open("links.pkl", "wb") as file:
        pickle.dump(links, file)
    with open("all_terms.pkl", "wb") as file:
        pickle.dump(all_terms, file)
    edges.to_pickle("./edges_dataframe.pkl")
    terms.to_pickle("./terms_dataframe.pkl")



main()