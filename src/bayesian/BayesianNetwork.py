"""
Data structure of a Bayesian Belief Network
as a Directed Acycling Graph (DAG)

Usage:

Options:

Examples:

License:

Copyright (c) 2015 Sebastien Dery

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from src.network.graph import Node, UndirectedNode, connect
from src.network.graph import Graph, UndirectedGraph



class BayesianBeliefNetwork(Graph):
    """
    A Directed Acyclic Graph

    """

    def __init__(self, nodes_dict, name=None, domains={}):
        """
        """
        self.nodes = nodes_dict.values()
        self.vars_to_nodes = nodes_dict
        self.domains = domains
        




