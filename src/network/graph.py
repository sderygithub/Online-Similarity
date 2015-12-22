"""
Generic Graph Class
Used in prototype Bayesian Network

GraphViz export code was adapted from 
http://stackoverflow.com/questions/23734030/how-can-python-write-a-dot-file-for-graphviz-asking-for-some-edges-to-be-colored

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

from StringIO import StringIO

class Node(object):

    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents[:]
        self.children = children[:]

    def __repr__(self):
        return '<Node %s>' % self.name


class UndirectedNode(object):

    def __init__(self, name, neighbours=[]):
        self.name = name
        self.neighbours = neighbours[:]

    def __repr__(self):
        return '<UndirectedNode %s>' % self.name


class Graph(object):

    def export(self, filename=None, format='graphviz'):
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        fh.write(self.get_graphviz_source())

    def get_topological_sort(self):
        l = []
        l_set = set()
        s = [n for n in self.nodes.values() if not n.parents]
        s.sort(reverse=True, key=lambda x:x.variable_name)
        while s:
            n = s.pop()
            l.append(n)
            l_set.add(n)
            # Now some of n's children may be
            # added to s if all their parents
            # are already accounted for.
            for m in n.children:
                if set(m.parents).issubset(l_set):
                    s.append(m)
                    s.sort(reverse=True, key=lambda x:x.variable_name)
        if len(l) == len(self.nodes):
            return l
        raise "Graph Has Cycles"


class UndirectedGraph(object):

    def __init__(self, nodes, name=None):
        self.nodes = nodes
        self.name = name

    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('graph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
            for neighbour in node.neighbours:
                edge = [node.name, neighbour.name]
                edges.add(tuple(sorted(edge)))
        for source, target in edges:
            fh.write('  %s -- %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()

    def export(self, filename=None, format='graphviz'):
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        fh.write(self.get_graphviz_source())


def connect(parent, child):
    parent.children.append(child)
    child.parents.append(parent)
