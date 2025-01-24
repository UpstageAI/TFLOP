"""
This script is based on the TED metric implementation code from the PubTabNet GitHub repository:
https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py
"""

from collections import deque

from apted import APTED, Config
from apted.helpers import Tree
import distance
from lxml import etree, html


class TableTree(Tree):
    """
    A representation of a tree structure used for computing edit distances between tables.

    Args:
        tag (str): The HTML tag of the node (e.g., "td", "tr", etc.).
        colspan (int, optional): The colspan attribute of the cell. Defaults to None.
        rowspan (int, optional): The rowspan attribute of the cell. Defaults to None.
        content (list, optional): Tokenized content of the cell. Defaults to None.
        *children: Child nodes of the current node.

    Methods:
        bracket():
            Returns a string representation of the tree in bracket notation.
    """

    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """
        Returns a string representation of the tree using bracket notation.

        Returns:
            str: Bracketed representation of the tree.
        """
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    """
    Custom configuration for APTED (Tree Edit Distance).

    Methods:
        maximum(*sequences):
            Returns the maximum possible value of a sequence.
        normalized_distance(*sequences):
            Computes a normalized Levenshtein distance between sequences.
        rename(node1, node2):
            Compares two nodes based on their attributes and content.
    """

    @staticmethod
    def maximum(*sequences):
        """
        Returns the maximum length among the given sequences.

        Args:
            *sequences: A variable number of sequences.

        Returns:
            int: The maximum length.
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """
        Computes a normalized Levenshtein distance between sequences.

        Args:
            *sequences: A variable number of sequences.

        Returns:
            float: A normalized distance between 0 and 1.
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """
        Compares attributes and content of two tree nodes.

        Args:
            node1 (TableTree): The first node.
            node2 (TableTree): The second node.

        Returns:
            float: The cost of renaming the nodes. Returns 0.0 if identical, 1.0 otherwise.
        """
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """
    Tree Edit Distance-based Similarity (TEDS) for evaluating table structure similarity.

    Args:
        structure_only (bool): If True, evaluates only the table structure, ignoring content.
        n_jobs (int): Number of parallel jobs for computation. Defaults to 1.
        ignore_nodes (list, optional): List of node tags to ignore during evaluation.

    Methods:
        tokenize(node):
            Tokenizes the text and structure of an HTML node.
        load_html_tree(node, parent=None):
            Converts an HTML tree into a tree structure compatible with APTED.
        evaluate(pred, true):
            Computes the TEDS score between predicted and ground truth table structures.
    """

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
            n_jobs >= 1
        ), "n_jobs must be an integer greater than 1"
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """
        Tokenizes an HTML node and its content into a list of tokens.

        Args:
            node (lxml.etree.Element): The HTML node to tokenize.
        """
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """
        Converts an HTML tree into a TableTree structure for APTED computation.

        Args:
            node (lxml.etree.Element): The root HTML node.
            parent (TableTree, optional): The parent node in the TableTree structure.

        Returns:
            TableTree: The converted TableTree structure.
        """
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """
        Computes the TEDS score between a predicted table and a ground truth table.

        Args:
            pred (str): HTML string of the predicted table.
            true (str): HTML string of the ground truth table.

        Returns:
            float: TEDS score between 0.0 and 1.0, where 1.0 indicates perfect similarity.
        """
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)

        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(
                tree_pred, tree_true, CustomConfig()
            ).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0
