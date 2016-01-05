
import search
import util
import sys
import copy

class Node:

	depth = 0
	parent_node = None
	curr_state = None

	def __init__(self, parentNode, currState, depth): 
		parent_node = parentNode
		curr_state = currState
		depth = depth

	def getParentNode(self):
		return self.parent_node

	def getCurrState(self):
		return self.curr_state

	def getDepth(self):
		return self.depth

	def addDepth(self, num):
		self.depth += num


