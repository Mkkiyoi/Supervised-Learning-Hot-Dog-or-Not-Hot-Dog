''' decision_tree.py
Matthew Kiyoi
mkkiyoi

Implementation of decision tree classifier based on CART algorithm
'''


import math
import numpy as np
import pickle
from sklearn import datasets
from pprint import pprint

class decision_node:
  '''
  Represents a node in the decision tree.
  leaf - true if the node is a leaf node, false if the node is the root or intermediate node
  true_branch - true subtree
  true_branch - false subtree
  prediction - what the current decision tree has classified the given data as
  '''

  def __init__(self, leaf = False, column = -1, attribute = None, prediction = None, true_branch = None, false_branch = None):
    self.leaf = leaf
    self.attribute_column = column
    self.attribute = attribute
    self.prediction = prediction
    self.true_branch = true_branch
    self.false_branch = false_branch


def get_dataset(file):
  '''
  Extracts CIFAR-10 data from the file
  '''
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def get_dataset_labels(file):
  '''
  Extracts CIFAR-10 data labels from the file
  '''
  with open(file, 'rb') as fo:
    dict = pickle.load(fo)
  return dict

def preprocess_data(data):
  dataset = data[b'data']
  types = data[b'labels']
  results = []
  for i in range(len(dataset)):
     results.append(np.append(dataset[i], types[i]))
  return results
    

def entropy(dataset):
  '''
  Returns 
  '''
  n = len(dataset)+0.0 # denominator for probabilities
  counts = find_unique_values(dataset)
  entropy = 0.0
  for count in counts:
    p = counts[count]/n # probability of this item within the population
    logp = math.log(p,2) # log, base 2, of the probability
    entropy -= p * logp # accumulation by subtraction of a negative number
  return entropy

## print(entropy([9,2,1,1,1,9])) # sample call with a “bag” represented as a list. 1.459


def is_numeric(value):
  '''
  Returns true if the value is numeric
  '''
  return isinstance(value, int) or isinstance(value, float)

def find_unique_values(dataset):
  '''
  Finds the unique values for a column in the dataset we are looking at.
  '''
  counts = {}
  for data in dataset:
    value = data[-1]
    if value not in counts:
      counts[value] = 0
    counts[value] += 1
  return counts

def split(dataset, attribute, value):
  '''
  Finds the best split based on the attribute selected and the value given.
  Checks equality for strings and checks >= for numeric data
  '''
  split_function = None
  if is_numeric(value):
    split_function = lambda data : data[attribute] >= value # Function to check if numeric value is less than data at the attribute
  else:
    split_function = lambda data : data[attribute] == value # Function to check if string value is equal to string data at attribute
  true_branch = []
  false_branch = []
  for data in dataset:
    if split_function(data):
      true_branch.append(data) # Divides dataset into a branch with values that return true from the condition
    else:
      false_branch.append(data) # Divides dataset into a branch with values that return false from the condition
  return (true_branch, false_branch)


def variance(rows):
  if len(rows) == 0: return 0
  data = [float(row[len(row) - 1]) for row in rows]
  mean = sum(data) / len(data)

  variance = sum([(d-mean)**2 for d in data]) / len(data)
  return variance


def build_decision_tree(dataset):
  '''
  Builds and returns a binary decision tree.
  Uses entropy as an evaluation
  '''

  if len(dataset) == 0:
    return decision_node()
  current_entropy = entropy(dataset) # Get the entropy of the current dataset
  best_info_gain = 0.0               # Store the best information gain
  best_attribute = None              # Store the best attribute to split on
  best_split = None                  # Store the split dataset as two sets
  num_attributes = len(dataset[0]) - 1 # Want to compare each attribute, where the last element is the value
  
  for attribute in range(0, num_attributes): # Loop through attributes, find which one gives the best gain
    attribute_values = [data[attribute] for data in dataset] # Get the attribute values for the attribute for each row of data
    for value in attribute_values:
      (true_branch, false_branch) = split(dataset, attribute, value) # Split the sets on the given attribute with the given value
      prob = float(len(true_branch)) / len(dataset) # get the probability that a row of data is in the true branch
      info_gain = current_entropy - prob * entropy(true_branch) - (1-prob) * entropy(false_branch) # Get the information gain splitting on this attribute with the value
      if info_gain > best_info_gain and true_branch and false_branch: # Check if we have greater info gain and both true/false branches are nonempty
        best_info_gain = info_gain                                    # Set the best information gain to the new information gain
        best_attribute = (attribute, value)                           # Set the best attribute to split on to the current attribute being considered
        best_split = (true_branch, false_branch)                      # Set the best dataset split to the current true/false branches
    if best_info_gain > 0: # If we have information gain keep building the decision tree
      true_branch = build_decision_tree(best_split[0]) # Recursively build decision tree on true branch set
      false_branch = build_decision_tree(best_split[1]) #  Recursively build decision tree on false branch set
      return decision_node(False, best_attribute[0], best_attribute[1], true_branch, false_branch)
    else: # No more information gain
      return decision_node(leaf = True, prediction = find_unique_values(dataset))
    

def classify_data(unknown_data, decision_tree):
  if decision_tree.leaf: # If we are at a leaf, return prediction of what data is
    return decision_tree.prediction
  else: # Recursively narrow down what data is using the value of the best attribute at current node in the tree
    value = unknown_data[decision_tree.attribute_column]
    branch = None
    if is_numeric(value):
      if value >= decision_tree.attribute:
        branch = decision_tree.true_branch
      else:
        branch = decision_tree.false_branch
    else:
      if value == decision_tree.attribute:
        branch = decision_tree.true_branch
      else:
        branch = decision_tree.false_branch
    return classify_data(unknown_data, branch)
        
iris_attributes = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
cifar_labels = []
def print_tree(tree):
  if tree:
    if tree.leaf:
      print("Leaf: " + cifar_labels[tree.prediction[1]])
    else:
      print("Branch: " + str(tree.attribute_column)+ ' = ' + str(tree.attribute))
      print_tree(tree.true_branch)
      print('Done with true branch.')
      print_tree(tree.false_branch)




if __name__ == '__main__':
##  data = datasets.load_iris()
##  dataset = preprocess_data(data)
  files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
  label_file = 'batches.meta'
  cifar_labels = get_dataset_labels(label_file)['label_names']
  data = get_dataset(files[0])
  dataset = preprocess_data(data)
  decision_tree = build_decision_tree(dataset)
  print_tree(decision_tree)
  

  
  
  






















