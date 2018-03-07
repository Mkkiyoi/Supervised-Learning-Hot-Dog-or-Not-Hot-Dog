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

##  def __str__(self):
##    result = ''
##    result += ('(Leaf: ' + str(self.leaf) + '\n')
##    result += ('Attribute column: ' + str(self.attribute_column) + '\n')
##    result += ('Attribute: ' + str(self.attribute) + '\n')
##    result += ('Prediction: ' + str(self.prediction) + '\n')
##    result += ('True Branch: ' + str(self.true_branch) + '\n')
##    result += ('False Branch: ' + str(self.false_branch) + '\n')
##    return result


def get_dataset(file):
  '''
  Extracts CIFAR-10 data from the file
  '''
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def get_dataset_labels(file):
  '''
  Extracts CIFAR-10 data labels from the file.
  Labels are encoded as strings instead of bytes
  '''
  with open(file, 'rb') as fo:
    dict = pickle.load(fo)
  return dict

def preprocess_data(data, labels, num):
  '''
  
  '''
  print('Preprocessing data...')
  dataset = data[b'data']
  types = data[b'labels']
  result_0 = []
  result_1 = []
  result_2 = []
  result_3 = []
  result_4 = []
  result_5 = []
  result_6 = []
  result_7 = []
  result_8 = []
  result_9 = []
  for i in range(len(dataset)):
    if types[i] == 0:
      if len(result_0) < num:
        result_0.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 1:
      if len(result_1) < num:
        result_1.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 2:
      if len(result_2) < num:
        result_2.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 3:
      if len(result_3) < num:
        result_3.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 4:
      if len(result_4) < num:
        result_4.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 5:
      if len(result_5) < num:
        result_5.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 6:
      if len(result_6) < num:
        result_6.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 7:
      if len(result_7) < num:
        result_7.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 8:
      if len(result_8) < num:
        result_8.append(np.append(dataset[i], labels[types[i]]))
    if types[i] == 9:
      if len(result_9) < num:
        result_9.append(np.append(dataset[i], labels[types[i]]))
    results = result_0 + result_1 + result_2 + result_3 + result_4 + result_5 + result_6 + result_7 + result_8 + result_9
    np.random.shuffle(results)
  return results

def is_numeric(value):
  '''
  Returns true if the value is numeric
  '''
  return isinstance(value, int) or isinstance(value, float)

def count_values(dataset):
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

def entropy(dataset):
  '''
  Returns 
  '''
  n = float(len(dataset)) # denominator for probabilities
  counts = count_values(dataset)
  entropy = 0.0
  for count in counts:
    p = float(counts[count])/n # probability of this item within the population
    logp = math.log(p,2) # log, base 2, of the probability
    entropy -= p * logp # accumulation by subtraction of a negative number
  return entropy

def gini_index(dataset):
  counts = count_values(dataset)
  impurity = 1
  for count in counts:
    probability = counts[count]/float(len(dataset))
    impurity -= probability**2
  return impurity

    
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

def build_decision_tree(dataset = [], max_level = None, level = 0):
  '''
  Builds and returns a binary decision tree.
  Uses gini impurity to evaluate sets and information gain
  '''

  if len(dataset) == 0:
    return decision_node()
  if max_level != None and level == max_level:
    value_prediction = count_values(dataset)
    return decision_node(True, None, None, value_prediction, None, None)
##  current_entropy = entropy(dataset) # Get the entropy of the current dataset
  gini = gini_index(dataset)
  num_attributes = len(dataset[0]) - 1 # Want to compare each attribute, where the last element is the value

  for attribute in range(0, num_attributes): # Loop through attributes, find which one gives the best gain
    best_info_gain = 0.0               # Store the best information gain
    best_attribute = None              # Store the best attribute to split on
    best_split = None                  # Store the split dataset as two sets
    attribute_values = [data[attribute] for data in dataset] # Get the attribute values for the attribute for each row of data
    for value in attribute_values:
      (true_branch, false_branch) = split(dataset, attribute, value) # Split the sets on the given attribute with the given value
      prob = float(len(true_branch)) / len(dataset) # get the probability that a row of data is in the true branch
      info_gain = gini - prob * gini_index(true_branch) - (1-prob) * gini_index(false_branch) # Get the information gain splitting on this attribute with the value
      if info_gain > best_info_gain and true_branch and false_branch: # Check if we have greater info gain and both true/false branches are nonempty
        best_info_gain = info_gain                                    # Set the best information gain to the new information gain
        best_attribute = (attribute, value)                           # Set the best attribute to split on to the current attribute being considered
        best_split = (true_branch, false_branch)                      # Set the best dataset split to the current true/false branches
    if best_info_gain > 0: # If we have information gain keep building the decision tree
      # print('Splitting on attribute: ' + str(best_attribute)+ ', with information gain of: ' + str(best_info_gain))
      true_branch = build_decision_tree(best_split[0], max_level, level+1) # Recursively build decision tree on true branch set
      false_branch = build_decision_tree(best_split[1], max_level, level+1) #  Recursively build decision tree on false branch set
      return decision_node(False, best_attribute[0], best_attribute[1], None, true_branch, false_branch)
    else: # No more information gain
      value_prediction = count_values(dataset)
      # print('No more information gain, created leaf node: ' + str(value_prediction))
      return decision_node(True, None, None, value_prediction, None, None)
    

def classify_data(unknown_data, decision_tree):
  '''
  Classifies data using the decision tree. 
  '''
  if decision_tree != None:
    if decision_tree.leaf: # If we are at a leaf, return prediction of what data is
      best_prediction = None
      best_prediction_count = 0
      for value in decision_tree.prediction:
        count = decision_tree.prediction[value]
        # print('Count is ' + str(count) + ' for ' + value)
        if count > best_prediction_count:
          best_prediction_count = count
          best_prediction = value
      return best_prediction
    else: # Recursively narrow down what data is using the value of the best attribute at current node in the tree
      value = unknown_data[decision_tree.attribute_column]
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

def prune_tree(tree, min_info_gain):
  '''
  Prunes the decision tree based on a minimum expected information gain. 
  '''
  if not tree.true_branch.leaf: # If the true branch is not a leaf, recurse on the branch
    prune_tree(tree.true_branch, min_info_gain)
  if not tree.false_branch.leaf: # If the false branch is not a leaf, recurse on the branch
    prune_tree(tree.false_branch, min_info_gain)
  if tree.true_branch.leaf and tree.false_branch.leaf: # If both branches are leaves, check to see if they can combined
    true_branch = []
    false_branch = []
    for value in tree.true_branch.prediction:
        true_branch += [[value]] * tree.true_branch.prediction[value] # Recreate the number of values originally in this branch
    for value in tree.false_branch.prediction:
        false_branch += [[value]] * tree.false_branch.prediction[value] # Recreate the number of values originally in this branch
    prob = float(len(true_branch)) / len(true_branch + false_branch) # Get the probability that a row of data is in the true branch
    delta_info_gain = gini_index(true_branch + false_branch) - prob * gini_index(true_branch) - (1-prob) * gini_index(false_branch)
    if delta_info_gain < min_info_gain: # Prune the the leaves and make the branch a leaf combining the sets of the original leaves 
      print('Pruning branch, information gain was: ' + str(delta_info_gain) + '\n')
      tree.true_branch = None
      tree.false_branch = None
      tree.leaf = True
      tree.prediction = count_values(true_branch + false_branch)
    
iris_attributes = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
cifar_labels = []
def print_tree(tree, level):
  '''
  prints out a textual representation of the decision tree.
  '''
  if tree:
    if tree.leaf:
      print('Leaf: ' + str(tree.prediction) + ', Level: ' + str(level) + '\n')
    else:
      print('Branch: ' + str(tree.attribute_column)+ ' = ' + str(tree.attribute) + ', Level: ' + str(level) + '\n')
      print_tree(tree.true_branch, level+1)
      print_tree(tree.false_branch, level+1)

def test():
  files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
  label_file = 'batches.meta'
  
  cifar_labels = get_dataset_labels(label_file)['label_names'] # Get the string representation of the labels for CIFAR data
  data = get_dataset(files[0])                                 # Get the 10000 x 3072 array of picture RGB values as trainging data
  training_dataset = preprocess_data(data, cifar_labels, 100)                # Append the label to the end of each picture array
  decision_tree = build_decision_tree(training_dataset, None, 0)          # Build the decision tree
##  prune_tree(decision_tree, 0.0027)                               # Prune the decision tree based on some minimum information gain
  print_tree(decision_tree, 0)                                 # print out the tree
  
  test_data = get_dataset(files[5])                            # Get test data
  test_dataset = preprocess_data(test_data, cifar_labels, 100)      # Similarly append labels to arrays
  total_successful = 0                                         # Record successful classifications
  total_unsuccessful = 0                                       # Record unsuccessful classifications
  for data in test_dataset:                                         # Classify all of the images in the test dataset
    classification = classify_data(data, decision_tree)
    actual_classification = data[-1]
    print('Classified ' + str(actual_classification) + ' as ' + str(classification) + '.')
    if classification == actual_classification:
      total_successful += 1
    else:
      total_unsuccessful += 1
  print('Statistics:\n')
  print('Percent successfully classified: ' + str(float(total_successful)/len(test_dataset)) + '\n')
  

if __name__ == '__main__':
  test()
  
  


  
  
  





















