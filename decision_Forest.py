'''
Made by: Duggan Burke
CSE 415

An implementiation of decision forests as a trainable classifier
'''

# Take a random subset of the data
# Make a new tree
# Train that tree using the random subset of data
# Bag it all

#We will be having 60000 images to process in out first dataset and quite a few to process for our second dataset

import decisionTree

class Forest:


    def __init__(numTrees, dataSet):
        #This will make the number of trees as needed and save a reference to the dataset we will use
        self.trees = ['an array with a bunch of trees in it']
        pass

    def trainTrees():
        for tree in self.trees:
            # gather X number of random images from the dataset and feed them into tree
        pass

    def predictClassifiers(images):
        decisions = {image, [decisions]}
        finalDecisions = {image, decision}
        for tree in self.trees:
            #feed each image through the tree and record its response in the decisions dictionary
        for image in decisions:
            #find the majority decision for the forest and write that to finalDecisions
            #will be accomplished with a simple for loop to tally up each response in the corresponding array

        return finalDecisions