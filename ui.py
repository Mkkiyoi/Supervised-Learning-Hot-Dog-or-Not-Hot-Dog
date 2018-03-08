'''
Made by Duggan Burke
CSE 415

A command line interface for using various classifiers 
'''

import sys, os
import decision_tree
import decision_Forests


def main_menu():
    os.system('clear')

    print("Yo wassup\n")
    print("What would you like to run?? Input '1' for a Decision Tree or '2' for a Random Forest")
    choice = raw_input(" >>  ")
    exec_menu(choice)

def exec_menu(choice):
    os.system('clear')
    cho = choice.lower()
    if cho == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            menu_actions['main_menu']()
    return

def menu_1():
    print("\nType the filename you would like to run the classifier on")
    filename = raw_input(" >>  ")
    print("\nPlease type the filename with the types of classifiers listed")
    label_file = raw_input(" >>  ")

    prepped_data = prep_data(filename, label_file)

    tree = decision_tree.build_decision_tree(prepped_data)
    decision_tree.print_tree(tree, 100)

def menu_2():
    print("\nType the filename you would like to run the classifier on")
    filename = raw_input(" >>  ")
    print("\nPlease type the filename with the types of classifiers listed")
    label_file = raw_input(" >>  ")
    print("\nHow many trees would you like to make in your forest?")
    num_trees = raw_input(" >>  ")

    prepped_data = prep_data(filename, label_file)
    forest = decision_Forests.Forest(num_trees, prepped_data)
    forest.trainTrees()
    guesses = forest.predictClassifiers(prepped_data)


def prep_data(filename, label_file):
    data = decision_tree.get_dataset(filename)
    cifar_labels = decision_tree.get_dataset_labels(label_file)['label_names']
    cleandata = decision_tree.preprocess_data(data, cifar_labels)

    return cleandata

def back():
    menu_actions['main_menu']()
 
def exit():
    sys.exit()

menu_actions = {
    'main_menu': main_menu,
    '1': menu1,
    '2': menu2,
    '9': back,
    '0': exit,
}
