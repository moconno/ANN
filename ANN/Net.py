'''
Created on Nov 8, 2015

@author: Michael O'Connor
'''

from ANN.Neuron import Neuron
import math

class Net(object):
    
    ''' the neural network '''
    layers = []
    
    ''' the hash table for the targets '''
    targets = {}
    
    ''' The target value for a set'''
    target = 0.00
    
    '''the error for the output layers neurons outputvalues'''
    error = 0.00
    
    
    
    ''' layers is a list of input, hidden, and output node sizes '''
    def __init__(self, layers):
         
        for index in range(0, len(layers)): 
            
            layer = Layer()
            
            '''create the neurons for the layer'''    
            ''''check if it is at the output layer '''
            if(index != len(layers)-1):
                for i in range(0, layers[index]):
                    layer.addneuron(Neuron(i, layers[index + 1]))
            else:
                for i in range(0, layers[index]):
                    layer.addneuron(Neuron(i))
                    
            self.layers.append(layer)
                    
               
    def feedforward(self):
        
        ''' start after the input layer'''
        for layer in self.layers[1:]:
            
            '''  get the previous layer '''
            prevlayer = self.layers[self.layers.index(layer) - 1]
            
            for index in range(0, len(layer.getneurons())):
                layer.getneuron(index).feedforward(prevlayer)
        
    def backpropogate(self):
        
        self.error = 0.00  
        '''loop through the output layer and find the error gradient'''
        for neuron in self.layers[-1].getneurons():
            self.error += math.fabs(self.targets.get(neuron.getindex()) - neuron.getoutputvalue())
            error = (self.targets.get(neuron.getindex()) - neuron.getoutputvalue())
            neuron.calculateoutputgradient(error)  
            
        self.error /= len(self.layers[-1].getneurons())
                
        ''' calculate the hidden layer gradients '''
        for index in range(len(self.layers)-2, 0, -1):
            nextlayer = self.layers[index + 1]
            for neuron in self.layers[index].getneurons():
                neuron.calculatehiddengradient(nextlayer)
                
        ''' update the input weights of all layers'''
        for index in range (len(self.layers) -1, 0, -1):
            prevlayer = self.layers[index - 1]
            for neuron in self.layers[index].getneurons():
                neuron.updateinputweights(prevlayer)
        
    
    def getresults(self, type):
        print(type)
        highvalue = None
        print("Current Session Target: ", self.target, "Average Error: ", self.error)
        for neuron in self.layers[-1].getneurons():
            print ("Output Neuron: ", neuron.getindex(), ":  ", "Value: ", neuron.getoutputvalue(), "Target: ", self.targets[neuron.getindex()])
            if highvalue == None:
                highvalue = neuron
            if neuron.getoutputvalue() > highvalue.getoutputvalue():
                highvalue = neuron
        print()
        print("Neuron", highvalue.getindex(), "is the most probable output")
        print()
    
    def getlayer(self, index):
        return self.layers[index]
    
    def settargets(self, target):
        for key in range(0, len(self.layers[-1].getneurons())):
            if(target == key):
                self.targets.__setitem__(key, 1.0)
                self.target = key
            else:
                self.targets.__setitem__(key, 0.0)
        
        
class Layer:
    
    layer = []
    
    def __init__(self): 
        self.layer = []
    
    def addneuron(self, neuron):
        self.layer.append(neuron)
    
    def getneuron(self, index):
        return self.layer[index]
    
    def getneurons(self):
        return self.layer
    