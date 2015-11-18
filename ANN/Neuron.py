'''
Created on Nov 8, 2015

@author: Michael O'Connor
'''
import random
import math


class Neuron(object):
    
    
    stepsize = .2
    
    def __init__(self, index, outputs = None):
        
        
        self.index = index
        
        self.outputvalue = 0.00
        
        self.gradient = 0.00
        
        self.inputsum = 0.00
        
        if outputs != None:
            self.outputweights = []
            for c in range(0, outputs):
                self.outputweights.append(Connection())
        
    
    def setoutputvalue(self, value):
        self.outputvalue = value
        
    def getoutputvalue(self):
        return self.outputvalue
    
    def getindex(self):
        return self.index
    
    def getgradient(self):
        return self.gradient
    
    def getoutputweights(self):
        return self.outputweights
    
    ''' Back propogate the deltas from the next layer to the hidden layer'''
    def sumweightsbydelta(self, nextlayer):
        
        sum = 0.0
        
        for neuron in nextlayer.getneurons():
            sum += self.outputweights[neuron.getindex()].getweight() * neuron.getgradient()
            
        return sum
    
    def updateinputweights(self, prevlayer):
        
        for neuron in prevlayer.getneurons():
               
            weight = neuron.getoutputweights()[self.index].getweight() + self.stepsize * neuron.getoutputvalue() * self.gradient 
             
            neuron.getoutputweights()[self.index].setweight(weight)         
    
    def calculatehiddengradient(self, nextlayer):
        self.gradient = self.sumweightsbydelta(nextlayer) * self.activationderivative()
    
    def calculateoutputgradient(self, error):
        self.gradient = error * self.activationderivative()
        
    
    ''' feeds the values from the previous layer to this neuron'''    
    def feedforward(self, prevlayer):
        self.inputsum = 0.0
        for neuron in prevlayer.getneurons():
            self.inputsum += neuron.getoutputvalue() * neuron.getoutputweights()[self.index].getweight()
        self.activationfunction()
    
    ''' apply the sigmoid function (logistic regression) to the sum of the previous layer'''
    def activationfunction(self):
        self.outputvalue = 1/(1 + math.exp(-self.inputsum))
        
    def activationderivative(self):
        '''value = (1/(1 + math.exp(-self.inputsum))) * (1 - ((1/1 + math.exp(-self.inputsum))))'''
        value = (math.exp(-self.inputsum)/(1 + math.exp(-self.inputsum))**2)
        return value
    
class Connection:
    
    def __init__(self):
        ''' Assign a random weight '''
        self.weight = round(random.uniform(0, .016), 2)
    
    def getweight(self):
        return self.weight
    
    def setweight(self, weight):
        self.weight = weight
    
    def addweight(self, deltaweight):
        self.weight += deltaweight
    