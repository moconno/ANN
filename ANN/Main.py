'''
Created on Nov 8, 2015

@author: Michael O'Connor
'''

from ANN.Net import Net

if __name__ == '__main__':
    
    ''' The number of layers derived from input, hidden, and output nodes '''
    layers = []
    
    
    ''' constant value of inputs - represents a layer'''
    INPUT_NODES = 64
    
    layers.append(INPUT_NODES)
    
    ''' constant value of hidden nodes - represents a layer'''
    HIDDEN_NODES = 64
    
    layers.append(HIDDEN_NODES)
    
    ''' constant value of output nodes - represents a layer '''
    OUTPUT_NODES = 10
    
    layers.append(OUTPUT_NODES)
    
    ''' A net with x layers '''
    net = Net(layers)
    
    train_file = open('optdigits_train.txt', 'r')
    test_file = open('optdigits_test.txt', 'r')
    
    '''Training'''
    for i in range(0,1000):
        line = train_file.readline()
        line = line.split(',')
        count = 0
        while(count < INPUT_NODES):
            neuron = net.getlayer(0).getneuron(count)
            neuron.setoutputvalue(int(line[count]))
            count += 1
        '''set the target vector'''         
        net.settargets(int(line[count]))
            
        net.feedforward() 
          
        net.backpropogate()
        
        net.getresults("Training Data")
        
    '''Testing'''
    for line in test_file:
        line = line.split(',')
        count = 0
        while(count < INPUT_NODES):
            neuron = net.getlayer(0).getneuron(count)
            neuron.setoutputvalue(int(line[count]))
            count += 1
            
        '''set the target vector'''         
        net.settargets(int(line[count]))
            
        net.feedforward() 
        
        net.getresults("Test Data")
        
    train_file.close()
            
       
        
        
            
        
        

