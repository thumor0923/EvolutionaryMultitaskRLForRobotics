#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 23:45:57 2022

@author: talli
"""
import numpy as np
from core import utils


class GA:

    def __init__(self,pop,env_constructor):

        
        
        self.env_constructor = env_constructor
        self.state_size = env_constructor.state_dim
        self.action_size = env_constructor.action_dim
        
        self._offspring = pop
        

    
        
    def getOffspringSize(self):
        return len(self._offspring)
    
    def getOffspring(self,index):
        return (self._offspring[ index ])
    
    def setOffspring(self,index,individual,state_size1,state_size2,action_size1,action_size2):
        
       hard_update(self._offspring[index],individual,state_size1,state_size2,action_size1,action_size2)
       
       

        
    def showOffsprings(self):
        for i in range(self.getOffspringSize()):
            print(self._offspring[i])
            
            
            
    
            
    def doEvaluation(self): 
        fitness=[]
        for i in range(self.getOffspringSize()):
            fitness.append(self.evaluateFitness(self._offspring[i]))
        #print(fitness)    
        index_rank = sorted(range(len(fitness)), key=fitness.__getitem__)
        index_rank.reverse()
        
        return index_rank
    
        
        
    def evaluateFitness(self,net):
        env = self.env_constructor.make_env()
        
        
        fitness = 0.0
        state = env.reset()
        state = utils.to_tensor(state)
        while True:# unless done
            action = net.clean_action(state)
            action = utils.to_numpy(action)
            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
            next_state = utils.to_tensor(next_state)
            fitness += reward
            state = next_state
            if done:
                break
        return  fitness
        
        
    
        
def hard_update(target, source , state_size1 , state_size2 , action_size1 , action_size2):
    """Hard update (clone) from target network to source

        Parameters:
              target (object): A pytorch model
              source (object): A pytorch model

        Returns:
            None
    """
    #print('state_size1',state_size1 , 'state_size2',state_size2, 'action_size1',action_size1 , 'action_size2',action_size2)
        
    count=0  
    for target_param, param in zip(target.parameters(), source.parameters()):

        count+=1
        if(count==1):
            if(state_size1 > state_size2):
                for i in range(256):
                    for j in range(state_size2):
                        target_param[i][j].data.copy_(param.data[i][j])
            if(state_size1 < state_size2):
                for i in range(256):
                    for j in range(state_size1):
                        target_param[i][j].data.copy_(param.data[i][j])
            if(state_size1 == state_size2):
                target_param.data.copy_(param.data)
                        
        if(count==2 or count==3 or count==4):
            target_param.data.copy_(param.data)
            
        if(count==5 or count==7 ):
            if(action_size1 > action_size2):
                for i in range(action_size2):
                    for j in range(256):
                        target_param[i][j].data.copy_(param.data[i][j])
            if(action_size1 < action_size2):
                for i in range(action_size1):
                    for j in range(256):
                        target_param[i][j].data.copy_(param.data[i][j])
            if(action_size1 == action_size2):
                target_param.data.copy_(param.data)
                
        if(count==6 or count==8):
            if(action_size1 > action_size2):
                for i in range(action_size2):
                    target_param[i].data.copy_(param.data[i])
            if(action_size1 < action_size2):
                for i in range(action_size1):
                    target_param[i].data.copy_(param.data[i])
            if(action_size1 == action_size2):
                target_param.data.copy_(param.data)

                        
                    
    



        
        
