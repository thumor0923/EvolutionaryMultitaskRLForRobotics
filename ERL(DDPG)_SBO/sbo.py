#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 23:45:57 2022

@author: talli
"""
import numpy as np
import random
from ga import GA
from core import mod_utils as utils


class SBO:

    def __init__(self):
        """
        A general Environment Constructor
        """
        self._task_size=2
        self._ga = []
        for i in range(self._task_size) :
            self._ga.append(0)

        self.index_rank=[]
        
        self.task_belong=[]
        
        self.task_belong_index=[]
       

        self._mutualism = np.zeros((self._task_size,self._task_size),dtype=int)
        self._neutralism = np.zeros((self._task_size,self._task_size),dtype=int)
        self._competition = np.zeros((self._task_size,self._task_size),dtype=int)
        self._commensalism = np.zeros((self._task_size,self._task_size),dtype=int)
        self._parasitism = np.zeros((self._task_size,self._task_size),dtype=int)
        self._amensalism = np.zeros((self._task_size,self._task_size),dtype=int)
        self._transfer_rate = np.zeros((self._task_size,self._task_size),dtype=float)
        
        
        self.beneficial_factor = 0.25
        self.harmful_factor = 0.5
        
        
    def setGAs(self , pop, env,state,action):
        self.task_belong.clear()
        self.task_belong_index.clear()
        
        self.pop = pop
        self.env = env
        self.state = state
        self.action = action
        
        

        for i in range(self._task_size) :
            belong=[]
            belong_index=[]
            
            self._ga[i] = GA(self.pop[i],self.env[i],self.state[i],self.action[i])
            
            for j in range(self._ga[i].getOffspringSize()):
                belong.append(i)
                belong_index.append(j)
                
            self.task_belong.append(belong)
            self.task_belong_index.append(belong_index)
            
    
    
    def doInitial(self):


        
        for i in range(self._task_size) :
            for j in range(self._task_size) :
                if i!=j :
                    self._mutualism[ i ][ j ] = 1
                    self._neutralism[ i ][ j ] = 1
                    self._competition[ i ][ j ] = 1
                    self._commensalism[ i ][ j ] = 1
                    self._parasitism[ i ][ j ] = 1
                    self._amensalism[ i ][ j ] = 1
                    
        self.updateTransferRate()
        
    
    def updateTransferRate(self):
        for i in range(self._task_size) :
            for j in range(self._task_size) :
                if i!=j :
                    pos = self._mutualism[ i ][ j ] + self._commensalism[ i ][ j ] + self._parasitism[ i ][ j ]
                    neg = self._amensalism[ i ][ j ] + self._competition[ i ][ j ]
                    neu = self._neutralism[ i ][ j ]
                    
                    self._transfer_rate[ i ][ j ] = float( pos ) / float( ( pos + neg + neu ) )
                    
                    if self._transfer_rate[ i ][ j ] >= 0.5 :
                        self._transfer_rate[ i ][ j ] = 0.5
            
                        
    def doTransfer(self):
        
        for task_i in range(self._task_size) :
            #print('before',self.task_belong[task_i])
            #print('before',self.task_belong_index[task_i])
            
            task_j = 0;
            rate = self._transfer_rate[ task_i ][ task_j ];
            for i in range(self._task_size) :
                if self._transfer_rate[ task_i ][ i ] > rate :
                    task_j = i
                    rate = self._transfer_rate[ task_i ][ task_j ]
                    
            
            #print(rate)      
            if random.random() < rate:
                amount = rate * self._ga[task_i].getOffspringSize()
                amount = int(amount)
                #print(amount)
                #print()
                
                for index_j in range(amount):
                    index_i = self._ga[ task_i ].getOffspringSize() - ( index_j + 1 )
                    self._ga[ task_i ].setOffspring( index_i, self._ga[ task_j ].getOffspring( index_j ) ,self._ga[ task_i ].state_size,self._ga[ task_j ].state_size,self._ga[ task_i ].action_size,self._ga[ task_j ].action_size)
                    
                    self.task_belong[task_i][index_i] = self.task_belong[task_j][index_j]
                    self.task_belong_index[task_i][index_i]=self.task_belong_index[task_j][index_j]
                    
            #print('after',self.task_belong[task_i])
            #print('after',self.task_belong_index[task_i])
  
           
    def doEvaluation(self):
        
        self.index_rank.clear()
        for i in range(self._task_size) :
            self.index_rank.append(self._ga[ i ].doEvaluation())
            #print('index_rank:',i,self.index_rank[i])
        #print()  

        
    def updateSymbiosis(self):
        for task_i in range(self._task_size) :
            offspring_size_i = self._ga[task_i].getOffspringSize()
            for index_i in range(offspring_size_i):
                if self.task_belong[task_i][index_i] != task_i:
                    task_j = self.task_belong[task_i][index_i]
                    offspring_size_j = self._ga[ task_j ].getOffspringSize()
                    for index_j in range(offspring_size_j):
                        if (self.task_belong[task_i][index_i]==self.task_belong[task_j][index_j] and 
                            self.task_belong_index[task_i][index_i] == self.task_belong_index[task_j][index_j]):
                            
                            self.updateRelation( task_i, index_i, task_j, index_j )
                            
                            
                    
                    
                
                
            
                    
    def updateRelation(self,task_i, index_i, task_j, index_j):
        
        Brank_i = (self.beneficial_factor * self._ga[ task_i ].getOffspringSize())-1;
        Hrank_i = self.harmful_factor * self._ga[ task_i ].getOffspringSize();

        Brank_j = (self.beneficial_factor *self. _ga[ task_j ].getOffspringSize())-1;
        Hrank_j = self.harmful_factor * self._ga[ task_j ].getOffspringSize();          
        
        rank_i = 0
        rank_j = 0
        
        for i in range(len(self.index_rank[task_i])):
            if self.index_rank[task_i][i] == index_i:
                rank_i = i
                
        for j in range(len(self.index_rank[task_j])):
            if self.index_rank[task_j][j] == index_j:
                rank_j = j
                
        
        #print('index_i',index_i,'rank_i',rank_i)
        #print('index_j',index_j,'rank_j',rank_j)
        #print()
        
        
        if rank_i <= Brank_i :
            if rank_j <= Brank_j:
                self._mutualism[ task_i ][ task_j ]+=1
            elif rank_j >= Hrank_j:
                self._parasitism[ task_i ][ task_j ]+=1
            else:
                self._commensalism[ task_i ][ task_j ]+=1
                   
        elif rank_i >= Hrank_i:
            if rank_j >= Hrank_j:
                self._competition[ task_i ][ task_j ]+=1
            elif rank_j > Brank_j and rank_j < Hrank_j:
                self._amensalism[ task_i ][ task_j ]+=1
                
        else:
            if ( rank_i > Brank_i and rank_i < Hrank_i ) and ( rank_j < Hrank_j and rank_j > Brank_j ):
                self._neutralism[ task_i ][ task_j ]+=1
                
                    



        
        
