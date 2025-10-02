import numpy as np, os, time, random
from envs_repo.constructor import EnvConstructor
from models.constructor import ModelConstructor
from core.params import Parameters
import argparse, torch
from algos.erl_trainer import ERL_Trainer
from core import utils
from sbo import SBO

parser = argparse.ArgumentParser()

#######################  COMMANDLINE - ARGUMENTS ######################
#parser.add_argument('--env', type=str, help='Env Name',  default='HalfCheetah-v2')
parser.add_argument('--seed', type=int, help='Seed', default=11)
parser.add_argument('--savetag', type=str, help='#Tag to append to savefile',  default='')
parser.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)
parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=1)
parser.add_argument('--buffer', type=float, help='Buffer size in million',  default=1.0)
parser.add_argument('--frameskip', type=int, help='Frameskip',  default=1)

parser.add_argument('--hidden_size', type=int, help='#Hidden Layer size',  default=256)
parser.add_argument('--critic_lr', type=float, help='Critic learning rate?', default=3e-4)
parser.add_argument('--actor_lr', type=float, help='Actor learning rate?', default=1e-4)
parser.add_argument('--tau', type=float, help='Tau', default=1e-3)
parser.add_argument('--gamma', type=float, help='Discount Rate', default=0.99)
parser.add_argument('--alpha', type=float, help='Alpha for Entropy term ',  default=0.1)
parser.add_argument('--batchsize', type=int, help='Seed',  default=512)
parser.add_argument('--reward_scale', type=float, help='Reward Scaling Multiplier',  default=1.0)
parser.add_argument('--learning_start', type=int, help='Frames to wait before learning starts',  default=5000)

#ALGO SPECIFIC ARGS
parser.add_argument('--popsize', type=int, help='#Policies in the population',  default=10)
parser.add_argument('--rollsize', type=int, help='#Policies in rollout size',  default=1)
parser.add_argument('--gradperstep', type=float, help='#Gradient step per env step',  default=1.0)
parser.add_argument('--num_test', type=int, help='#Test envs to average on',  default=5)

#Figure out GPU to use [Default is 0]
os.environ['CUDA_VISIBLE_DEVICES']=str(vars(parser.parse_args())['gpu_id'])

#######################  Construct ARGS Class to hold all parameters ######################
args = Parameters(parser)

#Set seeds
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

################################## Find and Set MDP (environment constructor) ########################
env_constructor = []
env_constructor.append(EnvConstructor('HalfCheetah-v2', args.frameskip))
env_constructor.append(EnvConstructor('Ant-v2', args.frameskip))



#######################  Actor, Critic and ValueFunction Model Constructor ######################



model_constructor = []
model_constructor.append(ModelConstructor(env_constructor[0].state_dim, env_constructor[0].action_dim, args.hidden_size))
model_constructor.append(ModelConstructor(env_constructor[1].state_dim, env_constructor[1].action_dim, args.hidden_size))



ai = []
ai.append(ERL_Trainer(args, model_constructor[0], env_constructor[0]))
ai.append(ERL_Trainer(args, model_constructor[1], env_constructor[1]))



test_tracker = []
test_tracker.append(utils.Tracker(args.savefolder, ['score_'  + env_constructor[0].env_name + args.savetag], '.csv'))
test_tracker.append(utils.Tracker(args.savefolder, ['score_'  + env_constructor[1].env_name + args.savetag], '.csv'))




sbo = SBO()
sbo.doInitial()
print('seed',args.seed)

for gen in range(1, 1000000000):
    
    
    
    ai[0].train(gen,test_tracker[0])
    ai[1].train(gen,test_tracker[1])
    
    
        
    
    
    pop =[]
    pop.append(ai[0].population)    
    pop.append(ai[1].population)  
    
    sbo.setGAs(pop,env_constructor)
    sbo.doTransfer()
    
    if ai[0].total_frames < args.total_steps and ai[1].total_frames < args.total_steps:
        sbo.doEvaluation()
        sbo.updateSymbiosis()
        sbo.updateTransferRate()
        
   
    

    if ai[0].total_frames > args.total_steps and ai[1].total_frames > args.total_steps: 
        break


try:
    for p in ai[0].task_pipes: p[0].send('TERMINATE')
    for p in ai[0].test_task_pipes: p[0].send('TERMINATE')
    for p in ai[0].evo_task_pipes: p[0].send('TERMINATE')
    
    for p in ai[1].task_pipes: p[0].send('TERMINATE')
    for p in ai[1].test_task_pipes: p[0].send('TERMINATE')
    for p in ai[1].evo_task_pipes: p[0].send('TERMINATE')
    
    
except:
    None
           
           
        

        
        
        
        

