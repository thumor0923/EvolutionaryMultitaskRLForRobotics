import numpy as np, os, time, sys, random
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
import gym, torch
from core import replay_memory
from core import ddpg as ddpg
import argparse
import csv
from sbo import SBO



render = False
parser = argparse.ArgumentParser()
#parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
#env_tag = vars(parser.parse_args())['env']


class Parameters:
    def __init__(self, env_tag):

        #Number of Frames to Run
        if env_tag == 'Hopper-v2': self.num_frames = 1000000
        elif env_tag == 'Ant-v2': self.num_frames = 1000000
        elif env_tag == 'Walker2d-v2': self.num_frames = 1000000
        else: self.num_frames = 1000000

        #USE CUDA
        self.is_cuda = True; self.is_memory_cuda = True

        #Sunchronization Period
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.synch_period = 10
        else: self.synch_period = 10

        #DDPG params
        self.use_ln = True
        self.gamma = 0.99; self.tau = 1e-3
        self.seed = 11
        self.batch_size = 512
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        ###### NeuroEvolution Params ########
        #Num of trials
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 1
        elif env_tag == 'Walker2d-v2': self.num_evals = 1
        else: self.num_evals = 1

        #Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.2
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.2


        self.pop_size = 10
        self.crossover_prob = 0.15
        self.mutation_prob = 0.9

        #Save Results
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = 'R_ERL/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

class Agent:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.evolver = utils_ne.SSNE(self.args)

        #Init population
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.Actor(args))

        #Turn off gradients and put in eval mode
        for actor in self.pop: actor.eval()

        #Init RL Agent
        self.rl_agent = ddpg.DDPG(args)
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
        self.ounoise = ddpg.OUNoise(args.action_dim)

        #Trackers
        self.num_games = 0; self.num_frames = 0; self.gen_frames = None

    def add_experience(self, state, action, next_state, reward, done):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.args.is_cuda: reward = reward.cuda()
        if self.args.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.args.is_cuda: done = done.cuda()
        action = utils.to_tensor(action)
        if self.args.is_cuda: action = action.cuda()
        self.replay_buffer.push(state, action, next_state, reward, done)

    def evaluate(self, net, is_render, is_action_noise=False, store_transition=True):
        total_reward = 0.0

        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda: state = state.cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if render and is_render: self.env.render()
            action = net.forward(state)
            action.clamp(-1,1)
            action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()

            next_state, reward, done, info = self.env.step(action.flatten())  #Simulate one step in environment
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward

            if store_transition: self.add_experience(state, action, next_state, reward, done)
            state = next_state
        if store_transition: self.num_games += 1

        return total_reward

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        self.gen_frames = 0

        ####################### EVOLUTION #####################
        all_fitness = []
        #Evaluate genomes/individuals
        for net in self.pop:
            fitness = 0.0
            for eval in range(self.args.num_evals): fitness += self.evaluate(net, is_render=False, is_action_noise=False)
            all_fitness.append(fitness/self.args.num_evals)
        #print('all_fitness',all_fitness)
        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        #Validation test
        champ_index = all_fitness.index(max(all_fitness))
        test_score = 0.0
        for eval in range(5): test_score += self.evaluate(self.pop[champ_index], is_render=True, is_action_noise=False, store_transition=False)/5.0

        #NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)


        ####################### DDPG #########################
        #DDPG Experience Collection
        self.evaluate(self.rl_agent.actor, is_render=False, is_action_noise=True) #Train

        #DDPG learning step
        if len(self.replay_buffer) > self.args.batch_size * 5:
            for _ in range(int(self.gen_frames*self.args.frac_frames_train)):
                transitions = self.replay_buffer.sample(self.args.batch_size)
                batch = replay_memory.Transition(*zip(*transitions))
                self.rl_agent.update_parameters(batch)

            #Synch RL Agent to NE
            if self.num_games % self.args.synch_period == 0:
                self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index])
                self.evolver.rl_policy = worst_index
                #print('rl_agent:',worst_index)
                #print('Synch from RL --> Nevo')

        return best_train_fitness, test_score, elite_index

if __name__ == "__main__":
    env_name1 = 'HalfCheetah-v2'
    env_name2 = 'Ant-v2'
    
    parameters1 = Parameters(env_name1)  # Create the Parameters class
    parameters2 = Parameters(env_name2)

    tracker1 = utils.Tracker(parameters1, ['erl'+ env_name1], '_score1.csv')  # Initiate tracker
    tracker2 = utils.Tracker(parameters2, ['erl'+ env_name2], '_score2.csv')

    frame_tracker1 = utils.Tracker(parameters1, ['frame_erl'+ env_name1], '_score1.csv')  # Initiate tracker
    frame_tracker2 = utils.Tracker(parameters2, ['frame_erl'+ env_name2], '_score2.csv')

    time_tracker1 = utils.Tracker(parameters1, ['time_erl'+ env_name1], '_score1.csv')
    time_tracker2 = utils.Tracker(parameters2, ['time_erl'+ env_name2], '_score2.csv')

    #Create Env
    env1 = utils.NormalizedActions(gym.make( env_name1 ))
    parameters1.action_dim = env1.action_space.shape[0]
    parameters1.state_dim = env1.observation_space.shape[0]
    
    env2 = utils.NormalizedActions(gym.make( env_name2 ))
    parameters2.action_dim = env2.action_space.shape[0]
    parameters2.state_dim = env2.observation_space.shape[0]

    print('seed',parameters1.seed)
    #Seed
    env1.seed(parameters1.seed);
    torch.manual_seed(parameters1.seed); np.random.seed(parameters1.seed); random.seed(parameters1.seed)

    env2.seed(parameters2.seed);
    torch.manual_seed(parameters2.seed); np.random.seed(parameters2.seed); random.seed(parameters2.seed)

    #Create Agent
    agent1 = Agent(parameters1, env1)
    print('Running', env_name1, ' State_dim:', parameters1.state_dim, ' Action_dim:', parameters1.action_dim)

    agent2 = Agent(parameters2, env2)
    print('Running2', env_name2, ' State_dim:', parameters2.state_dim, ' Action_dim:', parameters2.action_dim)

    next_save1 = 100; next_save2 = 100; time_start = time.time()
    
    sbo = SBO()
    sbo.doInitial()
    
    env=[]
    env.append(env1)
    env.append(env2)
    
    state = []
    state.append(parameters1.state_dim)
    state.append(parameters2.state_dim)
    
    action = []
    action.append(parameters1.action_dim)
    action.append(parameters2.action_dim)

    
    
    
    while agent1.num_frames <= parameters1.num_frames or agent2.num_frames <= parameters2.num_frames:
        

            best_train_fitness1, erl_score1, elite_index1 = agent1.train()
            print('#Frames:', agent1.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness1 if best_train_fitness1 != None else None, ' Test_Score:','%.2f'%erl_score1 if erl_score1 != None else None, ' Avg:','%.2f'%tracker1.all_tracker[0][1], 'ENV '+ env_name1)
            '''
            print('RL Selection Rate: Elite/Selected/Discarded', '%.2f'%(agent1.evolver.selection_stats['elite']/agent1.evolver.selection_stats['total']),
                                                             '%.2f' % (agent1.evolver.selection_stats['selected'] / agent1.evolver.selection_stats['total']),
                                                            '%.2f' % (agent1.evolver.selection_stats['discarded'] / agent1.evolver.selection_stats['total']))
            '''
            print()
            tracker1.update([erl_score1], agent1.num_games)
            frame_tracker1.update([erl_score1], agent1.num_frames)
            time_tracker1.update([erl_score1], time.time()-time_start)
            
            filename = 'result/'+env_name1 + '_SBO_' + str(parameters1.seed) + '.csv'
            
            with open(filename,'a+') as f:
                writer = csv.writer(f)
                writer.writerow([agent1.num_frames,'%.3f'%erl_score1])
            




            best_train_fitness2, erl_score2, elite_index2 = agent2.train()
            print('#Frames:', agent2.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness2 if best_train_fitness2 != None else None, ' Test_Score:','%.2f'%erl_score2 if erl_score2 != None else None, ' Avg:','%.2f'%tracker2.all_tracker[0][1], 'ENV '+ env_name2)
            '''
            print('RL Selection Rate: Elite/Selected/Discarded', '%.2f'%(agent2.evolver.selection_stats['elite']/agent2.evolver.selection_stats['total']),
                                                             '%.2f' % (agent2.evolver.selection_stats['selected'] / agent2.evolver.selection_stats['total']),
                                                              '%.2f' % (agent2.evolver.selection_stats['discarded'] / agent2.evolver.selection_stats['total']))
            '''
            print()
            tracker2.update([erl_score2], agent2.num_games)
            frame_tracker2.update([erl_score2], agent2.num_frames)
            time_tracker2.update([erl_score2], time.time()-time_start)
            
            
            filename2 ='result/'+env_name2 + '_SBO_' + str(parameters2.seed) + '.csv'
            
            with open(filename2,'a+') as f2:
                writer = csv.writer(f2)
                writer.writerow([agent2.num_frames,'%.3f'%erl_score2])
                
                
                
            population =[]
            population.append(agent1.pop)    
            population.append(agent2.pop)  
            
            sbo.setGAs(population,env,state,action)
            sbo.doTransfer()
            
            if agent1.num_frames < parameters1.num_frames and agent2.num_frames < parameters2.num_frames:
                sbo.doEvaluation()
                sbo.updateSymbiosis()
                sbo.updateTransferRate()
        
            if agent1.num_frames > parameters1.num_frames and agent2.num_frames > parameters2.num_frames:
                break
        


       












