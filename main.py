import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import copy

sample_gridworld_map_0 = pd.read_csv ("/Users/chikhoado/Desktop/PROJECTS/Gridworld-10/Gridworld-10_Dataset/sample_gridworld_map_0.csv")
sample_gridworld_map_1 = pd.read_csv ("/Users/chikhoado/Desktop/PROJECTS/Gridworld-10/Gridworld-10_Dataset/sample_gridworld_map_1.csv")
sample_gridworld_map_2 = pd.read_csv ("/Users/chikhoado/Desktop/PROJECTS/Gridworld-10/Gridworld-10_Dataset/sample_gridworld_map_2.csv")

eval_challenge = pd.read_csv ("/Users/chikhoado/Desktop/PROJECTS/Gridworld-10/Gridworld-10_Dataset/eval_challenge.csv")
eval_solution = pd.read_csv ("/Users/chikhoado/Desktop/PROJECTS/Gridworld-10/Gridworld-10_Dataset/eval_solution.csv")
train_dataset = pd.read_csv ("/Users/chikhoado/Desktop/PROJECTS/Gridworld-10/Gridworld-10_Dataset/train.csv")

def check_and_clean_dataset ():

    features = ["state", "action", "reward", "next_state", "done"]
    dataset = [eval_challenge, eval_solution, train_dataset]
    name_dataset = ["eval_challenge.csv", "eval_solution.csv", "train_dataset.csv"]

    for index, data in enumerate (dataset):
        print (f"\n-------------------- {name_dataset[index]} --------------------")
        print ("\n+ Missing Values:")
        print (data.isnull ().sum (), "\n")
        
        for fea in features:
            print (f"+ {fea}: ", data[fea].unique (), data[fea].dtype, end = '\n\n')

def pre_processing (csv_file):

    csv_file['done'] = csv_file['done'].astype (np.int64)
    
    state = torch.tensor (csv_file['state'].values, dtype = torch.long)
    action = torch.tensor (csv_file['action'].values, dtype = torch.long)
    reward = torch.tensor (csv_file['reward'].values, dtype = torch.float32)
    next_state = torch.tensor (csv_file['next_state'].values, dtype = torch.long)
    done = torch.tensor (csv_file['done'].values, dtype = torch.float32)

    state = nn.functional.one_hot (state, num_classes = 100).float ()
    next_state = nn.functional.one_hot (next_state, num_classes = 100).float ()

    return state, action, reward, next_state, done

def initialization ():

    input_neuron = 100
    output_neuron = len (train_dataset['action'].unique ())
    Deep_Q_Learning = Deep_Reinforcement_Learning (input_neuron, output_neuron)
    Deep_Double_Q_Learning_1 = Deep_Reinforcement_Learning (input_neuron, output_neuron)
    Deep_Double_Q_Learning_2 = Deep_Reinforcement_Learning (input_neuron, output_neuron)
    Deep_Expected_SARSA = Deep_Reinforcement_Learning (input_neuron, output_neuron)

    return Deep_Q_Learning, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, Deep_Expected_SARSA

def training (check, model_1, model_2, state, action, reward, next_state, done,):
    
    a = 0.005
    b = 0.9
    c = 0.999

    if check == 1 or check == 3:
        target_model = copy.deepcopy (model_1)
        loss_func = nn.HuberLoss ()
        optimizer_1 = optim.Adam (params = model_1.parameters (), lr = a, betas = (b, c), amsgrad = True)
    else:
        target_model_1 = copy.deepcopy (model_1)
        target_model_2 = copy.deepcopy (model_2)
        optimizer_1 = optim.Adam (params = model_1.parameters (), lr = a, betas = (b, c), amsgrad = True)
        optimizer_2 = optim.Adam (params = model_2.parameters (), lr = a, betas = (b, c), amsgrad = True)

    loss_func = nn.HuberLoss ()
    few_batches = 100
    batch_size = 128
    probability_select_each_action = 0.25
    gamma = 0.99
    n_times = np.int64 (train_dataset.shape[0] / batch_size)

    for idx in range (n_times):

        index_shuffled = np.random.choice (train_dataset.index, size = batch_size, replace = False)

        st = state[index_shuffled]
        ac = action[index_shuffled].reshape (batch_size, 1)
        re = reward[index_shuffled]
        next_st = next_state[index_shuffled]
        do = done[index_shuffled].reshape (batch_size)

        if check == 1 or check == 3:
            current_Q_value = model_1 (st)
        else:
            model_choice = np.random.choice ([1, 2], size = 2, replace = False)
            if model_choice[0] == 1:
                current_Q_value = model_1 (st)
            else:
                current_Q_value = model_2 (st)

        current_Q_value = current_Q_value.gather (dim = 1, index = ac).reshape (batch_size)

        if idx % few_batches == 0:
            if check == 1 or check == 3:
                target_model.load_state_dict (model_1.state_dict ())
            else:
                target_model_1.load_state_dict (model_1.state_dict ())
                target_model_2.load_state_dict (model_2.state_dict ())

        with torch.no_grad ():
            if check == 1:
                target_Q_value = target_model (next_st)
                target_Q_value = torch.max (target_Q_value, dim = 1)[0]

            elif check == 2:
                if model_choice[0] == 1:
                    temp = target_model_1 (next_st)
                    target_Q_value = target_model_2 (next_st)
                else:
                    temp = target_model_2 (next_st)
                    target_Q_value = target_model_1 (next_st)

                index_target_Q_value = torch.max (temp, dim = 1)[1].reshape (batch_size, 1)
                target_Q_value = target_Q_value.gather (dim = 1, index = index_target_Q_value)
                target_Q_value = target_Q_value.reshape (batch_size)

            else:
                target_Q_value = target_model (next_st)
                target_Q_value = torch.mean (target_Q_value, dim = 1)
                target_Q_value *= probability_select_each_action

            target_Q_value *= (1 - do) * gamma
            target_Q_value += re

        loss_memory = loss_func (current_Q_value, target_Q_value)
        
        if check == 1 or check == 3 or model_choice[0] == 1:
            optimizer_1.zero_grad ()
            loss_memory.backward ()
            optimizer_1.step ()
        else:
            optimizer_2.zero_grad ()
            loss_memory.backward ()
            optimizer_2.step ()

def testing (model_1, model_2, state_test):

    with torch.no_grad ():
        if model_2 == None:
            Q_value = model_1 (state_test)
        else:
            Q_value_1 = model_1 (state_test)
            Q_value_2 = model_2 (state_test)
            Q_value = (Q_value_1 + Q_value_2) / 2

    index_best_action = torch.max (Q_value, dim = 1)[1]
    return index_best_action

def evaluation (index_best_action, action_result):

    accuracy = 0
    for i in range (index_best_action.shape[0]):
        predict = index_best_action[i]
        target = action_result[i]
        
        if predict == target:
            accuracy += 1

    return "{:.2f}".format (accuracy / index_best_action.shape[0])

def experiment (Deep_Q_Learning, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, Deep_Expected_SARSA, state_test, action_result):

    index_best_action_1 = testing (Deep_Q_Learning, None, state_test)
    index_best_action_2 = testing (Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, state_test)
    index_best_action_3 = testing (Deep_Expected_SARSA, None, state_test)

    accuracy_model_1 = evaluation (index_best_action_1, action_result)
    accuracy_model_2 = evaluation (index_best_action_2, action_result)
    accuracy_model_3 = evaluation (index_best_action_3, action_result)

    return [accuracy_model_1, accuracy_model_2, accuracy_model_3]

def cumulative (reward_track):

    for i in range (1, len (reward_track)):
        reward_track[i] += reward_track[i - 1]
        
    return reward_track

def check_valid_next_state (mapp, action, current_state, obstacle, current_gamma, decay_gamma, reward_track):

    for index in action[0]:
        
        temp = current_state
        if index == 0:
            temp -= 10
        elif index == 1:
            temp += 10
        elif index == 2:
            temp -= 1
        else: 
            temp += 1

        if mapp == 0:
            if temp < 0 or temp > 99:
                reward_track.append (-0.3 * current_gamma)
            elif temp in obstacle:
                reward_track.append (-1.3 * current_gamma)
                return temp 
            elif temp != 99:
                reward_track.append (0.4 * current_gamma)
                return temp
            else:
                reward_track.append (20 * current_gamma)
                return temp
        
        elif mapp == 1:
            if temp < 0 or temp > 99:
                reward_track.append (-1.3 * current_gamma)
            elif temp in obstacle:
                reward_track.append (-3 * current_gamma)
                return temp
            elif temp != 99:
                reward_track.append (-0.6 * current_gamma)
                return temp
            else:
                reward_track.append (20 * current_gamma)
                return temp

        else:
            if temp < 0 or temp > 99:
                reward_track.append (-1.3 * current_gamma)
            elif temp in obstacle:
                for i in range (len (obstacle)):
                    if obstacle[i] == temp:
                        if i <= int (len (obstacle) / 2):
                            reward_track.append (-2.0 * current_gamma)
                        else:
                            reward_track.append (-4.5 * current_gamma)
                        return temp
            elif temp != 99:
                reward_track.append (-0.3 * current_gamma)
                return temp
            else:
                reward_track.append (20 * current_gamma)
                return temp

        current_gamma *= decay_gamma

    return None 

def total_return (mapp, model_1, model_2, obstacle):

    current_state = 0
    current_gamma = 1
    decay_gamma = 0.99
    reward_track = []

    with torch.no_grad ():

        while current_state != 99:
            temp = nn.functional.one_hot (torch.tensor (current_state), num_classes = 100).float ().reshape (1, 100)

            if model_2 == None:
                Q_value = model_1 (temp)
            else:
                Q_value_1 = model_1 (temp)
                Q_value_2 = model_2 (temp)
                Q_value = (Q_value_1 + Q_value_2) / 2

            action = torch.sort (Q_value, descending = True)[1]
            current_state = check_valid_next_state (mapp, action, current_state, obstacle, current_gamma, decay_gamma, reward_track)
            
            if current_state == None:
                return cumulative (reward_track), "\n~~~ Unsuccessfully getting terminal ! ~~~"
    
    return cumulative (reward_track), "\n~~~ Successfully reaching terminal ! ~~~"

def expected_cumulative_return (Deep_Q_Learning, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, Deep_Expected_SARSA):

    map_0_obstacle = torch.tensor ([11, 14, 20, 22, 23, 24, 25, 28, 31, 38, 47, 64, 82, 85, 86])
    map_1_obstacle = torch.tensor ([4, 7, 8, 14, 27, 30, 43, 45, 56, 62, 64, 75, 82, 90, 95])
    map_2_obstacle = torch.tensor ([9, 15, 18, 19, 28, 37, 42, 44, 46, 53, 60, 73, 75, 89, 94])

    cumulative_return_model_1_map_0 = total_return (0, Deep_Q_Learning, None, map_0_obstacle)
    cumulative_return_model_2_map_0 = total_return (0, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, map_0_obstacle)
    cumulative_return_model_3_map_0 = total_return (0, Deep_Expected_SARSA , None, map_0_obstacle)
    map_0 = [cumulative_return_model_1_map_0, cumulative_return_model_2_map_0, cumulative_return_model_3_map_0]

    cumulative_return_model_1_map_1 = total_return (1, Deep_Q_Learning, None, map_1_obstacle)
    cumulative_return_model_2_map_1 = total_return (1, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, map_1_obstacle)
    cumulative_return_model_3_map_1 = total_return (1, Deep_Expected_SARSA , None, map_1_obstacle)
    map_1 = [cumulative_return_model_1_map_1, cumulative_return_model_2_map_1, cumulative_return_model_3_map_1]

    cumulative_return_model_1_map_2 = total_return (2, Deep_Q_Learning, None, map_2_obstacle)
    cumulative_return_model_2_map_2 = total_return (2, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, map_2_obstacle)
    cumulative_return_model_3_map_2 = total_return (2, Deep_Expected_SARSA , None, map_2_obstacle)
    map_2 = [cumulative_return_model_1_map_2, cumulative_return_model_2_map_2, cumulative_return_model_3_map_2]

    return [map_0, map_1, map_2]

def plotting (expected_return, accuracy):

    expected_return_map_0 = expected_return[0]
    expected_return_map_1 = expected_return[1]
    expected_return_map_2 = expected_return[2]
    accuracy = [float (i) * 100 for i in accuracy]

    fig, ((graph1, graph2), (graph3, graph4)) = plt.subplots (2, 2, figsize = (20, 10))
    fig.suptitle ("Comparison of 3 DRLs")
    
    graph1.plot (expected_return_map_0[0][0], label = 'Deep Q-learning Network', color = 'blue', marker = '*')
    graph1.plot (expected_return_map_0[1][0], label = 'Deep Double-Q-learning Network', color = 'red', marker = '^')
    graph1.plot (expected_return_map_0[2][0], label = 'Deep Expected-SARSA Network', color = 'green', marker = '+')
    graph1.set_title ("Expected Cumulative Return in Gridworld Map 0")
    graph1.set_xlabel ("Number of Steps (s)")
    graph1.set_ylabel ("Current Cumulative Return (unit)")
    graph1.legend ()
    graph1.grid (True)

    graph2.plot (expected_return_map_1[0][0], label = 'Deep Q-learning Network', color = 'blue', marker = '*')
    graph2.plot (expected_return_map_1[1][0], label = 'Deep Double-Q-learning Network', color = 'red', marker = '^')
    graph2.plot (expected_return_map_1[2][0], label = 'Deep Expected-SARSA Network', color = 'green', marker = '+')
    graph2.set_title ("Expected Cumulative Return in Gridworld Map 1")
    graph2.set_xlabel ("Number of Steps (s)")
    graph2.set_ylabel ("Current Cumulative Return (unit)")
    graph2.legend ()
    graph2.grid (True)
    
    graph3.plot (expected_return_map_2[0][0], label = 'Deep Q-learning Network', color = 'blue', marker = '*')
    graph3.plot (expected_return_map_2[1][0], label = 'Deep Double-Q-learning Network', color = 'red', marker = '^')
    graph3.plot (expected_return_map_2[2][0], label = 'Deep Expected-SARSA Network', color = 'green', marker = '+')
    graph3.set_title ("Expected Cumulative Return in Gridworld Map 2")
    graph3.set_xlabel ("Number of Steps (s)")
    graph3.set_ylabel ("Current Cumulative Return (unit)")
    graph3.legend ()
    graph3.grid (True)
    
    models = ['DQN', 'DDQN', 'DESN']
    color = ['blue', 'red', 'green']
    graph4.bar (models, accuracy, color = color)
    graph4.set_title ("Accuracy in Evaluation Challenge")
    graph4.set_ylabel ("Accuracy (%)")

    plt.tight_layout ()
    plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.4, hspace = 0.4)
    plt.savefig ("Comparison_of_3_DRLs.png")
    plt.show ()


class Deep_Reinforcement_Learning (nn.Module):
    def __init__ (self, input_size, output_size):
        super ().__init__ ()

        self.transition_1 = nn.Linear (input_size, 128)
        self.transition_2 = nn.Linear (128, 64)
        self.transition_3 = nn.Linear (64, output_size)
        self.acti_func = nn.ReLU ()

    def forward (self, X):

        X = self.acti_func (self.transition_1 (X))
        X = self.acti_func (self.transition_2 (X))
        X = self.transition_3 (X)

        return X


if __name__ == '__main__':

    check_and_clean_dataset ()
    state, action, reward, next_state, done = pre_processing (train_dataset)
    state_test, action_test, reward_test, next_state_test, done_test = pre_processing (eval_challenge)
    state_result, action_result, reward_result, next_state_result, done_result = pre_processing (eval_solution)

    Deep_Q_Learning, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, Deep_Expected_SARSA = initialization ()
    training (1, Deep_Q_Learning, None, state, action, reward, next_state, done)
    training (2, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, state, action, reward, next_state, done)
    training (3, Deep_Expected_SARSA, None, state, action, reward, next_state, done)

    accuracy = experiment (Deep_Q_Learning, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, Deep_Expected_SARSA, state_test, action_result)
    expected_return = expected_cumulative_return (Deep_Q_Learning, Deep_Double_Q_Learning_1, Deep_Double_Q_Learning_2, Deep_Expected_SARSA)
    plotting (expected_return, accuracy)





# cd "/Users/chikhoado/Desktop/PROJECTS/Gridworld-10"
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pandas matplotlib scikit-learn torch torchvision torchaudio
# python "/Users/chikhoado/Desktop/PROJECTS/Gridworld-10/main.py"