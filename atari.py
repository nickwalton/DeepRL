# Import the gym module
import gym
import time
import keras
import random
import pdb
import numpy as np

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))
    
def transform_reward(reward):
    return np.sign(reward)
    

class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
            
    def sample_batch(self, n):
    
        sample_batch = []
        for _ in range(n):
            batch_ind = random.randint(self.start, self.end+1)
            sample_batch.append(self[batch_ind])
        return sample_batch
        
    def as_list(self):
        return self[self.start:self.end]
	
    def get_state(self):
        return self[self.end-4:self.end]
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            return self.data[(self.start + key) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
	    
    
def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
    """Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal

    """
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    pdb.set_trace()
    
    model.fit(
        [start_states, actions], actions * Q_values[:, None],
        nb_epoch=1, batch_size=len(start_states), verbose=0
    )
    
def get_epsilon_for_iteration(iteration):
	return 0.99 - (0.99/1000000)*iteration
    
def q_iteration(env, model, state, iteration, memory):

    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)

    # Choose the action 
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    new_frame, reward, is_done, _ = env.step(action)
    memory.append([state, action, new_frame, reward, is_done])

    # Sample and fit
    batch = np.array(memory.sample_batch(32))
    fit_batch(model, 1-epsilon, batch[:,0],batch[:,1],batch[:,3],batch[:,2],batch[:,4]) 
    
    return new_frame, reward, is_done, _   
    
def atari_model(n_actions):
    # We assume a theano backend here, so the "channels" are first.
    batch_size = 32
    ATARI_SHAPE = (batch_size, 105, 80, 3)

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    
    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.convolutional.Conv2D(16, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.convolutional.Conv2D(32, (4, 4), strides=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.multiply([output, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    
    return model


def train():

    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    model = atari_model(env.action_space.n)
    memory = RingBuf(1000000)
    n_train_iterations = 20000
    n_play_iterations = 100
    state = RingBuf(4)

    # Set up memory
    while(len(memory) < 1000):
        env.reset()
        # Set up state
        for _ in range(4):
            frame, reward, is_done, _ = env.step(env.action_space.sample())
            state.append(preprocess(frame))
            
        while not is_done:
            action = env.action_space.sample()
            frame, reward, is_done, _ = env.step(action)
            frame = state.append(preprocess(frame))
            memory.append([state.as_list(), action, frame, reward, is_done])
            state.append(frame)
        
        print("Length of memory is ", len(memory))

    for i in range(n_train_iterations):
        
        frame = env.reset()
        frame = state.append(preprocess(frame))
        state.append(frame)
        is_done = False
        
        while not is_done:

            new_frame, reward, is_done, _ = q_iteration(env, model, state.as_list(), i, memory)
            new_frame = state.append(preprocess(frame))
            		
            print("Train iterations is ", i)		

    for j in range(n_play_iterations):	

        # Reset it, returns the starting frame
        frame = env.reset()
        # Render
        env.render()

        is_done = False
        while not is_done:
          # Perform a random action, returns the new frame, reward and whether the game is over
          action = action = choose_best_action(model, state.as_list)
          frame, reward, is_done, _ = env.step(action)
          # Render
          env.render()
          time.sleep(0.1)
		  
		  
		  
if __name__ == '__main__':
	train()
