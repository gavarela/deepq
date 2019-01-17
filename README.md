# deepq

Teaching an AI to play the game peg solitaire using deep Q reinforcement learning.

## Intro & Files

The game is played by jumping pieces over other pieces to an empty spot, thereby 'eating' the piece that was jumped over and removing it from the game. The aim is to 'eat' all but one piece from the board. More info here: https://www.wikiwand.com/en/Peg_solitaire

The name in the file, Remain One, is a direct translation from the Portuguese name of the game, as a joke.

The files are organised like this:
1. NeuralNet.py: implements a simple fully-connected neural network class, used for the deep learning aspect of the project;
2. RemainOne.py: implements the game. Running it runs an instance of the HumanGame class, which allows someone to play the game on the terminal using simple imputs to represent moves (e.g. 'f4 up'). It also defines a ComputerGame class, used for the deep Q learning program;
3. DeepQLearning.py: implements a deep Q learning player/agent and a class with a reinforcement learning routine;
4. RemainOneAI.py: the script that brings it all together and runs the program.

Note: not working yet.


## Reinforcement Learning and Deep Q Learning

Traditional supervised learning is useful for teaching AI to make decisions in scenarios where there is a correct decision that we already know, e.g. classifying images, estimating values. Getting AI to make decisions in a situation where there is no ready repository of 'correct' decisions to learn from needs a whole different learning paradigm. This is the case of learning how to play a game.

We'll depict the game we want to play as a black box that articulates interactions between an environment (for us, the game) and an actor (the player). The environment has a given state and the actor chooses an action based on the environment. This action both changes the state of the environment and elicits a reward that depends on both the action and the state. The game goes on for many such periods until it's either beat or crashes (the player lost).

So the actor should choose the actions that it thinks will elicit most reward in the long term, considering the future states resulting from the current action. To mimick human action, we have the actor value the present more than the future (i.e. discount the future). That is, if r(s, a) is the reward from acting on state s with action a, the actor wants to maximise

<a href="https://www.codecogs.com/eqnedit.php?latex=V(s_0)&space;=&space;\sum_{t&space;=&space;0}^T&space;\gamma^t&space;r(s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s_0)&space;=&space;\sum_{t&space;=&space;0}^T&space;\gamma^t&space;r(s_t,&space;a_t)" title="V(s_0) = \sum_{t = 0}^T \gamma^t r(s_t, a_t)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=s_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_0" title="s_0" /></a> is the initial state of the game, <a href="https://www.codecogs.com/eqnedit.php?latex=s_{t&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_{t&plus;1}" title="s_{t+1}" /></a> is the state that results from acting on state <a href="https://www.codecogs.com/eqnedit.php?latex=s_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_t" title="s_t" /></a> with action <a href="https://www.codecogs.com/eqnedit.php?latex=a_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_t" title="a_t" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> is a discount factor.

Instead of having the network predict the entire sequence of discounted rewards for each action in the current state and choosing the action with the highest discounted rewards, we define instead the Q-value of action a on state s as the maximum value you can obtain after acting on state s with a, i.e. you get today's reward and the rewards corresponding to acting optimally in the future, starting with next period's state, s'. I.e.:

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,&space;a)&space;=&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}Q(s',&space;a')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s,&space;a)&space;=&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}Q(s',&space;a')" title="Q(s, a) = r(s, a) + \gamma \max_{a'}Q(s', a')" /></a>

Notice that <a href="https://www.codecogs.com/eqnedit.php?latex=\max_{a}Q(s,&space;a)&space;=&space;V(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max_{a}Q(s,&space;a)&space;=&space;V(s)" title="\max_{a}Q(s, a) = V(s)" /></a>, which is what we wanted to maximise in the first place. We can thus pick actions by running the network on a given state and choosing the action with highest Q(s, a).

So we'll let our player play the game, with random behaviour at first so it can build a 'memory' of diversified past experience to learn from, and train a neural network (the actor's 'brain') on this memory of past states, actions and rewards. the network (Q-network) should predict the Q-value in a given state for each possible action, so the actor can just pick the action with highest predicted Q-value. After each turn, we retrain the network on the new memory example and the idea is that this iteration should make the network's estimate of Q(s, a) approach the true Q(s, a), leading to better and better decisions as the actor plays the game (learning).

## The Algorithm

1. Initialise the class instances:

   a) The game (an environment which provides a state, takes in an action and returns a reward);

   b) An actor with a neural network which can interact with the above game;
2. Run game. In each turn:
   
   a) Run a lottery to see if action will be random or if we'll allow the actor to pick the best action based on its network (run network on current state, pick action with highest predicted Q-value). This lottery will be initially biased towards random action and progressively loosened;
   
   b) Store old state, action, reward and new state (inc. if game ended) into the memory: (s, a, r, s', c);
   
   c) Train network on a single training example generated by:
      
      - running network on s' and on s;
       
      - update network's prediction on s using its prediction on s' and the observed reward of acting a on s:
         
         <a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,&space;a)&space;\to&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}&space;Q(s',&space;a')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s,&space;a)&space;\to&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}&space;Q(s',&space;a')" title="Q(s, a) \to r(s, a) + \gamma \max_{a'} Q(s', a')" /></a>
       
      - train network using s as the input and the above updated prediction as the output, forcing the network to continuously update its own predictions; 

3. At the end of the game, we train the network on larger sample of memory:
   
   a) Extract batch from memory
   
   b) Rowwise on memory batch, do the first two steps described in 2c above to generate a larger training sample
   
   c) Train network on this sample

4. Go back to step 2. Run game again a set number of times to allow the network to learn.

The result should be an actor whose network properly predicts Q-values given states as inputs and can thus play the game by maximising its expected Q-value.

