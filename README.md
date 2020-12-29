# deepq

Teaching an AI to play the game peg solitaire using deep Q reinforcement learning. 

The game is simple. There is a board with spots arranged in a large cross. At the beginning, every spot but one has a marble in it. The aim is to remove all but one marble from the board. This is done by moving marbles _over_ each other, thereby removing the one that was jumped over. More info here: https://www.wikiwand.com/en/Peg_solitaire.

Here's a sneak peak of a trained AI playing a smaller version of the game, for an idea of how the game looks:

![Alt Text](https://github.com/gavarela/deepq/blob/master/images/Solved%20MRO2.gif)

If you want to play the game, I implemented it in pygame and you can play it by cloning the repository locally and running the file `play/ROGame.py`.

The name in the files, Remain One, is a direct translation from the Portuguese name of the game, which started as a joke and now would be kind of impractical to change so it stays.

(I'm cleaning up code and running the learning routine on the full-sized version of the game. Expect full updates before the end of 2020, likely around Christmas.)


## Reinforcement Learning and Deep Q Learning

Traditional supervised learning is useful for teaching AI to make decisions in scenarios where there is a correct decision we already know and want to emulate, e.g. classifying images, estimating values. Getting AI to make decisions in a situation where there is no ready repository of known decisions to learn from needs a whole different learning paradigm. This is the case of learning how to play a game.

We'll depict the game we want to play as a black box that articulates interactions between an environment (the game) and an actor (the player). The environment has a given state (in our case, the distribution of marbles on the board) and the actor chooses an action (a move) based on this state. This action both changes the state of the environment (removes a marble and changes location of another) and, for the purposes of the algorithm, elicits a reward. We code this reward so that it depends on both the action and the state. The same action can be worth more in some states than others. In our case, the reward is just inversely proportional to the number of marbles on the board. That way, removing a marble is _better_ later in the game, when there's fewer remaining.

The game goes on for many such state-action-new state periods until it's either beat or it crashes (the player lost). In chess, for example, the state is the board configuration (positions of the pieces) and an action is a movement of a piece; this movement changes the board configuration (state) and may or may not elicit a reward (points for removing an opposing piece).

So at each turn, the actor should choose the action it thinks will elicit most reward _in the long term_, considering the future states resulting from the current action. To mimick human action, we have the actor value the present more than the future (i.e. discount the future). That is, if _r(s, a)_ is the reward from acting on state _s_ with action _a_, the actor wants to maximise

<a href="https://www.codecogs.com/eqnedit.php?latex=V(s_0)&space;=&space;\sum_{t&space;=&space;0}^T&space;\gamma^t&space;r(s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V(s_0)&space;=&space;\sum_{t&space;=&space;0}^T&space;\gamma^t&space;r(s_t,&space;a_t)" title="V(s_0) = \sum_{t = 0}^T \gamma^t r(s_t, a_t)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=s_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_0" title="s_0" /></a> is the initial state of the game, <a href="https://www.codecogs.com/eqnedit.php?latex=s_{t&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_{t&plus;1}" title="s_{t+1}" /></a> is the state that results from acting on state <a href="https://www.codecogs.com/eqnedit.php?latex=s_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_t" title="s_t" /></a> with action <a href="https://www.codecogs.com/eqnedit.php?latex=a_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_t" title="a_t" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> is a discount factor.

The actor will have a brain, which will be a neural network mapping states to values, to make this decision. However, instead of having the network predict the entire sequence of discounted rewards for each action in the current state and choosing the action with the highest discounted rewards, it'll predict instead the _Q_-value of action _a_ on state _s_, defined as the maximum value you can obtain after acting on state _s_ with a. i.e. you get today's reward and the rewards corresponding to acting optimally in the future, starting with next period's state, _s'_. I.e.:

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,&space;a)&space;=&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}Q(s',&space;a')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s,&space;a)&space;=&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}Q(s',&space;a')" title="Q(s, a) = r(s, a) + \gamma \max_{a'}Q(s', a')" /></a>

Notice that <a href="https://www.codecogs.com/eqnedit.php?latex=\max_{a}Q(s,&space;a)&space;=&space;V(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max_{a}Q(s,&space;a)&space;=&space;V(s)" title="\max_{a}Q(s, a) = V(s)" /></a>, which is what we wanted to maximise in the first place. We can thus pick actions by running the network on a given state and choosing the action with highest Q(s, a).

So we'll let our player play the game, with random behaviour at first so it can build a 'memory' of diversified past experience to learn from, and train a neural network (the actor's 'brain') on this memory of past states, actions and rewards. The network should predict the _Q_-value of a given state for each possible action, so the actor can just pick the action with highest predicted Q-value. After each turn, we re-train the network on the new memory example and repeat. Over iterations, the network's estimate of _Q(s, a)_ should approach the true _Q(s, a)_, leading to better and better decisions as the actor plays the game.


## Some issues

There are a few aspects of peg solitaire that make it tricky to teach an AI using reinforcement learning.

The first is that to win the game, we have to play a _perfect_ game, not just a good one. The algorithm is trained to maximise the discounted sum of rewards so if over the course of learning, the actor doesn't get to 'experience' many perfect game, it may well learn to attain a high number of points and stop there, content. There are a few things I implemented to counteract this:
- I made the rewards increase convexly as the number of marbles left at the end of each turn falls. The idea is that by valueing removing the _last_ marble so much more than removing a marble near the beginning or middle of the game, the actor will weigh it's memories of playing long games more heavily when learning
- I set the discount rate high (close to 1) so that it values points it'll get in future actions in the same game as much as current ones
- I made sure the actor plays a LOT of fully random games at first. Since it won't know how to play the game, if we let the actor choose their own actions at first, it will never actually play a full game and won't have that in it's memory to learn. _Slowly_, as it learns, the actor plays more and more games non-randomly and hopefully this slow change helps get more full games into its memory

A second issue is that the game is fully deterministic, i.e. there's a fear the actor will just memorise one specific sequence of moves to beat the game and repeat that, without actually learning anything about the game. Luckily, we can play this game in different ways, depending on which spot we leave empty before the game starts (in the gif above it's the centre spot but there's others that work just as well). What I did to counteract this, then, is I make sure to train the actor only on games that start with the centre spot empty but when I track its progress, I make it play games that start with different empty spots. That way, if we see progress, it's because the actor is learning what sorts of moves are helpful, not just memorising one sequence of moves that works!


## The Algorithm

This is a description of an old version of the algorithm I'm using. Will update it...

1. Initialise the class instances:

   a) The game (a class which provides a state, takes in an action and returns a reward);

   b) An actor with a neural network that can interact with the above game;
   
2. Run game a few times. For each game, repeat turns until the game is won or lost. In each turn:
   
   a) Run a lottery to see if action will be random or if we'll allow the actor to pick the best action based on its network. This lottery will be initially biased towards random action and progressively loosened.
   
   b) If random, pick a random action. If we'll let the actor pick the move,
   
      i. run network on current state;
      ii. pick action with highest predicted Q-value;
   
   c) Store old state, chosen action, the reward and the new state (inc. an indicator for whether the game ended, _c_) in memory: _(s, a, r, s', c)_;

3. At the end of a set of games, train the network on a sample of the memory. In batches made from this sample:
   
   a) Run network on the batch of _s_ and _s'_, to get sets of _Q_-values for all actions in each state: _{Q(s, a)}_ and _{Q(s', a)}_;
   
   b) Update network's prediction of _{Q(s, a)}_ using its prediction on s' and the observed reward of acting a on s:
         
   <a href="https://www.codecogs.com/eqnedit.php?latex=Q(s,&space;a)&space;\to&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}&space;Q(s',&space;a')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s,&space;a)&space;\to&space;r(s,&space;a)&space;&plus;&space;\gamma&space;\max_{a'}&space;Q(s',&space;a')" title="Q(s, a) \to r(s, a) + \gamma \max_{a'} Q(s', a')" /></a>

   c) Train network using _s_ as the input and the above updated prediction for _{Q(s, a)}_ as the output, forcing the network to continuously update its own predictions; 

4. Go back to step 2. Every few repetitions of steps 2 and 3, track actor's progress by making it play a game fully non-randomly and storing the number of turns it lasted. This is just step 2 with non-random actions.

5. Repeat 2, 3 and 4 for as many epochs as it takes for the actor to learn the game.

The result should be an actor whose network properly predicts Q-values given states as inputs and can thus play the game by maximising its expected Q-value.

Here's a plot of the actor learning to play the game shown in the gif above:

![MRO2 Progress](https://github.com/gavarela/deepq/blob/master/images/MRO2.jpg)


## Tests

Tests: n-step Q Learning, varying learning rate, momentum rate and regularisation rate, etc.
