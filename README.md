# deepq

Teaching an AI to play the game peg solitaire using deep Q reinforcement learning.

## Intro & Files

The game is played by jumping pieces over other pieces to an empty spot, thereby 'eating' the piece that was jumped over and removing it from the game. The aim is to 'eat' all but one piece from the board. More info here: https://www.wikiwand.com/en/Peg_solitaire

The name in the file, Remain One, is a direct translation from the Portuguese name of the game.

The files are organised like this:
1. NeuralNet.py: implements a simple fully-connected neural network class, used for the deep learning aspect of the project;
2. RemainOne.py: implements the game. Running it runs an instance of the HumanGame class, which allows someone to play the game on the terminal using simple imputs to represent moves (e.g. 'f4 up'). It also defines a ComputerGame class, used for the deep Q learning program;
3. DeepQLearning.py: implements a deep Q learning player/agent and a class with a reinforcement learning routine;
4. RemainOneAI.py: the script that brings it all together and runs the program.


## Reinforcement Learning and Deep Q Learning

...

