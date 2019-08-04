# Pytorch multiprocessing PPO implementation playing Breakout

## How it works

The optimization is a standard PPO implementation, however the point was to push the limits of what a limited computer could do in reinforcement learning. Thus I use multiple processes to play the game and gather experiences. However, as my previous experiences taught me, if multiples processes try to access a single gpu, most of the computation time will be lost waiting for their turn on the gpu, rather than actually playing the game, resulting in a very limited speedup between multiprocessed and not multiprocessed algorithms. Furthermore it necessitated the net to be copied on multiple processes, wich was very VRAM consuming.
This algorithm works differently.

* multiple processes play the game
* a single process has access to the gpu
* when a playing process requires the gpu, it sends the operation to execute to the gpu process, and the gpu process sends back the result

This way, the speed limitation will go one step further, on your gpu or ram most likely

## Requirements

* Pytorch (gpu highly recommanded)
* Numpy
* gym (Atari)
* a few standard libraries such as argparse, time, os (you most likely already have them)
there might be a few modifications to make to run it in python 2

# Before any execution

Be careful, this requires a lot of ram, especially with many processes, so keep your ram in check and kill the program before it freezes your computer if necessary

## How to begin the training

* Clone this repository: `git clone https://github.com/CSautier/Breakout`
* Launch the game in bash: `python Breakout.py`

## Useful resources

* https://openai.com/
* https://arxiv.org/pdf/1707.06347.pdf

**Feel free to use as much of this code as you want but mention my github if you found this useful**.  
**For more information, you can contact me on my github.**
