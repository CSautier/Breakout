import os
import gym
import time
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from parameters import parameters


def sample(action):
    tresh = random.random()
    p = 0.
    for i, prob in enumerate(action):
        p += prob
        if p > tresh:
            return i, prob


def process_frame(frame):
    return (frame/255).transpose((2, 0, 1))


def process_reward(reward):
    return 0.01*reward


def generate_game(env, pid, process_queue, common_dict):
    observation = env.reset()
    done = False
    reward_list = []
    action_list = []
    prob_list = []
    observation_list = []
    frame_count = 0
    while not done:
        observation = process_frame(observation)
        observation_list.append(observation)
        process_queue.put((pid, observation))
        while pid not in common_dict:
            time.sleep(0.0001)
        action_prob = common_dict[pid]
        del common_dict[pid]
        action, prob = sample(action_prob)
        observation, reward, done, info = env.step(action)
        action_list.append(action)
        prob_list.append(prob)
        reward_list.append(process_reward(reward))
        frame_count += 1
    print("Score: {:4.0f}".format(100*sum(reward_list)))
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += reward_list[i + 1] * parameters.GAMMA  # compute the discounted obtained reward for each step
    return observation_list, reward_list, action_list, prob_list


def play(env, pid, process_queue, common_dict):
    while True:
        counter = 0
        observation = env.reset()
        done = False
        while not done:
            counter += 1
            if counter >= 2000:
                break
            process_queue.put((pid, process_frame(observation)))
            while pid not in common_dict:
                time.sleep(0.0001)
            action_prob = common_dict[pid]
            del common_dict[pid]
            action = sample(action_prob)[0]
            observation, _, done, _ = env.step(action)
            env.render()


def play_to_gif(env, pid, process_queue, common_dict):
    display = False
    episode = 0
    while 'epoch' not in common_dict:
        time.sleep(0.001)
    while True:
        if common_dict['epoch'] % 25 == 0 and not display:
            display = True
            episode = common_dict['epoch']
        observation = env.reset()
        frames = []
        done = False
        while not done:
            process_queue.put((pid, process_frame(observation)))
            while pid not in common_dict:
                time.sleep(0.0001)
            action_prob = common_dict[pid]
            del common_dict[pid]
            action = sample(action_prob)[0]
            observation, _, done, _ = env.step(action)
            if display:
                frames.append(env.render(mode='rgb_array'))
            else:
                env.render()
        if display:
            display_frames_as_gif(frames, 'Episode {}.gif'.format(episode))
            display = False


def display_frames_as_gif(frames, name):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=33)
    try:
        anim.save('gifs/' + name)
    except:
        anim.save('gifs/' + name, writer=animation.PillowWriter(fps=40))


def cpu_thread(render, memory_queue, process_queue, common_dict, workers):
    import psutil
    p = psutil.Process()
    p.cpu_affinity(workers)
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        env = gym.make('Breakout-v0')
        pid = os.getpid()
        print('process started with pid: {} on core(s) {}'.format(os.getpid(), workers), flush=True)
        if render == 1:
            play(env, pid, process_queue, common_dict)
        elif render == 2:
            play_to_gif(env, pid, process_queue, common_dict)
        else:
            while True:
                observation_list, reward_list, action_list, prob_list = generate_game(env, pid, process_queue, common_dict)
                memory_queue.put((observation_list, reward_list, action_list, prob_list))
    except Exception as e:
        print(e, flush=True)
