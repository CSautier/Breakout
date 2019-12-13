import os
import torch
from model import PPO
import torch.optim as optim
from parameters import parameters


def process_observations(observation, model):
    with torch.no_grad():
        action = model.forward(observation)
        return action.cpu().squeeze().numpy()


def destack_process(model, process_queue, common_dict):
    if process_queue.qsize() > 0:
        model.eval()
        for _ in range(process_queue.qsize()):  # for instead of while to not get stuck
            pid, obs = process_queue.get(True)
            action = process_observations(torch.Tensor(obs).unsqueeze(0).to(parameters.DEVICE), model)
            common_dict[pid] = action


def destack_memory(memory_queue, observations, rewards, actions, probs):
    while memory_queue.qsize() > 0 and len(observations) <= parameters.MAXLEN:
        try:
            _, __, ___, ____ = memory_queue.get(True)
            observations = torch.cat((observations, torch.Tensor(_).to(parameters.DEVICE)))
            rewards = torch.cat((rewards, torch.Tensor(__).to(parameters.DEVICE)))
            actions = torch.cat((actions, torch.LongTensor(___).to(parameters.DEVICE)))
            probs = torch.cat((probs, torch.Tensor(____).to(parameters.DEVICE)))
        except RuntimeError as e:
            print(e)
            return True, observations, rewards, actions, probs
    return False, observations, rewards, actions, probs


def run_epoch(epochs, model, optimizer, observations, rewards, actions, probs):
    model.train()
    for _ in range(parameters.EPOCH_STEPS):
        perm = torch.randperm(len(probs))
        for i in range(0, len(probs), parameters.BATCH_SIZE):
            if i + parameters.BATCH_SIZE > len(probs):
                break
            optimizer.zero_grad()
            lossactor, losscritic = model.loss(observations[perm[i:i+parameters.BATCH_SIZE]], rewards[perm[i:i+parameters.BATCH_SIZE]], actions[perm[i:i+parameters.BATCH_SIZE]], probs[perm[i:i+parameters.BATCH_SIZE]])
            if epochs > 10:
                (lossactor + losscritic).backward()
            else:
                losscritic.backward()
            optimizer.step()
        print('Loss actor: {0:7.3f}  Loss critic: {1:7.3f}'.format(1000 * lossactor, 1000 * losscritic))


def gpu_thread(load, memory_queue, process_queue, common_dict, workers):
    # the only thread that has an access to the gpu, it will then perform all the NN computation
    import psutil
    p = psutil.Process()
    p.cpu_affinity(workers)
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        print('process started with pid: {} on core(s) {}'.format(os.getpid(), workers), flush=True)
        model = PPO()
        model.to(parameters.DEVICE)
        optimizer = optim.RMSprop(model.parameters(), lr=parameters.LEARNING_RATE)
        epochs = 0
        if load:
            checkpoint = torch.load('./model/breakout.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs = checkpoint['epochs']
        observations = torch.Tensor([]).to(parameters.DEVICE)
        rewards = torch.Tensor([]).to(parameters.DEVICE)
        actions = torch.LongTensor([]).to(parameters.DEVICE)
        probs = torch.Tensor([]).to(parameters.DEVICE)
        common_dict['epoch'] = epochs
        while True:
            memory_full, observations, rewards, actions, probs = \
                destack_memory(memory_queue, observations, rewards, actions, probs)
            destack_process(model, process_queue, common_dict)
            if len(observations) > parameters.MAXLEN or memory_full:
                epochs += 1
                print('-' * 60 + '\n        epoch ' + str(epochs) + '\n' + '-' * 60)
                run_epoch(epochs, model, optimizer, observations, rewards, actions, probs)
                observations = torch.Tensor([]).to(parameters.DEVICE)
                rewards = torch.Tensor([]).to(parameters.DEVICE)
                actions = torch.LongTensor([]).to(parameters.DEVICE)
                probs = torch.Tensor([]).to(parameters.DEVICE)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epochs': epochs
                }, './model/breakout.pt')
                common_dict['epoch'] = epochs
    except Exception as e:
        print(e)
        print('saving before interruption', flush=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs
        }, './model/breakout.pt')
