from cpu_thread import cpu_thread
from gpu_thread import gpu_thread
import os
import argparse
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PPO training')
parser.add_argument('--load', default=False,
                    help='Whether or not to load pretrained weights. '
                         'You must have started an alread trained net for it to work',
                    dest='load', type=str2bool)
parser.add_argument('--render', default=True, help='Show the game running in a separate process. '
                                                   'This slows the training a bit',
                    dest='render', type=str2bool)
parser.add_argument('--gif', default=False, help='Create a gif of a game every once in a while',
                    dest='gif', type=str2bool)


def main(args):
    assert args.render or not args.gif, 'If you want to display a gif, you must set render to true'
    if args.load is False and os.path.isfile('./model/breakout.pt'):
        while True:
            load = input('Are you sure you want to erase the previous training? (y/n) ')
            if load.lower() in ('y', 'yes', '1'):
                break
            elif load.lower() in ('n', 'no', '0'):
                import sys
                sys.exit()

    # create shared variables between all the processes
    manager = mp.Manager()
    # used to send the results of the net
    common_dict = manager.dict()
    # a queue of batches to be fed to the training net
    mem_queue = manager.Queue(1500 * mp.cpu_count())
    # a queue of operations pending
    process_queue = manager.Queue(mp.cpu_count()-1)

    with mp.Pool() as pool:
        try:
            workers: int = pool._processes
            print(f"Running pool with {workers//2} workers")
            pool.apply_async(gpu_thread, (args.load, mem_queue, process_queue, common_dict, [0, 1]))
            if args.render:
                pool.apply_async(cpu_thread, (2 if args.gif else 1, mem_queue, process_queue, common_dict, [2, 3]))
            for i in range(2*(1+args.render), workers, 2):
                pool.apply_async(cpu_thread, (0, mem_queue, process_queue, common_dict, [i, i+1]))

            # Wait for children to finish
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.join()


if __name__ == "__main__":
    args = parser.parse_args()
    mp.set_start_method('spawn')
    main(args)
