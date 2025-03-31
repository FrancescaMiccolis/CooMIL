import copy
import itertools

def get_experiments(args):
    args1 = copy.copy(args)
    args1.mem = 64
    args1.model="conslide"
    args1.dataset = "seq-wsi"
    args1.exp_desc= "conslide"
    args1.buffer_size=1100
    args1.alpha=0.2
    args1.seed = 12
    args1.beta=0.2
    hyperparameters = [[5,6,7,8,9]]
    args_list = []
    for element in itertools.product(*hyperparameters):
        args2 = copy.copy(args1)
        args2.fold = element
        if args.debug_mode:
            args2.n_epochs=50
        else:
            args2.n_epochs=2
        args_list.append(args2)
    return args_list
