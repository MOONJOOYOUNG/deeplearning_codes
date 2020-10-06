import numpy as np
from collections import Iterable

class Forgetting_Events(object):
    def __init__(self, data_size):
        self.file_name = np.full((data_size),'______save_files_name______')
        self.prev_acc = np.zeros((data_size))
        self.forgetting = np.zeros((data_size))
        self.first_learn = np.zeros((data_size))

    def update(self, data_idx, correctness, filename, epoch=None):
        data_idx = data_idx.cpu().numpy()
        curr_acc = correctness.cpu().numpy()

        # update file_name
        if epoch == 1:
            self.file_name[data_idx] = filename

        # update forgetting count
        forgetting_idx = np.where(self.prev_acc[data_idx] > curr_acc)[0]
        self.forgetting[data_idx[forgetting_idx]] += 1
        # update prev_acc
        self.prev_acc[data_idx] = curr_acc

        # update first learn
        first_learn_idx = np.where((self.first_learn[data_idx] == 0) & (curr_acc == 1))[0]
        self.first_learn[data_idx[first_learn_idx]] = epoch

    def get_forgettable_examples(self, sorted=None):
        forgettable_idx = np.where(self.forgetting > 0)[0]
        forgettable_examples = self.forgetting[forgettable_idx]
        forgettable_fimenames = self.file_name[forgettable_idx]

        if sorted == 'reverse':
            sort_idx = np.argsort(forgettable_examples)[::-1]
        else:
            sort_idx = np.argsort(forgettable_examples)

        return forgettable_examples[sort_idx], forgettable_fimenames[sort_idx], self.first_learn[forgettable_idx]

    def get_unforgettable_examples(self, sorted=None):
        unforgettable_idx = np.where(self.forgetting == 0)[0]
        unforgettable_examples = self.forgetting[unforgettable_idx]
        unforgettable_fimenames = self.file_name[unforgettable_idx]

        if sorted == 'reverse':
            sort_idx = np.argsort(self.first_learn[unforgettable_idx])[::-1]
        else:
            sort_idx = np.argsort(self.first_learn[unforgettable_idx])

        return unforgettable_examples[sort_idx], unforgettable_fimenames[sort_idx], self.first_learn[unforgettable_idx]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, path, int_form=':04d', float_form=':.6f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)

        return log

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], correct.squeeze()





