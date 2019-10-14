import torch
import utils as u


class ExperienceReplayManager:
    def __init__(self, memory_size=100000, img_shape=(1, 28, 28), replace_probability=0.2, ep_bs=32):
        self.memory_size = memory_size
        self.memory_batch = torch.empty((memory_size, *img_shape), dtype=torch.float32).to(u.dev())
        self.memory_label = torch.empty((memory_size, ), dtype=torch.int64).to(u.dev())

        self.in_use = 0
        self.bs = ep_bs
        self.replace_probability = replace_probability

    def append_with_ep(self, b, l):
        bs = b.shape[0]
        # update ep buffer
        if self.in_use < self.memory_size:  # ep buffer filling up
            end = min(self.in_use + bs, self.memory_size)
            self.memory_batch[self.in_use:end] = b[:end-self.in_use]
            self.memory_label[self.in_use:end] = l[:end-self.in_use]
            self.in_use = end
            return b, l
        else:
            indices_to_replace = torch.randint(0, self.memory_size, size=(bs, ))
            max_val = int(self.replace_probability * bs)
            end = torch.randint(high=max_val, size=(1, ))[0]
            b_replay = self.memory_batch[indices_to_replace[:end]].clone()
            l_replay = self.memory_label[indices_to_replace[:end]].clone()
            self.memory_batch[indices_to_replace] = b
            self.memory_label[indices_to_replace] = l
            b = torch.cat([b, b_replay], dim=0)
            l = torch.cat([l, l_replay], dim=0)
            return b, l
