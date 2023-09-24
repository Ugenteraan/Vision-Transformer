'''Module to produce a positional encoding matrix.
'''

import math
import torch


class PositionalEncoder:
    '''Sine and Cosine positional encoder module.
    '''

    def __init__(self, token_length, output_dim, n=1000):
        '''Param init.
        '''

        self.token_length = token_length
        self.output_dim = output_dim
        self.n = n
        self.P = torch.zeros((token_length, output_dim), requires_grad=False)

    
    def calc_pos_embedding(self):
        '''Calculate the positional embedding matrix.
        '''
        k = torch.arange(start=0, end=self.token_length, requires_grad=False)
        i = torch.arange(start=0, end=math.ceil(self.output_dim/2),requires_grad=False)
        
        denominators = torch.pow(self.n, torch.div(torch.mul(i, 2), self.output_dim))

        stacked_denominators = denominators.reshape(1,-1).repeat_interleave(self.token_length, dim=0)

        values = torch.div(k, stacked_denominators.transpose(0,1))

        self.P.transpose(0,1)[::2] = torch.sin(values)

        try:
            self.P.transpose(0,1)[1::2] = torch.cos(values)

        except RuntimeError:
            self.P.transpose(0,1)[1::2] = torch.cos(values[:-1])

        return None


    def __call__(self):
        
        self.calc_pos_embedding()

        return self.P

