import numpy as np
import torch
import re

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from torch.cuda.amp import autocast

class EnsembleRSSM(nn.Module):
    def __init__(
            self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
            act='elu', norm='none', std_act='softplus', min_std=0.1):
        super.__init__()
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._cell = GRUCell(self._deter, norm=True)
        self._cast = lambda x: x.to(torch.float32)

    def initial(self, batch_size):
        dtype = torch.float32
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete], dtype=dtype),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete], dtype=dtype),
                deter=self._cell.get_initial_state(batch_size=batch_size, dtype=dtype))
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], dtype=dtype),
                std=torch.zeros([batch_size, self._stoch], dtype=dtype),
                stoch=torch.zeros([batch_size, self._stoch], dtype=dtype),
                deter=self._cell.get_initial_state(batch_size=batch_size, dtype=dtype))
        return state
    
    @torch.jit.script
    def observe(self, embed, action, is_first, state=None):
        
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        
        if state is None:
            state = self.initial(torch.tensor(action.shape[0]))
            
        post, prior = common.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (swap(action), swap(embed), swap(is_first)), (state, state))
        
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        
        return post, prior
    
    @torch.jit.script
    def imagine(self, action, state=None):
        
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        
        if state is None:
            state = self.initial(torch.tensor(action.shape[0]))
            
        assert isinstance(state, dict), state
        
        action = swap(action)
        prior = self.static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        
        return prior
    
    def get_feat(self, state):
        
        stoch = self._cast(state['stoch'])
        
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
            
        return torch.cat([stoch, state['deter']], dim=-1)
    
    def get_dist(self, state, ensemble=False):
        
        if ensemble:
            state = self._suff_stats_ensemble(state['deter'])
        
        if self._discrete:
            logit = state['logit']
            logit = logit.float()
            dist = dist.OneHotCategorical(logits=logit)
        else:
            mean, std = state['mean'], state['std']
            mean = mean.float()
            std = std.float()
            dist = dist.MultivariateNormal(mean, torch.diag(std))
            
        return dist
    
    @torch.jit.script
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        
        def multiply_tensor(tensor, factor):
            return factor * tensor
        
        prev_state, prev_action = torch.jit.script(
            torch.nest.map_structure(
                lambda x: multiply_tensor(1.0 - is_first.to(x.dtype), x),
                (prev_state, prev_action)))
        
        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self.get('obs_out', torch.nn.Linear, self._hidden)(x)
        x = self.get('obs_out_norm', NormLayer, self._norm)(x)
        x = self._act(x)
        stats = self._suff_stats_layer('obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior
    
    @torch.jit.script
    def img_step(self, prev_state, prev_action, sample=True):
        
        prev_stoch = self._cast(prev_state['stoch'])
        prev_action = self._cast(prev_action)
        
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
            
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.get('img_in', torch.nn.Linear, self._hidden)(x)
        x = self.get('img_in_norm', NormLayer, self._norm)(x)
        x = self._act(x)
        deter = prev_state['deter']
        x, deter = self._cell(x, [deter])
        deter = deter[0]  # Keras wraps the state in a list.
        stats = self._suff_stats_ensemble(x)
        index = torch.randint(high=self._ensemble, size=(), dtype=torch.int32)
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior
    
    def _suff_stats_ensemble(self, inp):
        
        bs = list(inp.shape[:-1])
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        
        for k in range(self._ensemble):
            x = self.get(f'img_out_{k}', torch.nn.Linear, self._hidden)(inp)
            x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
            x = self._act(x)
            stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
            
        stats = {k: torch.stack([x[k] for x in stats], 0)
                 for k, v in stats[0].items()}
        stats = {k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
                 for k, v in stats.items()}
        return stats
    
    def _suff_stats_layer(self, name, x):
        
        if self._discrete:
            x = self.get(name, torch.nn.Linear, self._stoch * self._discrete, None)(x)
            logit = x.reshape(x.shape[:-1] + [self._stoch, self._discrete])
            return {'logit': logit}
        
        else:
            x = self.get(name, torch.nn.Linear, 2 * self._stoch, None)(x)
            mean, std = torch.split(x, [self._stoch] * 2, dim=-1)
            std = {
                'softplus': lambda: torch.nn.functional.softplus(std),
                'sigmoid': lambda: torch.sigmoid(std),
                'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}
        
    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        
        kld = torch.distributions.kl.kl_divergence
        sg = lambda x: torch.nest.map_structure(torch.Tensor.detach, x)
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = torch.maximum(value, free).mean()
        
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free)
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
                
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
            
        return loss, value
    
class Encoder(torch.nn.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    super(Encoder, self).__init__()
    self.shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    print('Encoder CNN inputs:', list(self.cnn_keys))
    print('Encoder MLP inputs:', list(self.mlp_keys))
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers

    @torch.jit.script
    def forward(self, data):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[:-len(shape)]
        data = {
            k: torch.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
            for k, v in data.items()}
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = torch.cat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])
    
    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** i * self._cnn_depth
            with autocast():
                x = torch.nn.Conv2d(x.shape[-1], depth, kernel, stride=2)(x)
                x = NormLayer(self._norm)(x)
                x = self._act(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = torch.cat(list(data.values()), -1)
        for i, width in enumerate(self._mlp_layers):
            with autocast():
                x = torch.nn.Linear(x.shape[-1], width)(x)
                x = NormLayer(self._norm)(x)
                x = self._act(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, shape, layers, units, act='elu', norm='none', **out):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out
        
        self.linears = nn.ModuleList()
        for index in range(self._layers):
            self.linears.append(nn.Linear(units, units))
        
        self.norms = nn.ModuleList()
        for index in range(self._layers):
            if norm == 'layer':
                self.norms.append(nn.LayerNorm(units))
            else:
                self.norms.append(nn.Identity())
        
        self.dist = DistLayer(shape=self._shape, **self._out)
                
    def forward(self, features):
        x = features.float()
        x = x.reshape(-1, x.shape[-1])
        
        for linear, norm in zip(self.linears, self.norms):
            x = linear(x)
            x = norm(x)
            x = self._act(x)
        
        x = x.reshape(*features.shape[:-1], -1)
        x = self.dist(x)
        
        return x

    
class GRUCell(nn.Module):

    def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(size * 2, 3 * size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm(size * 3)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, self._size, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]
    
class DistLayer(nn.Module):
    def __init__(self, shape, dist='mse', min_std=0.1, init_std=0.0):
        super().__init__()
        self.shape = shape
        self.dist = dist
        self.min_std = min_std
        self.init_std = init_std
        self.out_layer = nn.Linear(in_features=None, out_features=torch.prod(torch.tensor(shape)))
        if self.dist in ('normal', 'tanh_normal', 'trunc_normal'):
            self.std_layer = nn.Linear(in_features=None, out_features=torch.prod(torch.tensor(shape)))

    def forward(self, inputs):
        out = self.out_layer(inputs).reshape((-1,) + self.shape)
        if self.dist in ('normal', 'tanh_normal', 'trunc_normal'):
            std = self.std_layer(inputs).reshape((-1,) + self.shape)
        if self.dist == 'mse':
            dist = td.Normal(out, 1.0)
            return td.Independent(dist, len(self.shape))
        if self.dist == 'normal':
            dist = td.Normal(out, std)
            return td.Independent(dist, len(self.shape))
        if self.dist == 'binary':
            dist = td.Bernoulli(out)
            return td.Independent(dist, len(self.shape))
        if self.dist == 'tanh_normal':
            mean = 5 * torch.tanh(out / 5)
            std = torch.nn.functional.softplus(std + self.init_std) + self.min_std
            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, td.transforms.TanhTransform())
            dist = td.Independent(dist, len(self.shape))
            return dist.sample()
        if self.dist == 'trunc_normal':
            std = 2 * torch.sigmoid((std + self.init_std) / 2) + self.min_std
            dist = td.TruncatedNormal(torch.tanh(out), std, -1, 1)
            return td.Independent(dist, 1)
        if self.dist == 'onehot':
            dist = td.OneHotCategorical(logits=out)
            return td.Independent(dist, len(self.shape))
        raise NotImplementedError(self.dist)  

class NormLayer(nn.Module):

    def __init__(self, name):
        super().__init__()

        if name == 'none':
            self._layer = None
        elif name == 'layer':
            self._layer = nn.LayerNorm(normalized_shape=-1, elementwise_affine=True)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if not self._layer:
            return features
        return self._layer(features)

    
def get_act(name):
    if name == 'none':
        return lambda x: x
    elif name == 'mish':
        return lambda x: x * torch.tanh(F.softplus(x))
    elif hasattr(F, name):
        return getattr(F, name)
    else:
        raise NotImplementedError(name)