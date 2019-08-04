import torch
import math
from functools import reduce
from torch.optim.optimizer import Optimizer, required


class opSGD_lopAdam(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, hypergrad_lr=1e-6, lr_betas=(0.9,0.999), lr_eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, hypergrad_lr=hypergrad_lr, lr_betas=lr_betas, lr_eps=lr_eps)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(opSGD_lopAdam, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SGDHD doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._params_numel = reduce(lambda total, p: total + p.numel(), self._params, 0)

    def _gather_flat_grad_with_weight_decay(self, weight_decay=0):
        views = []
        for p in self._params:
            if p.grad is None:
                view = torch.zeros_like(p.data)
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            if weight_decay != 0:
                view.add_(weight_decay, p.data.view(-1))
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._params_numel

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        grad = self._gather_flat_grad_with_weight_decay(weight_decay)

        # NOTE: SGDHD has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['grad_prev'] = torch.zeros_like(grad)
            # Exponential moving average of hypergradient values
            state['exp_avg_h'] = grad.new_tensor(0)
            # Exponential moving average of squared hypergradient values
            state['exp_avg_h_sq'] = grad.new_tensor(0)
            
            state['exp_avg_sq'] = torch.zeros_like(grad) ##########
        # References and beta1_h, beta2_h coefficients for Hypergradient Descent Adam (HD Adam) of the learning rate
        exp_avg_h, exp_avg_h_sq = state['exp_avg_h'], state['exp_avg_h_sq']
        beta1_h, beta2_h = group['lr_betas']
        
        exp_avg_sq = state['exp_avg_sq']

        state['step'] += 1
        if state['step'] > 1:
            grad_prev = state['grad_prev']
            # Hypergradient for SGD
            h = torch.dot(grad, grad_prev)
            h = -h

            # Hypergradient Descent Adam (HD Adam) of the learning rate:
            exp_avg_h.mul_(beta1_h).add_(1 - beta1_h, h)
            exp_avg_h_sq.mul_(beta2_h).addcmul_(1 - beta2_h, h, h)
            #denom_ = exp_avg_h_sq.sqrt().add_(group['lr_eps'])
            denom_= torch.sum(exp_avg_sq).add_(group['lr_eps'])              #############

            bias_correction1_ = 1 - beta1_h ** state['step']
            bias_correction2_ = 1 - beta2_h ** state['step']
            step_size_ = group['hypergrad_lr'] * math.sqrt(bias_correction2_) / bias_correction1_
                    
            group['lr'] -= step_size_ * exp_avg_h / denom_
            #group['lr'].addcdiv_(-step_size_,exp_avg_h,denom_)

        if momentum != 0:
            if 'momentum_buffer' not in state:
                buf = state['momentum_buffer'] = torch.zeros_like(grad)
                buf.mul_(momentum).add_(grad)
            else:
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, grad)
            if nesterov:
                grad.add_(momentum, buf)
            else:
                grad = buf
        exp_avg_sq.mul_(0.999).addcmul_(1 - 0.999   , grad, grad) ##############
        state['grad_prev'] = grad

        self._add_grad(-group['lr'], grad)

        return loss