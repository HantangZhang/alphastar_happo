""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 14:25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as cate
from configs.hyper_parameters import Model_Parameters as MP
from configs.hyper_parameters import Action_Space_Parameters as ASP
from algorithm.model.alpha.spatial_encoder import ResBlock1D
from algorithm.model.base.glu import GLU
from utils.util import check

class ActionTypeHead(nn.Module):

    def __init__(self, core_output_dim=MP.core_hidden_dim, n_resblocks=MP.n_resblocks,
                 original_256=MP.original_256, max_action_num=ASP.level1_action_dim,
                 context_size=MP.context_size, autoregressive_embedding_size=MP.autoregressive_embedding_size,
                 use_level1_action_type_mask=ASP.use_level1_action_type_mask,
                 temperature=ASP.temperature, device=torch.device("cpu")):
        super().__init__()

        self.temperature = temperature
        self.embed_fc = nn.Linear(core_output_dim, original_256)
        self.resblock_stack = nn.ModuleList([
            ResBlock1D(inplanes=256, planes=256, seq_len=1)
            for _ in range(n_resblocks)
        ])
        self.max_action_num = max_action_num
        self.glu_1 = GLU(input_size=original_256, context_size=context_size,
                         output_size=max_action_num)
        self.fc_1 = nn.Linear(max_action_num, original_256)
        self.glu_2 = GLU(input_size=original_256, context_size=context_size,
                         output_size=autoregressive_embedding_size)
        self.glu_3 = GLU(input_size=core_output_dim, context_size=context_size,
                         output_size=autoregressive_embedding_size)
        self.softmax = nn.Softmax(dim=-1)

        self.use_level1_action_type_mask = use_level1_action_type_mask

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)


    def forward(self, core_output, scalar_context, level1_action_mask=None):

        batch_size = core_output.shape[0]

        x = self.embed_fc(core_output)
        x = x.unsqueeze(-1)
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = F.relu(x)
        x = x.squeeze(-1)

        # action type through a `GLU` gated by `scalar_context`.
        action_type_logits = self.glu_1(x, scalar_context)

        if self.use_level1_action_type_mask and level1_action_mask is not None:
            action_mask = level1_action_mask.bool()
            action_type_logits = action_type_logits + (~action_mask * (-1e9))

        temperature = self.temperature
        action_type_logits = action_type_logits / temperature

        action_type_probs = self.softmax(action_type_logits)
        action_type_probs = action_type_probs.reshape(batch_size, -1)

        # level1_action
        dist = cate.Categorical(probs=action_type_probs)
        level1_action = dist.sample()
        level1_action = level1_action.reshape(batch_size, -1)

        level1_action_log_prob = dist.log_prob(level1_action.squeeze(-1)).view(level1_action.size(0), -1).sum(-1).unsqueeze(-1)
        # level1_action_log_prob = level1_action_log_prob.reshape(batch_size, -1)

        action_type_one_hot = tensor_one_hot(level1_action, self.max_action_num)
        action_type_one_hot = action_type_one_hot.squeeze(-2)

        z = F.relu(self.fc_1(action_type_one_hot))
        z = self.glu_2(z, scalar_context)
        t = self.glu_3(core_output, scalar_context)
        autoregressive_embedding = z + t

        return level1_action_log_prob, level1_action, autoregressive_embedding

    def evaluate_actions(self, core_output, scalar_context, level1_action, level1_action_mask=None, active_masks=None):
        x = self.embed_fc(core_output)
        x = x.unsqueeze(-1)
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = F.relu(x)
        x = x.squeeze(-1)

        # action type through a `GLU` gated by `scalar_context`.
        action_type_logits = self.glu_1(x, scalar_context)

        if self.use_level1_action_type_mask and level1_action_mask is not None:
            action_mask = level1_action_mask.bool()
            action_type_logits = action_type_logits + (~action_mask * (-1e9))

        temperature = self.temperature
        action_type_logits = action_type_logits / temperature
        action_type_probs = self.softmax(action_type_logits)

        dist = cate.Categorical(probs=action_type_probs)
        level1_action_probs = dist.log_prob(level1_action.squeeze(-1)).view(level1_action.size(0), -1).sum(-1).unsqueeze(-1)
        if active_masks is not None:
            dist_entropy = (dist.entropy() * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = dist.entropy().mean()

        level1_action = level1_action.to(torch.int64)
        action_type_one_hot = tensor_one_hot(level1_action, self.max_action_num)
        action_type_one_hot = action_type_one_hot.squeeze(-2)

        z = F.relu(self.fc_1(action_type_one_hot))
        z = self.glu_2(z, scalar_context)
        t = self.glu_3(core_output, scalar_context)
        autoregressive_embedding = z + t

        return level1_action_probs, dist_entropy, autoregressive_embedding

def tensor_one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    cuda_check = labels.is_cuda
    if cuda_check:
        get_cuda_device = labels.get_device()

    y = torch.eye(num_classes)

    if cuda_check:
        y = y.to(get_cuda_device)

    return y[labels]

if __name__ == '__main__':
    # model = ActionTypeHead()
    # core_output = torch.ones((3, 256))
    # scalar_context = torch.ones((3, 128))
    # level1_action_mask = torch.ones((3, 6))
    # active_masks = torch.zeros((3, 1))
    # active_masks[2] = 0
    # level1_action = torch.ones((3, 1))
    # a, b, c = model(core_output, scalar_context)
    # # a, b = model.evaluate_actions(core_output, scalar_context, level1_action, level1_action_mask=None, active_masks=None)
    # print(a.shape)
    # print(b)

    action = torch.zeros((30, 1))
    action = action.to(torch.int64)
    y = tensor_one_hot(action, 4)
    print(y)
