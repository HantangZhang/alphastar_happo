""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 14:26
"""

import torch
import torch.nn as nn
import torch.distributions.categorical as cate
from algorithm.model.alpha import mask_func
from configs.hyper_parameters import Model_Parameters as MP
from configs.hyper_parameters import Action_Space_Parameters as AHP
'''
we first use a mask to determine which entity types can accept the action-type
We also compute a mask representing which units can be selected

alphastar先选实体类型，再在可选的实体类型里面选择哪些实体可以被选
选目标的时候，先决定哪些实体类型可以作为目标，然后再在目标类型里选择哪些目标可以被选择


空战场景里并没有实体类型需要选择，直接给出哪些目标可以被选择就可以了

'''

class SelectedTargetHead(nn.Module):

    def __init__(self, embedding_size=MP.entity_embedding_size, original_64=MP.original_64,
                 autoregressive_embedding_size=MP.autoregressive_embedding_size, original_256=MP.original_256,
                 temperature=AHP.temperature, device=torch.device("cpu")):
        super().__init__()

        self.conv_1 = nn.Conv1d(in_channels=embedding_size,
                                out_channels=original_64, kernel_size=1, stride=1,
                                padding=0, bias=True)
        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)
        self.fc_2 = nn.Linear(original_256, original_64)
        self.softmax = nn.Softmax(dim=-1)

        self.temperature = temperature
        self.to(device)

    def forward(self, autoregressive_embedding, level1_action, entity_embeddings):
        entity_size = entity_embeddings.shape[1]
        mask = mask_func.generate_target_mask(level1_action).to(level1_action.device)
        # 将实体的信息embeddding成16个chnnel
        key = self.conv_1(entity_embeddings.transpose(-1,-2)).transpose(-1, -2)

        x = self.fc_1(autoregressive_embedding)
        query = self.fc_2(x).unsqueeze(1)

        # 在空战场景中， entity_size应该就是或者加上所有导弹吧
        # key: [batch_size x entity_size x key_size]
        # query：[batch_size x sqe_len x hidden_size]    sqe_len就是指的auto_embedding
        y = torch.bmm(key, query.transpose(-1, -2))

        # batch_size x entity_size
        y = y.squeeze(-1)
        # mask中是false的会被赋一个很大的负数
        target_jet_logits = y.masked_fill(~mask, -1e9)

        temperature = self.temperature
        target_jet_logits = target_jet_logits / temperature

        target_jet_prob = self.softmax(target_jet_logits)
        dist = cate.Categorical(probs=target_jet_prob)
        target_jet = dist.sample()

        target_probs = dist.log_prob(target_jet).view(target_jet.size(0), -1).sum(-1).unsqueeze(-1)

        target_jet = target_jet.unsqueeze(dim=1)
        no_target = torch.sum(mask, 1)
        no_target_mask = ~no_target.bool()
        target_jet[no_target_mask] = entity_size - 1

        return target_probs, target_jet
        # 如果有目标，相加的结果就一定不为0，转为bool值后就是TRUE，作为index检索的时候target_jet[true]就会将这个位置的值转为指定的数值
        # 所以要用~将有目标的转为False，no_target_mask就是有目标的为FALSE, 无目标地为TRUE
        # no_target = torch.sum(mask, 1)
        # no_target_mask = ~no_target.bool()

        # # 如果这个动作不需要选择目标，那个这个输出头将被忽略
        # target_jet_probs = self.softmax(target_jet_logits)
        # target_jet = torch.multinomial(target_jet_probs, 1)

        # target_jet = target_jet.unsqueeze(dim=1)
        # 如果这个动作不需要选择目标，即mask为在对应位置为TRUE, 所以改为entity编号最后一个实体，它是None
        # target_jet[no_target_mask] = entity_size - 1

        # target_jet_logits = target_jet_logits.unsqueeze(dim=1)
        # target_jet_logits[no_target_mask] = 0
        # target_jet_prob2 = self.softmax(target_jet_logits)

    def evaluate_actions(self, autoregressive_embedding, level1_action, entity_embeddings, target_jet, active_masks=None):
        entity_size = entity_embeddings.shape[1]
        mask = mask_func.generate_target_mask(level1_action).to(level1_action.device)

        # 将实体的信息embeddding成16个chnnel
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2)

        x = self.fc_1(autoregressive_embedding)
        query = self.fc_2(x).unsqueeze(1)

        # 在空战场景中， entity_size应该就是或者加上所有导弹吧
        # key: [batch_size x entity_size x key_size]
        # query：[batch_size x sqe_len x hidden_size]    sqe_len就是指的auto_embedding
        y = torch.bmm(key, query.transpose(-1, -2))

        # batch_size x entity_size
        y = y.squeeze(-1)
        # mask中是false的会被赋一个很大的负数
        target_jet_logits = y.masked_fill(~mask, -1e9)

        temperature = self.temperature
        target_jet_logits = target_jet_logits / temperature

        target_jet_prob = self.softmax(target_jet_logits)
        dist = cate.Categorical(probs=target_jet_prob)

        target_probs = dist.log_prob(target_jet.squeeze(-1)).view(target_jet.size(0), -1).sum(-1).unsqueeze(-1)

        if active_masks is not None:
            dist_entropy = (dist.entropy() * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = dist.entropy().mean()

        return target_probs, dist_entropy


if __name__ == '__main__':
    # batch_size = 1
    # entity_size = 512
    # mask = torch.arange(entity_size).float()
    # mask = mask.repeat(batch_size, 1)
    # entity_num = torch.tensor([55])
    # mask = mask < entity_num.unsqueeze(dim=1)
    # level1_action = torch.tensor([2])
    # mask = mask_func.generate_target_mask(level1_action).to(level1_action.device)
    # mask = mask.bool()
    # print(type(mask))
    # y = torch.empty((1, 10))
    # target = y.masked_fill(~mask, -1e9)
    # print(target)
    #
    # target_jet = torch.full((10, 1), 4)
    # # no_target = torch.zeros((10, 1))
    # mask = torch.zeros((10, 11))
    #
    # mask[0][3] = 1
    # mask[3][2] = 1
    # mask[3][5] = 1
    # no_target = torch.sum(mask, 1)
    # # no_target = no_target.bool()
    # print(no_target)
    # no_target_mask = ~no_target.bool()
    # print(no_target_mask, 2222)
    #
    # target_jet[no_target_mask] = 212
    # print(target_jet)
    model = SelectedTargetHead()
    autoregressive_embedding = torch.ones((3, 256))
    level1_action = torch.ones((3, 1))
    level1_action[0] = 0
    entity_embeddings = torch.ones((3, 20, 256))
    target_jet = torch.full((3, 1), 7, dtype=torch.float32)
    active_masks = torch.ones((3, 1))

    res1, res2 = model(autoregressive_embedding, level1_action, entity_embeddings)
    print(res1, res2)
    res3, res4 = model.evaluate_actions(autoregressive_embedding, level1_action, entity_embeddings, target_jet)
    print(res3, res4)