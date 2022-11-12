import torch
from algorithm.model.alpha.airbattle_model import Actor, Critic

class HappoAlphaPolicy:

    def __init__(self, args, device=torch.device("cpu")):
        self.args = args
        self.device = device

        self.actor = Actor(device=device)
        self.critic = Critic(device=device)

        # self.level1_action_optimizer = torch.optim.Adam(self.actor.action_type_head.parameters(),
        #                                                 lr=self.lr, eps=self.opti_eps,
        #                                                 weight_decay=self.weight_decay)
        # self.target_optimizer = torch.optim.Adam(self.actor.target_unit_head.parameters(),
        #                                                 lr=self.lr, eps=self.opti_eps,
        #                                                 weight_decay=self.weight_decay)
        # self.location_optimizer = torch.optim.Adam(self.actor.location_head.parameters(),
        #                                                 lr=self.lr, eps=self.opti_eps,
        #                                                 weight_decay=self.weight_decay)


    def get_actions(self, core_output, scalar_context, entity_embeddings, baseline_state, level1_action_mask):
        '''
        core_output [batch, core_output_dim
        scalar_contex [batch, scalar_out_dim

        '''

        l1_action, l1_action_prob, location, location_prob, target, target_prob = self.actor(core_output, scalar_context, entity_embeddings, level1_action_mask)

        values = self.critic(core_output, baseline_state)

        return values, l1_action, l1_action_prob, location, location_prob, target, target_prob

    def get_values(self, core_output, baseline_state):

        values = self.critic(core_output, baseline_state)
        return values

    def evaluate_actions(self, core_output, scalar_context, entity_embeddings, level1_action_mask, level1_action,
                         target_jet, location_id, active_masks, baseline_state):
        # 和actor不同的是它还输出value
        level1_action_log_prob, level1_entropy, target_probs, target_entropy, location_probs, location_entropy = self.actor.evaluate_actions(
            core_output, scalar_context, entity_embeddings, level1_action_mask, level1_action, target_jet, location_id,
            active_masks
        )
        # 这里actor更新的时候，会更新encoder，然后释放梯度，如果这个core再带着梯度传进去，就会再更新一次encoder，但因为前面释放了梯度，所以会报错，所以这里要把core的梯度啥放
        core_output = core_output.detach()
        values = self.critic(core_output, baseline_state)

        return values, level1_action_log_prob, level1_entropy, target_probs, target_entropy, location_probs, location_entropy

