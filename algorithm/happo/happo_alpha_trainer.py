import torch
import torch.nn as nn
import numpy as np

from utils.popart import PopArt
from utils.util import get_gard_norm, huber_loss, mse_loss, check

class HappoAlphaTrainer():

    def __init__(self, args, encoder, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.encoder = encoder
        self.policy = policy

        self.lr = args.lr
        self.critic_lr = args.critic_lr

        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        params = list(self.encoder.parameters()) + list(self.policy.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(params,
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(),
                                                 lr=self.lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

        self._use_popart = args.use_popart

        # ppo算法参数
        self.clip_param = args.clip_param

        # 训练
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        # 两个coef的作用 todo
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        # 两个损失计算的参数 todo
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        # 是否使用
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def ppo_update(self, sample, agent_id, update_actor=True):
        '''
        state_batch, obs_batch, level1_action_mask_batch, level1_action_batch, target_jet_batch, location_id_batch, active_masks_batch, \
                  old_level1_action_probs_batch, target_prob_batch, location_prob_batch, adv_targ_batch, level1_factor_batch, \
                  target_factor_batch, location_factor_batch, value_preds_batch, returns_batch
        '''

        state, obs, baseline_state, level1_action_mask, level1_action, target_jet, location_id, active_masks, old_level1_action_probs_batch, \
         old_target_probs, old_location_probs, adv_targ, \
        level1_factor, target_factor, location_factor, value_preds_batch, return_batch = sample

        old_level1_action_probs_batch = check(old_level1_action_probs_batch).to(**self.tpdv)
        old_location_probs = check(old_location_probs).to(**self.tpdv)
        old_target_probs = check(old_target_probs).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        level1_factor = check(level1_factor).to(**self.tpdv)
        target_factor = check(target_factor).to(**self.tpdv)
        location_factor = check(location_factor).to(**self.tpdv)

        core_output, scalar_context, entity_embeddings = self.encoder(state, obs)
        values, level1_action_log_prob, level1_entropy, target_probs, target_entropy, location_probs, location_entropy = self.policy.evaluate_actions(
            core_output, scalar_context[:, agent_id, :], entity_embeddings, level1_action_mask, level1_action, target_jet, location_id,
            active_masks, baseline_state
        )

        imp_weights_level1 = torch.prod(torch.exp(level1_action_log_prob - old_level1_action_probs_batch), dim=-1, keepdim=True)
        imp_weights_target = torch.prod(torch.exp(target_probs - old_target_probs), dim=-1, keepdim=True)
        imp_weights_location = torch.prod(torch.exp(location_probs - old_location_probs), dim=-1, keepdim=True)

        # # 一级动作损失计算及更新参数
        surr1_level1 = imp_weights_level1 * adv_targ
        surr2_level1 = torch.clamp(imp_weights_level1, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_level1_action_loss = (-torch.sum(level1_factor * torch.min(surr1_level1, surr2_level1),
                                         dim=-1,
                                         keepdim=True) * active_masks).sum() / active_masks.sum()
        # self.policy.level1_action_optimizer.zero_grad()
        level1_action_loss = (policy_level1_action_loss - level1_entropy * self.entropy_coef)
        # level1_action_loss.backward()
        # if self._use_max_grad_norm:
        #     level1_action_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.action_type_head.parameters(), self.max_grad_norm)
        # else:
        #     level1_action_grad_norm = get_gard_norm(self.policy.actor.action_type_head.parameters())
        #
        # self.policy.level1_action_optimizer.step()
        #
        # target
        surr1_target = imp_weights_target * adv_targ
        surr2_target = torch.clamp(imp_weights_target, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_target_loss = (-torch.sum(target_factor * torch.min(surr1_target, surr2_target),
                                                dim=-1,
                                                keepdim=True) * active_masks).sum() / active_masks.sum()
        # self.policy.target_optimizer.zero_grad()
        target_loss = (policy_target_loss - target_entropy * self.entropy_coef)
        # target_loss.backward()
        # if self._use_max_grad_norm:
        #     target_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.target_unit_head.parameters(), self.max_grad_norm)
        # else:
        #     target_grad_norm = get_gard_norm(self.policy.actor.target_unit_head.parameters())
        # self.policy.target_optimizer.step()
        #
        # location
        surr1_location = imp_weights_location * adv_targ
        surr2_location = torch.clamp(imp_weights_location, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_location_loss = (-torch.sum(location_factor * torch.min(surr1_location, surr2_location),
                                         dim=-1,
                                         keepdim=True) * active_masks).sum() / active_masks.sum()
        # self.policy.location_optimizer.zero_grad()
        location_loss = (policy_location_loss - location_entropy * self.entropy_coef)
        # location_loss.backward()
        # if self._use_max_grad_norm:
        #     location_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.location_head.parameters(), self.max_grad_norm)
        # else:
        #     location_grad_norm = get_gard_norm(self.policy.actor.location_head.parameters())
        # self.policy.location_optimizer.step()

        self.actor_optimizer.zero_grad()
        loss = level1_action_loss + target_loss + location_loss
        loss.backward()
        if self._use_max_grad_norm:
            action_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            action_grad_norm = get_gard_norm(self.policy.actor.parameters())
        # 这里不仅更新了actor，还更新了encoder
        self.actor_optimizer.step()

        # value
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks)
        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.critic_optimizer.step()

        # self.encoder_optimizer.step()

        return value_loss, critic_grad_norm, action_grad_norm, \
               level1_entropy, target_entropy, location_entropy, imp_weights_level1, imp_weights_target, imp_weights_location, \
               level1_action_loss, target_loss, location_loss


    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def train(self, buffer, update_actor=True):
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['level1_policy_loss'] = 0
        train_info['target_loss'] = 0
        train_info['location_loss'] = 0
        train_info['level1_entropy'] = 0
        train_info['target_entropy'] = 0
        train_info['location_entropy'] = 0
        train_info['level1_grad_norm'] = 0
        train_info['target_grad_norm'] = 0
        train_info['location_grad_norm'] = 0
        train_info['action_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio_level1'] = 0
        train_info['ratio_target'] = 0
        train_info['ratio_location'] = 0



        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, action_grad_norm, \
                level1_entropy, target_entropy, location_entropy, imp_weights_level1, imp_weights_target, imp_weights_location,\
                level1_action_loss, target_loss, location_loss= self.ppo_update(
                    sample, update_actor=update_actor, agent_id=buffer.agent_id
                )
                train_info['value_loss'] += value_loss.item()
                train_info['level1_policy_loss'] += level1_action_loss.item()
                train_info['target_loss'] += target_loss.item()
                train_info['location_loss'] += location_loss.item()
                train_info['level1_entropy'] += level1_entropy.item()
                train_info['target_entropy'] += target_entropy.item()
                train_info['location_entropy'] += location_entropy.item()
                train_info['action_grad_norm'] += action_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio_level1'] += imp_weights_level1.mean()
                train_info['ratio_target'] += imp_weights_target.mean()
                train_info['ratio_location'] += imp_weights_location.mean()


        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

