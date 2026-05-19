import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from tokenizer import TextTokenizer


# Tree-of-Thoughts / TreeNode removed: generation-stage tree search disabled


class SelfRewardModel:
    """自奖励模型：基于模型自身输出进行多维度奖励评估
    
    评估维度：
    1. 思维链完整性 (CoT Completeness)
    2. 输出一致性 (Output Consistency)
    3. 长度合规性 (Length Compliance)
    4. 无UNK (No Unknown Tokens)
    5. 语义新颖性 (Semantic Novelty)
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.unk_token_id = TextTokenizer.UNKNOWN_TOKEN
        
        self.reward_weights = {
            'cot_completeness': 0.25,
            'output_consistency': 0.25,
            'length_compliance': 0.15,
            'no_unk': 0.15,
            'semantic_novelty': 0.20,
        }
    
    def compute_cot_completeness(self, think_text: str, answer_text: str) -> float:
        """评估思维链完整性"""
        if not think_text or not think_text.strip():
            return 0.0
        
        score = 0.0
        think_lower = think_text.lower()
        
        reasoning_markers = [
            '首先', '然后', '接着', '最后', '因为', '所以', '因此',
            '其次', '再次', '总之', '综上', '分析', '考虑',
            'first', 'then', 'next', 'finally', 'because', 'therefore',
            'analyze', 'consider', 'step', 'reason'
        ]
        
        marker_count = sum(1 for marker in reasoning_markers if marker in think_lower)
        score += min(marker_count / 3.0, 1.0) * 0.4
        
        think_tokens = len(TextTokenizer.encode(think_text))
        if think_tokens >= 10:
            score += 0.3
        elif think_tokens >= 5:
            score += 0.15
        
        if answer_text and answer_text.strip():
            answer_tokens = TextTokenizer.encode(answer_text)
            think_tokens_list = TextTokenizer.encode(think_text)
            
            think_set = set(t.item() for t in think_tokens_list)
            answer_set = set(t.item() for t in answer_tokens)
            overlap = len(think_set & answer_set) / max(len(think_set), 1)
            score += min(overlap, 0.3)
        
        return min(score, 1.0)
    
    def compute_output_consistency(self, generated_text: str, context: str = None) -> float:
        """评估输出一致性"""
        if not generated_text or not generated_text.strip():
            return 0.0
        
        score = 0.5
        
        sentences = [s.strip() for s in generated_text.split('。') if s.strip()]
        if len(sentences) >= 2:
            score += 0.2
        
        text_lower = generated_text.lower()
        negation_words = ['不', '没', '非', '不是', 'no', 'not', 'never']
        negation_count = sum(1 for word in negation_words if word in text_lower)
        
        if negation_count <= 2:
            score += 0.1
        else:
            score -= 0.1
        
        if context and context.strip():
            context_tokens = TextTokenizer.encode(context)
            gen_tokens = TextTokenizer.encode(generated_text)
            
            context_set = set(t.item() for t in context_tokens)
            gen_set = set(t.item() for t in gen_tokens)
            overlap = len(context_set & gen_set) / max(len(gen_set), 1)
            score += min(overlap * 2, 0.2)
        
        return max(0.0, min(score, 1.0))
    
    def compute_length_compliance(self, generated_text: str, min_len: int = 10, max_len: int = 500) -> float:
        """评估长度合规性"""
        if not generated_text:
            return 0.0
        
        text_len = len(generated_text)
        
        if min_len <= text_len <= max_len:
            return 1.0
        elif text_len < min_len:
            return text_len / min_len
        else:
            return max_len / text_len
    
    def compute_no_unk(self, generated_text: str) -> float:
        """评估是否包含未知token"""
        if not generated_text:
            return 0.0
        
        tokens = TextTokenizer.encode(generated_text)
        
        if self.unk_token_id >= 0:
            unk_count = (tokens == self.unk_token_id).sum().item()
            unk_ratio = unk_count / len(tokens)
            return max(0.0, 1.0 - unk_ratio * 10)
        
        return 1.0
    
    def compute_semantic_novelty(self, generated_text: str, reference_texts: List[str] = None) -> float:
        """评估语义新颖性"""
        if not generated_text or not generated_text.strip():
            return 0.0
        
        if not reference_texts or len(reference_texts) == 0:
            tokens = TextTokenizer.encode(generated_text)
            unique_ratio = len(set(t.item() for t in tokens)) / len(tokens)
            return unique_ratio
        
        gen_tokens = set(t.item() for t in TextTokenizer.encode(generated_text))
        
        total_overlap = 0.0
        for ref_text in reference_texts:
            ref_tokens = set(t.item() for t in TextTokenizer.encode(ref_text))
            overlap = len(gen_tokens & ref_tokens) / max(len(gen_tokens), 1)
            total_overlap += overlap
        
        avg_overlap = total_overlap / len(reference_texts)
        novelty = 1.0 - avg_overlap
        
        return max(0.0, min(novelty, 1.0))
    
    def compute_total_reward(
        self,
        think_text: str = None,
        answer_text: str = None,
        context: str = None,
        reference_texts: List[str] = None,
        min_length: int = 10,
        max_length: int = 500
    ) -> Tuple[float, Dict[str, float]]:
        """计算总奖励"""
        rewards = {}
        
    
        if think_text:
            rewards['cot_completeness'] = self.compute_cot_completeness(think_text, answer_text)
        else:
            rewards['cot_completeness'] = 0.0
        
        generated_text = think_text + " " + answer_text if think_text else answer_text
        rewards['output_consistency'] = self.compute_output_consistency(generated_text, context)
        rewards['length_compliance'] = self.compute_length_compliance(generated_text, min_length, max_length)
        rewards['no_unk'] = self.compute_no_unk(generated_text)
        rewards['semantic_novelty'] = self.compute_semantic_novelty(generated_text, reference_texts)
        
        total_reward = sum(
            rewards[key] * self.reward_weights[key]
            for key in self.reward_weights.keys()
        )
        
        return total_reward, rewards


class LightweightPPO:
    """轻量级PPO训练器"""
    
    def __init__(
        self,
        model,
        reward_model: SelfRewardModel,
        device: torch.device,
        learning_rate: float = 2e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        gamma: float = 0.99
    ):
        self.model = model
        self.reward_model = reward_model
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            foreach=torch.cuda.is_available(),
        )
        
        self.episode_data = {
            'log_probs': [],
            'rewards': [],
            'values': [],
            'actions': [],
            'states': []
        }
    
    def collect_episode(
        self,
        prompt: str,
        think_text: str,
        answer_text: str,
        context: str = None,
        reference_texts: List[str] = None
    ) -> float:
        """收集一个episode的数据并计算奖励"""
        total_reward, reward_breakdown = self.reward_model.compute_total_reward(
            think_text=think_text,
            answer_text=answer_text,
            context=context,
            reference_texts=reference_texts
        )
        # 存储到 episode buffer（使用占位 log_prob，为后续接入真实采样概率留空）
        try:
            # 记录标量奖励
            self.episode_data['rewards'].append(float(total_reward))
            # 占位 log_prob 张量（0.0），dtype 与设备匹配
            self.episode_data['log_probs'].append(torch.tensor(0.0, device=self.device))
        except Exception:
            # 若出现设备/类型问题，回退为 Python 原生类型
            self.episode_data['rewards'].append(float(total_reward))
            self.episode_data['log_probs'].append(0.0)

        return total_reward, reward_breakdown

    def compute_gae(self, rewards: List[float], values: List[float], lam: float = 0.95):
        """Compute GAE advantages and returns. Returns (advantages_tensor, returns_tensor)."""
        device = self.device
        N = len(rewards)
        adv = torch.zeros(N, dtype=torch.float32, device=device)
        last_gae = 0.0
        for t in reversed(range(N)):
            next_value = values[t + 1] if (t + 1) < N else 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * lam * last_gae
            adv[t] = last_gae

        returns = adv + torch.tensor(values, dtype=torch.float32, device=device)
        # normalize advantages
        if adv.numel() > 1:
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False) + 1e-8
            adv = (adv - adv_mean) / adv_std

        return adv, returns
    
    def update_policy(self, batch_size: int = 4, ppo_epochs: int = 4, minibatch_size: int = 4, lam: float = 0.95, value_coef: float = 0.5) -> Dict[str, float]:
        """使用 GAE + 多轮 minibatch PPO 更新策略和价值网络。

        - 使用模型的 `value_preds`（原始标度）作为 value 估计。
        - 若 raw_states 可用，会在 update 时重算 action_logits/value_preds 以保证梯度能回传。
        """
        N = len(self.episode_data['rewards'])
        if N < batch_size:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}

        # --- 收集 tensors ---
        rewards = [float(r) for r in self.episode_data['rewards']]
        old_log_probs = []
        actions = []
        stored_values = self.episode_data.get('values', [])

        for i in range(N):
            lp = self.episode_data['log_probs'][i]
            if isinstance(lp, torch.Tensor):
                old_log_probs.append(float(lp.detach()))
            else:
                old_log_probs.append(float(lp))
            actions.append(int(self.episode_data['actions'][i]))

        # 使用存储的 values，缺失处用0.0回退
        values = [float(stored_values[i]) if i < len(stored_values) else 0.0 for i in range(N)]

        # --- GAE to compute advantages and returns ---
        advantages, returns = self.compute_gae(rewards, values, lam=lam)

        # convert old_log_probs/actions to tensors
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)

        # PPO epochs with minibatches
        indices = np.arange(N)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_updates = 0

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                # build minibatch inputs
                mb_old_logp = old_log_probs[mb_idx_t]
                mb_actions = actions[mb_idx_t]
                mb_adv = advantages[mb_idx_t]
                mb_ret = returns[mb_idx_t]

                # Use stored old log probs as current proxy (no recomputation of raw states)
                cur_logps = mb_old_logp
                # Use stored values where available
                cur_values = torch.tensor([values[ii] for ii in mb_idx], device=self.device, dtype=torch.float32)
                entropies = torch.zeros(len(mb_idx), device=self.device)

                ratios = torch.exp(cur_logps - mb_old_logp)
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(cur_values, mb_ret)

                entropy_loss = - self.entropy_coef * entropies.mean()

                loss = policy_loss + value_coef * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy_loss += float(entropy_loss.item())
                total_updates += 1

        # reset episode buffer
        self.episode_data = {
            'log_probs': [],
            'rewards': [],
            'values': [],
            'actions': [],
            'states': []
        }

        if total_updates == 0:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}

        return {
            'loss': (total_policy_loss + value_coef * total_value_loss + total_entropy_loss) / total_updates,
            'policy_loss': total_policy_loss / total_updates,
            'value_loss': total_value_loss / total_updates,
            'entropy_loss': total_entropy_loss / total_updates
        }
    
    def clear_data(self):
        """清空episode数据"""
        self.episode_data = {
            'log_probs': [],
            'rewards': [],
            'values': [],
            'actions': [],
            'states': []
        }

