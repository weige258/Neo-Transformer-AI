import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from tokenizer import TextTokenizer


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
    
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """计算优势函数"""
        advantages = []
        returns = 0
        
        for reward in reversed(rewards):
            returns = reward + self.gamma * returns
            advantages.insert(0, returns)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.tolist()
    
    def update_policy(self, batch_size: int = 4) -> Dict[str, float]:
        """更新策略网络"""
        if len(self.episode_data['rewards']) < batch_size:
            return {'loss': 0.0, 'policy_loss': 0.0, 'entropy_loss': 0.0}
        
        advantages = self.compute_advantages(self.episode_data['rewards'])
        
        reward_threshold = sorted(self.episode_data['rewards'])[len(self.episode_data['rewards']) // 2]
        high_reward_indices = [
            i for i, r in enumerate(self.episode_data['rewards'])
            if r >= reward_threshold
        ]
        
        if len(high_reward_indices) == 0:
            high_reward_indices = list(range(len(self.episode_data['rewards'])))
        
        total_loss = None
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        update_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        for idx in high_reward_indices:
            log_prob = self.episode_data['log_probs'][idx]
            advantage = advantages[idx]

            # Ensure tensors
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(log_prob, device=self.device, dtype=torch.float32)
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, device=self.device, dtype=torch.float32)

            ratio = torch.exp(log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage

            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = -log_prob.mean()
            entropy_loss = -self.entropy_coef * entropy

            loss = policy_loss + entropy_loss

            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            update_count += 1

        if update_count > 0 and total_loss is not None:
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
        
        self.episode_data = {
            'log_probs': [],
            'rewards': [],
            'values': [],
            'actions': [],
            'states': []
        }
        
        return {
            'loss': total_loss / max(update_count, 1),
            'policy_loss': total_policy_loss / max(update_count, 1),
            'entropy_loss': total_entropy_loss / max(update_count, 1)
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
