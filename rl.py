import logging
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass
from tokenizer import TextTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """树节点"""
    token_id: int
    log_prob: float
    reward: float = 0.0
    cumulative_reward: float = 0.0
    children: List['TreeNode'] = None
    parent: 'TreeNode' = None
    visit_count: int = 0
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def get_path(self) -> List[int]:
        """获取从根节点到当前节点的路径"""
        path = []
        node = self
        while node is not None:
            path.append(node.token_id)
            node = node.parent
        return path[::-1][1:]


class SelfRewardModel:
    """自奖励模型：基于手工规则的多维度奖励评估。

    评估维度：
    1. 思维链完整性 (CoT Completeness)
    2. 输出一致性 (Output Consistency)
    3. 长度合规性 (Length Compliance)
    4. 无UNK (No Unknown Tokens)
    5. 语义新颖性 (Semantic Novelty)
    
    此模块始终启用，在每次训练步骤中自动计算奖励并更新策略。
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
        
        # 【优化】基于最新研究调整奖励阈值参数
        # 参考: "当奖励信号质量不足时，暂停 RL，仅使用 SFT"
        #       "奖励过低会导致训练不稳定，建议阈值 0.3-0.5"
        self.reward_history: list[float] = []  # 最近N次奖励值
        self.reward_window_size = 50           # 【优化】滑动窗口大小从20增加到50，更稳定
        self.min_reward_threshold = 0.35       # 【优化】最低平均奖励阈值从0.3提高到0.35
        self.min_reward_growth = 0.005         # 【优化】最低增长率从0.01降低到0.005，更宽容
        self.min_reward_std_threshold = 0.25   # 【优化】最大允许标准差从0.3降低到0.25，更严格
    
    def should_enable_rl(self) -> tuple[bool, str]:
        """智能决策是否启用 RL 训练
        
        基于 SuperRL 的 Adaptive Switch 设计：
        - 监控奖励密度、增长率、稳定性
        - 当奖励信号质量不足时，暂停 RL，仅使用 SFT
        - 零额外开销：复用已有的奖励计算结果
        
        Returns:
            (should_enable: bool, reason: str)
        """
        if len(self.reward_history) < 10:
            # 初始阶段：数据不足，启用 RL 探索
            return True, "初始探索阶段"
        
        # 取最近N次的奖励
        recent_rewards = self.reward_history[-self.reward_window_size:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        # 计算奖励增长率（线性回归斜率）
        if len(recent_rewards) >= 10:
            n = len(recent_rewards)
            x_mean = (n - 1) / 2
            y_mean = avg_reward
            numerator = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(recent_rewards))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            growth_rate = numerator / max(denominator, 1e-8)
        else:
            growth_rate = 0.0
        
        # 计算奖励标准差（稳定性指标）
        variance = sum((r - avg_reward) ** 2 for r in recent_rewards) / len(recent_rewards)
        std_reward = variance ** 0.5
        
        # 决策逻辑
        reasons = []
        
        # 条件1：平均奖励过低（信号稀疏）
        if avg_reward < self.min_reward_threshold:
            reasons.append(f"奖励过低({avg_reward:.3f}<{self.min_reward_threshold})")
            return False, f"暂停RL: {'; '.join(reasons)} - 转为纯SFT训练"
        
        # 条件2：奖励持续下降（模型未收敛）
        if growth_rate < -self.min_reward_growth:
            reasons.append(f"奖励下降(增长率{growth_rate:.4f})")
            return False, f"暂停RL: {'; '.join(reasons)} - 转为纯SFT训练"
        
        # 条件3：奖励波动过大（训练不稳定）
        if std_reward > self.min_reward_std_threshold:
            reasons.append(f"奖励波动大(σ={std_reward:.3f})")
            return False, f"暂停RL: {'; '.join(reasons)} - 转为纯SFT训练"
        
        # 条件4：奖励适中且稳定，适合 RL
        return True, f"启用RL: 平均奖励={avg_reward:.3f}, 增长率={growth_rate:.4f}, σ={std_reward:.3f}"
    
    def record_reward(self, reward: float):
        """记录奖励值到历史"""
        self.reward_history.append(reward)
        # 保持窗口大小
        if len(self.reward_history) > self.reward_window_size * 2:
            self.reward_history = self.reward_history[-self.reward_window_size * 2:]
    
    def compute_cot_completeness(self, think_text: str, answer_text: str) -> float:
        """评估思维链完整性
        
        【修复缺陷五】防止奖励黑客：不仅检查关键词存在，还检查关键词的合理分布和多样性
        避免模型通过简单堆砌关键词（如"首先首先然后因为所以"）获取高分
        """
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
        
        # 【修复】计算关键词的多样性而非简单计数
        # 防止模型通过重复相同关键词刷分
        found_markers = [marker for marker in reasoning_markers if marker in think_lower]
        unique_marker_count = len(set(found_markers))  # 唯一关键词数量
        total_marker_count = len(found_markers)  # 总出现次数
        
        # 多样性得分：唯一关键词数量 / 总关键词数量
        # 如果所有关键词都不同，得分为1；如果大量重复，得分接近0
        diversity_score = unique_marker_count / max(total_marker_count, 1)
        
        # 覆盖率得分：找到的唯一关键词数量 / 总关键词库大小
        coverage_score = unique_marker_count / len(reasoning_markers)
        
        # 综合得分：多样性和覆盖率的加权平均
        marker_score = 0.6 * diversity_score + 0.4 * min(coverage_score * 3, 1.0)
        score += marker_score * 0.4
        
        # 检查关键词是否合理分布（不应集中在文本开头或结尾）
        if len(think_lower) > 20:
            third_len = len(think_lower) // 3
            first_third = think_lower[:third_len]
            last_third = think_lower[-third_len:]
            
            # 计算每个三分之一段落中的关键词数量
            markers_in_parts = [
                sum(1 for marker in reasoning_markers if marker in first_third),
                sum(1 for marker in reasoning_markers if marker in think_lower[third_len:2*third_len]),
                sum(1 for marker in reasoning_markers if marker in last_third)
            ]
            
            # 如果关键词均匀分布在三个段落中，给予奖励
            if all(m > 0 for m in markers_in_parts):
                score += 0.15  # 均匀分布奖励
            elif sum(1 for m in markers_in_parts if m > 0) >= 2:
                score += 0.05  # 至少两段有关键词
        
        # 长度合理性检查
        think_tokens = len(TextTokenizer.encode(think_text))
        if 10 <= think_tokens <= 200:  # 合理范围
            score += 0.3
        elif 5 <= think_tokens < 10 or 200 < think_tokens <= 500:
            score += 0.15
        elif think_tokens > 500:  # 过长惩罚
            score += 0.05
        
        # 思维链与答案的相关性检查（不应完全重合）
        if answer_text and answer_text.strip():
            answer_tokens = TextTokenizer.encode(answer_text)
            think_tokens_list = TextTokenizer.encode(think_text)
            
            think_set = set(t.item() for t in think_tokens_list)
            answer_set = set(t.item() for t in answer_tokens)
            overlap = len(think_set & answer_set) / max(len(think_set), 1)
            # 适度重叠是好的（0.2-0.6），过高或过低都不好
            if 0.2 <= overlap <= 0.6:
                score += 0.15
            elif 0.1 <= overlap < 0.2 or 0.6 < overlap <= 0.8:
                score += 0.05
        
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
        """计算总奖励并自动记录到历史"""
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
        
        # 【智能切换】自动记录奖励到历史
        self.record_reward(total_reward)
        
        return total_reward, rewards


class LightweightPPO:
    """轻量级PPO训练器"""
    
    def __init__(
        self,
        model,
        reward_model: SelfRewardModel,
        device: torch.device,
        learning_rate: float = 5e-7,           # 【优化】默认学习率降低到5e-7
        min_learning_rate: float = 1e-8,       # 【优化】最小学习率降低到1e-8
        warmup_steps: int = 200,               # 【优化】warmup步数增加到200
        total_training_steps: int = 30000,
        clip_ratio: float = 0.2,               # 保持标准值0.2
        entropy_coef: float = 0.02,            # 【优化】熵系数从0.01增加到0.02，增强探索
        gamma: float = 0.99,                   # 保持标准折扣因子
        ppo_epochs: int = 2,                   # 【新增】PPO epoch数（基于研究推荐）
        mini_batch_num: int = 4                # 【新增】mini-batch数量（基于研究推荐）
    ):
        self.model = model
        self.reward_model = reward_model
        self.device = device
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        
        # 【新增】保存 PPO epoch 和 mini-batch 参数
        self.ppo_epochs = ppo_epochs
        self.mini_batch_num = mini_batch_num
        
        # 【PPO学习率配置】
        self.base_learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps
        self.ppo_training_steps = 0  # PPO训练步数计数器
        
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
            'states': [],
            'prompts': [],
            'generated_texts': []
        }
    
    def get_ppo_learning_rate(self, current_step: int) -> float:
        """计算PPO当前学习率（支持Warmup + Cosine Decay）
        
        Args:
            current_step: 当前PPO训练步数
            
        Returns:
            当前学习率
        """
        import math
        
        if current_step < self.warmup_steps:
            # Warmup阶段：线性增长
            warmup_progress = current_step / max(self.warmup_steps - 1, 1)
            return self.min_learning_rate + (self.base_learning_rate - self.min_learning_rate) * warmup_progress
        else:
            # Cosine Decay
            progress = (current_step - self.warmup_steps) / max(self.total_training_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_learning_rate + (self.base_learning_rate - self.min_learning_rate) * cosine_decay
    
    def apply_ppo_learning_rate(self):
        """应用学习率到PPO优化器"""
        lr = self.get_ppo_learning_rate(self.ppo_training_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def collect_episode(
        self,
        prompt: str,
        think_text: str,
        answer_text: str,
        context: str = None,
        reference_texts: List[str] = None,
        model_log_probs: torch.Tensor | None = None,
    ) -> float:
        """收集一个episode的数据并计算奖励。

        Args:
            prompt: 输入提示
            think_text: 思维链文本
            answer_text: 回答文本
            context: 上下文
            reference_texts: 参考文本列表
            model_log_probs: 模型输出的对数概率（可选）。若为 None，
                             将在update时通过当前策略重新计算。
        """
        total_reward, reward_breakdown = self.reward_model.compute_total_reward(
            think_text=think_text,
            answer_text=answer_text,
            context=context,
            reference_texts=reference_texts
        )

        self.episode_data['rewards'].append(float(total_reward))

        # 保存生成的文本内容，用于后续重新计算current_log_prob
        generated_text = ""
        if think_text:
            generated_text += think_text
        if answer_text:
            if generated_text:
                generated_text += " "
            generated_text += answer_text
        
        # 保存prompt和generated_text用于后续计算
        self.episode_data['prompts'].append(prompt)
        self.episode_data['generated_texts'].append(generated_text)

        if model_log_probs is not None:
            # 使用模型真实输出的对数概率
            if isinstance(model_log_probs, torch.Tensor):
                self.episode_data['log_probs'].append(
                    model_log_probs.detach().to(self.device).mean()
                )
            else:
                self.episode_data['log_probs'].append(
                    torch.tensor(float(model_log_probs), device=self.device)
                )
        else:
            # 使用占位值，将在update_policy时重新计算真实的current_log_prob
            self.episode_data['log_probs'].append(torch.tensor(0.0, device=self.device))

        return total_reward, reward_breakdown
    
    def _compute_current_log_prob(self, prompt: str, generated_text: str) -> torch.Tensor:
        """计算当前策略下生成文本的对数概率
        
        Args:
            prompt: 输入提示文本
            generated_text: 生成的文本
            
        Returns:
            平均对数概率值
        """
        if not generated_text or not generated_text.strip():
            return torch.tensor(0.0, device=self.device)
        
        # 编码prompt和generated_text
        prompt_tokens = TextTokenizer.encode(prompt).to(self.device)
        generated_tokens = TextTokenizer.encode(generated_text).to(self.device)
        
        if generated_tokens.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 拼接完整的输入序列
        full_sequence = torch.cat([prompt_tokens, generated_tokens])
        
        # 前向传播获取logits
        with torch.set_grad_enabled(True):
            result = self.model(full_sequence, use_cache=False)
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result
        
        # 计算logits对应的log_prob
        # logits shape: (seq_len, vocab_size)
        # 我们需要计算generated_tokens中每个token的log_prob
        # 对于自回归模型，位置i的logit预测位置i+1的token
        if len(logits) < len(generated_tokens) + 1:
            return torch.tensor(0.0, device=self.device)
        
        # 提取对应generated_tokens位置的logits
        # prompt有P个token，generated有G个token
        # logits[P:P+G]对应预测generated_tokens的logits
        prompt_len = len(prompt_tokens)
        generated_logits = logits[prompt_len:prompt_len + len(generated_tokens)]
        
        # 计算log_softmax
        log_probs = F.log_softmax(generated_logits, dim=-1)
        
        # 提取实际generated_tokens对应的log_prob
        # log_probs[i, generated_tokens[i]]就是第i个token的log_prob
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # 返回平均log_prob
        return token_log_probs.mean()
    
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
        """更新策略网络
        
        修复：正确计算PPO的重要性采样比率
        ratio = exp(current_log_prob - old_log_prob)
        """
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
            old_log_prob = self.episode_data['log_probs'][idx]
            advantage = advantages[idx]

            # Ensure tensors
            if not isinstance(old_log_prob, torch.Tensor):
                old_log_prob = torch.tensor(old_log_prob, device=self.device, dtype=torch.float32)
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, device=self.device, dtype=torch.float32)

            # 修复：使用当前策略重新计算current_log_prob
            # 这是PPO算法的核心：ratio = exp(log π_new(a|s) - log π_old(a|s))
            prompt = self.episode_data['prompts'][idx]
            generated_text = self.episode_data['generated_texts'][idx]
            
            if prompt and generated_text:
                # 通过当前模型重新前向传播计算真实的current_log_prob
                current_log_prob = self._compute_current_log_prob(prompt, generated_text)
                
                # 计算importance sampling ratio
                # ratio = π_new(a|s) / π_old(a|s) = exp(log π_new(a|s) - log π_old(a|s))
                ratio = torch.exp(current_log_prob - old_log_prob.detach())
                
                # PPO clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage

                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus for exploration
                entropy = -current_log_prob.mean()
                entropy_loss = -self.entropy_coef * entropy

                loss = policy_loss + entropy_loss
            else:
                # 如果缺少prompt或generated_text，跳过此样本
                print(f"[Warning] PPO更新: 样本{idx}缺少prompt或generated_text，跳过", flush=True)
                continue

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

            # 【PPO学习率调度】在更新前应用学习率
            current_lr = self.apply_ppo_learning_rate()
            self.ppo_training_steps += 1

            self.optimizer.step()
            
            # 每100步打印一次学习率信息
            if self.ppo_training_steps % 100 == 0:
                print(f"[PPO] Step {self.ppo_training_steps}, Current LR: {current_lr:.2e}, "
                      f"Base LR: {self.base_learning_rate:.2e}", flush=True)
        
        # 清空episode数据
        self.episode_data = {
            'log_probs': [],
            'rewards': [],
            'values': [],
            'actions': [],
            'states': [],
            'prompts': [],
            'generated_texts': []
        }
        
        return {
            'loss': total_loss.item() if total_loss is not None else 0.0,
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
            'states': [],
            'prompts': [],
            'generated_texts': []
        }


class TreeReinforcementLearning:
    """树强化学习生成器"""
    
    def __init__(
        self,
        model,
        reward_model: SelfRewardModel,
        device: torch.device,
        max_depth: int = 100,
        beam_width: int = 4,
        exploration_coef: float = 1.0,
        temperature: float = 0.7
    ):
        self.model = model
        self.reward_model = reward_model
        self.device = device
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.exploration_coef = exploration_coef
        self.temperature = temperature
        
        self.root = TreeNode(token_id=None, log_prob=0.0)
    
    def select_node(self, node: TreeNode) -> TreeNode:
        """使用UCB算法选择节点"""
        if not node.children:
            return node
        
        def ucb_score(child: TreeNode) -> float:
            if child.visit_count == 0:
                return float('inf')
            
            exploitation = child.cumulative_reward / child.visit_count
            exploration = self.exploration_coef * torch.sqrt(
                torch.log(torch.tensor(node.visit_count + 1)) / child.visit_count
            ).item()
            
            return exploitation + exploration
        
        selected = max(node.children, key=ucb_score)
        return self.select_node(selected)
    
    def expand_node(
        self,
        node: TreeNode,
        prompt_tokens: torch.Tensor,
        current_tokens: List[int],
        context: str = None
    ) -> List[TreeNode]:
        """扩展节点，生成候选子节点"""
        if node.depth >= self.max_depth:
            return []
        
        if current_tokens:
            current_tokens_tensor = torch.tensor(current_tokens, device=self.device, dtype=torch.long)
        else:
            current_tokens_tensor = torch.tensor([], device=self.device, dtype=torch.long)
        input_tokens = torch.cat([prompt_tokens, current_tokens_tensor])
        
        with torch.inference_mode():
            result = self.model(input_tokens, use_cache=True)
            if isinstance(result, tuple):
                logits, _ = result
            else:
                logits = result
        
        next_logits = logits[-1]
        next_probs = F.softmax(next_logits / self.temperature, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(next_logits, k=self.beam_width)
        
        new_children = []
        for i in range(self.beam_width):
            token_id = top_k_indices[i].item()
            log_prob = torch.log(top_k_probs[i] + 1e-10).item()
            
            child = TreeNode(
                token_id=token_id,
                log_prob=log_prob,
                parent=node,
                depth=node.depth + 1
            )
            new_children.append(child)
        
        node.children = new_children
        return new_children
    
    def evaluate_node(
        self,
        node: TreeNode,
        prompt_tokens: torch.Tensor,
        current_tokens: List[int],
        think_tokens: List[int] = None,
        context: str = None
    ) -> float:
        """评估节点的奖励值"""
        full_tokens = prompt_tokens.tolist() + current_tokens
        
        generated_text = TextTokenizer.decode(torch.tensor(full_tokens))
        
        think_text = None
        answer_text = generated_text
        
        if think_tokens is not None:
            think_text = TextTokenizer.decode(torch.tensor(think_tokens))
            answer_text = generated_text[len(think_text):]
        
        total_reward, _ = self.reward_model.compute_total_reward(
            think_text=think_text,
            answer_text=answer_text,
            context=context
        )
        
        return total_reward
    
    def backpropagate(self, node: TreeNode, reward: float):
        """反向传播奖励值"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.cumulative_reward += reward
            current = current.parent
    
    def search(
        self,
        prompt: str,
        context: str = None,
        max_iterations: int = 100,
        thinking_available: bool = True
    ) -> Tuple[str, float, Dict[str, float]]:
        """执行树搜索"""
        prompt_tokens = TextTokenizer.encode(prompt).to(self.device)
        
        self.root = TreeNode(token_id=None, log_prob=0.0)
        
        initial_children = self.expand_node(
            self.root,
            prompt_tokens,
            [],
            context
        )
        
        for iteration in range(max_iterations):
            selected_node = self.select_node(self.root)
            
            current_tokens = selected_node.get_path()
            new_children = self.expand_node(
                selected_node,
                prompt_tokens,
                current_tokens,
                context
            )
            
            for child in new_children:
                child_tokens = child.get_path()
                reward = self.evaluate_node(
                    child,
                    prompt_tokens,
                    child_tokens,
                    context=context
                )
                child.reward = reward
                
                self.backpropagate(child, reward)
        
        best_node = self._select_best_node()
        best_tokens = best_node.get_path()
        
        generated_text = TextTokenizer.decode(torch.tensor(best_tokens))
        
        total_reward, reward_breakdown = self.reward_model.compute_total_reward(
            answer_text=generated_text,
            context=context
        )
        
        return generated_text, total_reward, reward_breakdown
    
    def _select_best_node(self) -> TreeNode:
        """选择最佳节点"""
        def collect_leaves(node: TreeNode, leaves: List[TreeNode]):
            if not node.children:
                leaves.append(node)
            else:
                for child in node.children:
                    collect_leaves(child, leaves)
        
        leaves = []
        collect_leaves(self.root, leaves)
        
        if not leaves:
            return self.root
        
        best_leaf = max(
            leaves,
            key=lambda n: n.cumulative_reward / max(n.visit_count, 1)
        )
        
        return best_leaf
    
    def beam_search_with_reward(
        self,
        prompt: str,
        context: str = None,
        max_length: int = 100,
        beam_width: int = 4,
        thinking_available: bool = True
    ) -> Tuple[str, float, Dict[str, float]]:
        """基于奖励的束搜索"""
        prompt_tokens = TextTokenizer.encode(prompt).to(self.device)
        
        beams = [
            {
                'tokens': [],
                'log_prob': 0.0,
                'reward': 0.0,
                'finished': False
            }
        ]
        
        for step in range(max_length):
            new_beams = []
            
            for beam in beams:
                if beam['finished']:
                    new_beams.append(beam)
                    continue
                
                if beam['tokens']:
                    beam_tokens = torch.tensor(beam['tokens'], device=self.device, dtype=torch.long)
                else:
                    beam_tokens = torch.tensor([], device=self.device, dtype=torch.long)
                current_tokens = torch.cat([
                    prompt_tokens,
                    beam_tokens
                ])
                
                with torch.inference_mode():
                    result = self.model(current_tokens, use_cache=True)
                    if isinstance(result, tuple):
                        logits, _ = result
                    else:
                        logits = result
                
                next_logits = logits[-1]
                next_probs = F.softmax(next_logits / self.temperature, dim=-1)
                
                top_k_probs, top_k_indices = torch.topk(next_logits, k=beam_width)
                
                for i in range(beam_width):
                    token_id = top_k_indices[i].item()
                    log_prob = torch.log(top_k_probs[i] + 1e-10).item()
                    
                    if token_id == TextTokenizer.END_GENERATION_TOKEN:
                        finished = True
                    else:
                        finished = False
                    
                    new_beam = {
                        'tokens': beam['tokens'] + [token_id],
                        'log_prob': beam['log_prob'] + log_prob,
                        'reward': 0.0,
                        'finished': finished
                    }
                    
                    if finished or step == max_length - 1:
                        full_tokens = prompt_tokens.tolist() + new_beam['tokens']
                        generated_text = TextTokenizer.decode(torch.tensor(full_tokens))
                        
                        total_reward, _ = self.reward_model.compute_total_reward(
                            answer_text=generated_text,
                            context=context
                        )
                        new_beam['reward'] = total_reward
                    
                    new_beams.append(new_beam)
            
            beams = sorted(
                new_beams,
                key=lambda b: b['log_prob'] + b['reward'],
                reverse=True
            )[:beam_width]
            
            if all(beam['finished'] for beam in beams):
                break
        
        best_beam = max(beams, key=lambda b: b['log_prob'] + b['reward'])
        best_tokens = best_beam['tokens']
        
        generated_text = TextTokenizer.decode(torch.tensor(best_tokens))
        
        total_reward, reward_breakdown = self.reward_model.compute_total_reward(
            answer_text=generated_text,
            context=context
        )
        
        return generated_text, total_reward, reward_breakdown
