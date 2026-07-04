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
            'cot_completeness': 0.20,
            'output_consistency': 0.20,
            'length_compliance': 0.10,
            'no_unk': 0.10,
            'semantic_novelty': 0.15,
            'answer_completeness': 0.25,   # 【新增】回答完整性：严惩"只有思维链无回答"的作弊行为
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
    
    @staticmethod
    def evaluate_generation_completeness(tokens: list) -> float:
        """检查生成序列的完整性：思维链结束后是否有实际回答内容。
        
        这是对抗"奖励作弊"（Reward Hacking）的核心机制。
        模型可能学会只输出思维链就直接结束以获取高奖励，
        此函数检测并严厉惩罚这种行为。
        
        Args:
            tokens: 原始 token ID 列表（包含特殊 token）
            
        Returns:
            惩罚分数：0.0 = 正常，负值 = 无回答（越大越严厉）
        """
        try:
            think_end_idx = tokens.index(TextTokenizer.THINK_END_TOKEN)
        except ValueError:
            # 没有 THINK_END，可能是无思维链的纯回答模式，不惩罚
            return 0.0
        
        # 找到 THINK_END 之后的所有 token
        answer_tokens = tokens[think_end_idx + 1:]
        
        # 过滤掉特殊 token（END_GENERATION, UNKNOWN, START_GENERATION 等）
        # 只保留普通字符 token（id > 6 且有效）
        valid_answer_tokens = [
            t for t in answer_tokens 
            if t > 6 and TextTokenizer._is_valid_token(t)
        ]
        
        # 如果思维链结束后没有有效回答内容 → 毁灭性惩罚
        if len(valid_answer_tokens) == 0:
            return -10.0  # 极大负奖励，彻底扼杀作弊动机
        
        # 回答太短（少于5个有效token）→ 适度惩罚
        if len(valid_answer_tokens) < 5:
            return -2.0
        
        return 0.0
    
    def compute_answer_completeness(
        self, 
        think_text: str = None, 
        answer_text: str = None,
        full_tokens: list = None
    ) -> float:
        """评估回答完整性：思维链之后是否包含有意义的回答。
        
        这是防止 PPO 策略网络作弊的关键维度。
        如果模型只输出思维链就直接结束，将受到严厉惩罚。
        
        Args:
            think_text: 思维链文本
            answer_text: 回答文本
            full_tokens: 完整的 token ID 序列（可选，用于精确检查）
            
        Returns:
            0.0 ~ 1.0，无回答时返回 0.0
        """
        # 如果有 token 序列，使用精确的 token 级别检查
        if full_tokens is not None and len(full_tokens) > 0:
            penalty = self.evaluate_generation_completeness(full_tokens)
            if penalty < 0:
                return 0.0  # 毁灭性惩罚 → 此项奖励为0
        
        # 基于文本的检查
        has_think = think_text and think_text.strip()
        has_answer = answer_text and answer_text.strip()
        
        # 情况1：没有思维链（纯问答模式），不扣分
        if not has_think:
            return 0.5  # 中性分数，不奖励也不惩罚
        
        # 情况2：有思维链但没有回答 → 零分！
        if has_think and not has_answer:
            return 0.0
        
        # 情况3：有思维链且有回答
        if has_think and has_answer:
            score = 0.6  # 基础分
            
            # 回答长度合理性
            answer_len = len(answer_text)
            if answer_len >= 20:
                score += 0.3  # 足够长的回答
            elif answer_len >= 10:
                score += 0.2
            elif answer_len >= 5:
                score += 0.1
            
            # 回答不应是思维链的重复
            if len(think_text) > 0 and len(answer_text) > 0:
                # 简单的字符级重叠检查
                think_chars = set(think_text)
                answer_chars = set(answer_text)
                if len(answer_chars - think_chars) > 0:
                    score += 0.1  # 回答有新内容
            
            return min(score, 1.0)
        
        return 0.5
    
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
        max_length: int = 500,
        full_tokens: list = None  # 【新增】原始token序列，用于精确检测奖励作弊
    ) -> Tuple[float, Dict[str, float]]:
        """计算总奖励并自动记录到历史
        
        Args:
            think_text: 思维链文本
            answer_text: 回答文本
            context: 上下文
            reference_texts: 参考文本列表
            min_length: 最小长度
            max_length: 最大长度
            full_tokens: 【新增】完整token ID序列，用于精确检测"只有思维链无回答"
        """
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
        
        # 【新增】回答完整性评估：严惩"只有思维链无回答"的作弊行为
        rewards['answer_completeness'] = self.compute_answer_completeness(
            think_text=think_text,
            answer_text=answer_text,
            full_tokens=full_tokens
        )
        
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
        mini_batch_num: int = 4,               # 【新增】mini-batch数量（基于研究推荐）
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
        
        # 【修复】PPO使用独立优化器，避免与SFT共享导致梯度污染
        # 原实现共享优化器会导致：动量状态污染、学习率调度冲突、梯度方向混乱
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            foreach=torch.cuda.is_available(),
        )
        self.using_shared_optimizer = False
        
        self.episode_data = {
            'log_probs': [],
            'entropies': [],  # 【修复HIGH #4】逐token分布熵
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
    ) -> Tuple[float, Dict[str, float]]:
        """收集一个 episode 的数据并计算奖励。

        【修复】始终在收集时计算旧策略的逐 token log_prob，
        不再使用 0.0 占位值（之前会导致 ratio 计算完全失效）。

        Args:
            prompt: 输入提示
            think_text: 思维链文本
            answer_text: 回答文本
            context: 上下文
            reference_texts: 参考文本列表
            model_log_probs: 【已弃用】保留参数兼容性，实际会被忽略
        """
        total_reward, reward_breakdown = self.reward_model.compute_total_reward(
            think_text=think_text,
            answer_text=answer_text,
            context=context,
            reference_texts=reference_texts
        )

        self.episode_data['rewards'].append(float(total_reward))

        # 构建完整生成文本
        generated_text = ""
        if think_text:
            generated_text += think_text
        if answer_text:
            if generated_text:
                generated_text += " "
            generated_text += answer_text
        
        self.episode_data['prompts'].append(prompt)
        self.episode_data['generated_texts'].append(generated_text)

        # 【核心修复】在 no_grad + eval 下记录旧策略的逐 token log_prob + 熵
        # 【修复】限制episode数据总量，防止显存泄漏
        max_episodes = int(64)  # 最大存储episode数
        if len(self.episode_data['rewards']) >= max_episodes:
            for key in self.episode_data:
                if isinstance(self.episode_data[key], list):
                    self.episode_data[key] = self.episode_data[key][-max_episodes//2:]
        
        if prompt and generated_text:
            # 【修复】显存保护：跳过过长序列的log_prob计算
            try:
                prompt_tokens = TextTokenizer.encode(prompt).to(self.device)
                generated_tokens = TextTokenizer.encode(generated_text).to(self.device)
                total_tokens = len(prompt_tokens) + len(generated_tokens)
                if total_tokens > 1024:  # 超长序列跳过，防显存爆炸
                    self.episode_data['log_probs'].append(None)
                    self.episode_data['entropies'].append(None)
                    return total_reward, reward_breakdown
            except Exception:
                self.episode_data['log_probs'].append(None)
                self.episode_data['entropies'].append(None)
                return total_reward, reward_breakdown
            
            self.model.eval()
            with torch.no_grad():
                old_token_lps, old_entropies = self._compute_token_log_probs_and_entropy(prompt, generated_text)
            self.model.train()  # 恢复训练模式
            # 【修复】计算完后立即清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if old_token_lps is not None:
                self.episode_data['log_probs'].append(old_token_lps.detach())
                self.episode_data['entropies'].append(old_entropies.detach() if old_entropies is not None else torch.tensor([0.0], device=self.device))
            else:
                self.episode_data['log_probs'].append(
                    torch.tensor([0.0], device=self.device)
                )
                self.episode_data['entropies'].append(
                    torch.tensor([0.0], device=self.device)
                )
        else:
            self.episode_data['log_probs'].append(None)
            self.episode_data['entropies'].append(None)

        return total_reward, reward_breakdown
    
    def _compute_token_log_probs_and_entropy(
        self, prompt: str, generated_text: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """计算当前策略下每个生成 token 的对数概率和分布熵。
        
        这是标准 PPO 所要求的：
        - ratio 必须在每个 token 级别计算和裁剪（逐 token log_prob）
        - 熵必须基于完整的 token 分布计算，而非仅采样 token 的 log_prob
        
        Args:
            prompt: 输入提示文本
            generated_text: 生成的文本
            
        Returns:
            (token_log_probs, entropies):
            - token_log_probs: (G,) 逐 token log_prob
            - entropies: (G,) 逐 token 分布熵
            无法计算时返回 (None, None)
        """
        if not generated_text or not generated_text.strip():
            return None, None
        
        prompt_tokens = TextTokenizer.encode(prompt).to(self.device)
        generated_tokens = TextTokenizer.encode(generated_text).to(self.device)
        
        if generated_tokens.numel() == 0:
            return None, None
        
        full_sequence = torch.cat([prompt_tokens, generated_tokens])
        
        result = self.model(full_sequence, use_cache=False)
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        
        prompt_len = len(prompt_tokens)
        gen_len = len(generated_tokens)
        
        if len(logits) < gen_len + 1:
            return None, None
        
        # ── 修复：向前平移1位，使 logits[i-1] 对应 generated_tokens[i] ──
        # 自回归模型中，预测第t个token的logit位于位置t-1
        # 预测 generated_tokens[0] 的 logit 位于 logits[prompt_len - 1]
        # 预测最后一个生成token的 logit 位于 logits[prompt_len + gen_len - 2]
        generated_logits = logits[prompt_len - 1:prompt_len + gen_len - 1]
        log_probs = F.log_softmax(generated_logits, dim=-1)
        probs = F.softmax(generated_logits, dim=-1)
        
        # 逐 token log_prob
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # 【修复HIGH #4】正确的熵计算：H(p) = -Σ p(x) log p(x)
        # 使用完整分布而非仅采样 token
        entropies = -(probs * log_probs).sum(dim=-1)  # (G,)
        
        return token_log_probs, entropies  # (G,), (G,)

    def _compute_token_log_probs(
        self, prompt: str, generated_text: str
    ) -> torch.Tensor | None:
        """【保留兼容】返回 token log_probs，旧接口。"""
        lps, _ = self._compute_token_log_probs_and_entropy(prompt, generated_text)
        return lps
    
    def _compute_current_log_prob(self, prompt: str, generated_text: str) -> torch.Tensor:
        """【保留兼容】返回平均对数概率（标量），旧接口。"""
        token_lps = self._compute_token_log_probs(prompt, generated_text)
        if token_lps is None:
            return torch.tensor(0.0, device=self.device)
        return token_lps.mean()
    
    def compute_advantages(self, rewards: List[float], values: List[float] = None) -> List[float]:
        """计算优势函数（GAE - Generalized Advantage Estimation）

        当 values 可用时，使用 GAE(λ) 降低方差：
          A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
          δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        当 values 不可用时，退化为带 baseline 的折扣累积奖励：
          A_t = G_t - mean(G)，其中 G_t = Σ_{l=0}^{T-t} γ^l r_{t+l}
        """
        if not rewards:
            return []

        if values is not None and len(values) == len(rewards) and any(v != 0.0 for v in values):
            lam = 0.95
            advantages = []
            gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1]
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * lam * gae
                advantages.insert(0, gae)
        else:
            advantages = []
            returns = 0
            for reward in reversed(rewards):
                returns = reward + self.gamma * returns
                advantages.insert(0, returns)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.tolist()
    
    def update_policy(self, batch_size: int = 4) -> Dict[str, float]:
        """标准 PPO 策略更新（逐 token 比率 + 裁剪 + 多轮次）。
        
        核心改进（相比之前的简化实现）：
        1. 逐 token 计算 ratio = exp(π_new - π_old)，而非先均值再单一 ratio
        2. 逐 token 裁剪：clamp(ratio, 1-ε, 1+ε)，精确控制每步策略变化幅度
        3. 多轮次 PPO：在相同 rollout 数据上迭代 ppo_epochs 次（充分利用数据）
        4. 每次 epoch 重新前向计算 current_log_prob（因为模型参数已更新）
        
        参考：Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
        """
        if len(self.episode_data['rewards']) < batch_size:
            return {'loss': 0.0, 'policy_loss': 0.0, 'entropy_loss': 0.0}
        
        advantages_raw = self.compute_advantages(self.episode_data['rewards'])
        
        # 选择高奖励样本（中位数以上）
        reward_threshold = sorted(self.episode_data['rewards'])[len(self.episode_data['rewards']) // 2]
        high_reward_indices = [
            i for i, r in enumerate(self.episode_data['rewards'])
            if r >= reward_threshold
        ]
        if len(high_reward_indices) == 0:
            high_reward_indices = list(range(len(self.episode_data['rewards'])))
        # 【修复】限制高奖励样本数量，防止PPO更新时做过多forward
        high_reward_indices = high_reward_indices[:8]
        
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        update_count = 0
        
        # ── 多轮次 PPO（核心） ──
        for epoch in range(self.ppo_epochs):
            # 【修复】共享优化器时不调用 zero_grad，避免清零 SFT 累积的梯度
            if not self.using_shared_optimizer:
                self.optimizer.zero_grad(set_to_none=True)
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_update_count = 0
            
            for idx in high_reward_indices:
                old_token_lps = self.episode_data['log_probs'][idx]
                advantage = advantages_raw[idx]
                
                if old_token_lps is None:
                    continue
                
                if not isinstance(old_token_lps, torch.Tensor):
                    old_token_lps = torch.tensor([old_token_lps], device=self.device, dtype=torch.float32)
                if not isinstance(advantage, torch.Tensor):
                    advantage = torch.tensor(advantage, device=self.device, dtype=torch.float32)
                
                prompt = self.episode_data['prompts'][idx]
                generated_text = self.episode_data['generated_texts'][idx]
                
                if not prompt or not generated_text:
                    continue
                
                # 【关键】每轮 epoch 重新计算当前策略的逐 token log_prob和熵
                # 【修复MED-6】一次前向同时获取两者，避免2x计算浪费
                current_token_lps, current_entropies = self._compute_token_log_probs_and_entropy(
                    prompt, generated_text
                )
                if current_token_lps is None:
                    continue
                
                # ── 对齐序列长度（安全网） ──
                old_len = old_token_lps.numel()
                new_len = current_token_lps.numel()
                min_len = min(old_len, new_len)
                if min_len == 0:
                    continue
                
                # 截断到相同长度
                old_lps = old_token_lps[:min_len].to(self.device)
                new_lps = current_token_lps[:min_len].to(self.device)
                
                # ── 逐 token 重要性采样比率 ──
                # ratio_t = π_new(a_t|s_t) / π_old(a_t|s_t)
                #         = exp(log π_new - log π_old)
                log_ratio = new_lps - old_lps.detach()
                ratio = torch.exp(log_ratio)  # (min_len,)
                
                # ── PPO Clipped Surrogate Objective（逐 token） ──
                # L^CLIP(θ) = E_t[min(ratio_t * A_t, clip(ratio_t, 1-ε, 1+ε) * A_t)]
                surr1 = ratio * advantage        # 未裁剪
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage
                
                # 逐 token 取 min 后取均值
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ── 熵奖励（鼓励探索） ──
                # 【修复CRIT-2+MED-6】使用当前策略的分布熵（已从上方一次前向获取）
                if current_entropies is not None:
                    entropy_val = current_entropies[:min_len].mean()
                else:
                    entropy_val = -new_lps.mean()
                entropy_loss = -self.entropy_coef * entropy_val
                
                # ── 近似 KL 散度（监控指标） ──
                # KL(π_old || π_new) ≈ mean(log π_old - log π_new)
                approx_kl = (old_lps.detach() - new_lps).mean()
                
                loss = policy_loss + entropy_loss
                
                # 梯度累积（多轮次 + 多样本混合）
                (loss / (self.ppo_epochs * len(high_reward_indices))).backward()
                
                epoch_policy_loss += policy_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                epoch_update_count += 1
            
            # ── 每个 epoch 结束后执行优化器步进 ──
            if epoch_update_count > 0:
                # 梯度裁剪（标准 PPO 做法）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 【修复】共享优化器时，学习率由主训练流调度器统一管理，PPO不覆盖
                if not self.using_shared_optimizer:
                    current_lr = self.apply_ppo_learning_rate()
                    self.optimizer.step()
                else:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    # 【修复Bug #2】PPO步后必须清零梯度，防止SFT残余+PPO梯度混合
                    self.optimizer.zero_grad(set_to_none=True)
                self.ppo_training_steps += 1
                
                total_policy_loss += epoch_policy_loss
                total_entropy_loss += epoch_entropy_loss
                update_count += epoch_update_count
                
                if self.ppo_training_steps % 50 == 0:
                    avg_kl = total_approx_kl / max(epoch_update_count, 1)
                    print(f"[PPO] Epoch {epoch+1}/{self.ppo_epochs}, Step {self.ppo_training_steps}, "
                          f"LR={current_lr:.2e}, PolicyLoss={epoch_policy_loss/max(epoch_update_count,1):.4f}, "
                          f"Entropy={epoch_entropy_loss/max(epoch_update_count,1):.4f}, "
                          f"ApproxKL={avg_kl:.4f}", flush=True)
        
        # ── 清空 episode 数据 ──
        self.episode_data = {
            'log_probs': [],
            'entropies': [],
            'rewards': [],
            'values': [],
            'actions': [],
            'states': [],
            'prompts': [],
            'generated_texts': []
        }
        
        # 【修复】PPO更新后强制清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        n_updates = max(update_count, 1)
        return {
            'loss': (total_policy_loss + total_entropy_loss) / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'approx_kl': total_approx_kl / n_updates
        }
    
    def clear_data(self):
        """清空episode数据"""
        self.episode_data = {
            'log_probs': [],
            'entropies': [],  # 【修复CRIT-1】遗漏导致内存泄漏
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
                think_tokens = None
                if thinking_available:
                    try:
                        think_end_idx = child_tokens.index(TextTokenizer.THINK_END_TOKEN)
                        think_tokens = child_tokens[:think_end_idx + 1]
                    except ValueError:
                        think_tokens = child_tokens
                reward = self.evaluate_node(
                    child,
                    prompt_tokens,
                    child_tokens,
                    think_tokens=think_tokens,
                    context=context
                )
                child.reward = reward
                
                self.backpropagate(child, reward)
        
        best_node = self._select_best_node()
        best_tokens = best_node.get_path()
        
        generated_text = TextTokenizer.decode(torch.tensor(best_tokens))
        
        think_text = None
        answer_text = generated_text
        if thinking_available:
            try:
                think_end_idx = best_tokens.index(TextTokenizer.THINK_END_TOKEN)
                think_text = TextTokenizer.decode(torch.tensor(best_tokens[:think_end_idx + 1]))
                answer_text = TextTokenizer.decode(torch.tensor(best_tokens[think_end_idx + 1:]))
            except ValueError:
                pass
        
        total_reward, reward_breakdown = self.reward_model.compute_total_reward(
            think_text=think_text,
            answer_text=answer_text,
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
                        
                        think_text = None
                        answer_text = generated_text
                        if thinking_available:
                            try:
                                think_end_idx = new_beam['tokens'].index(TextTokenizer.THINK_END_TOKEN)
                                think_text = TextTokenizer.decode(torch.tensor(new_beam['tokens'][:think_end_idx + 1]))
                                answer_text = TextTokenizer.decode(torch.tensor(new_beam['tokens'][think_end_idx + 1:]))
                            except ValueError:
                                pass
                        
                        total_reward, _ = self.reward_model.compute_total_reward(
                            think_text=think_text,
                            answer_text=answer_text,
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
        
        think_text = None
        answer_text = generated_text
        if thinking_available:
            try:
                think_end_idx = best_tokens.index(TextTokenizer.THINK_END_TOKEN)
                think_text = TextTokenizer.decode(torch.tensor(best_tokens[:think_end_idx + 1]))
                answer_text = TextTokenizer.decode(torch.tensor(best_tokens[think_end_idx + 1:]))
            except ValueError:
                pass
        
        total_reward, reward_breakdown = self.reward_model.compute_total_reward(
            think_text=think_text,
            answer_text=answer_text,
            context=context
        )
        
        return generated_text, total_reward, reward_breakdown