import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass
from tokenizer import TextTokenizer


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
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
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

        self.optimizer.zero_grad()

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
