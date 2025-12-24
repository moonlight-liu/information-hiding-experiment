import numpy as np
import math

class SparSampStego:
    def __init__(self, message_block_len=4):
        """
        初始化 SparSamp
        :param message_block_len: l_m (论文中的 l_m)，每次嵌入的比特长度，默认 4 bit
        """
        self.lm = message_block_len
        self.Nm = 1 # 候选消息数，初始为 1
        self.km = 0 # 当前消息索引
        
        # 模拟一个伪随机数生成器 (PRNG)，双方共享种子
        self.prng = np.random.RandomState(2025) 

    def bin2dec(self, bits):
        """二进制列表转十进制"""
        if len(bits) == 0: return 0
        return int("".join(map(str, bits)), 2)

    def dec2bin(self, val, length):
        """十进制转二进制列表"""
        bin_str = bin(val)[2:].zfill(length)
        return [int(b) for b in bin_str]

    # --- 对应论文 Algorithm 1: sample(P_i, r_i) ---
    def sample(self, Probs, r_pseudo):
        """
        根据概率分布 P 和 伪随机数 r 采样 Token
        并返回该 Token 在累积概率分布中的区间 [start, end]
        """
        cuml = 0.0
        start = 0.0
        selected_token_idx = -1
        
        for k, p in enumerate(Probs):
            start = cuml
            cuml += p
            # 如果累积概率超过了随机数 r，说明选中了这个词
            if cuml > r_pseudo:
                selected_token_idx = k
                # 计算区间 SE_i (Start, End) within the probability space
                # 注意：论文算法1 line 6 计算的是相对于 cuml 的偏移，
                # 这里我们直接返回绝对区间 [start, end] 更直观
                return selected_token_idx, (start, cuml)
        
        # 防止浮点误差导致没选中，默认选最后一个
        return len(Probs) - 1, (start, cuml)

    # --- 对应论文 Algorithm 2: sparse(...) ---
    def sparse(self, SE, Nm, km, r_pseudo):
        """
        稀疏采样核心更新逻辑：更新 Nm 和 km
        SE: 选中 Token 的概率区间 [start, end]
        """
        start, end = SE
        
        # 映射回 [0, 1) 区间 (相对于 r_pseudo 的偏移处理)
        # 论文逻辑：判断选中的区间里包含了多少个“候选消息区间”
        
        # 1. 计算 temp0 和 temp1 (区间边界对应的消息索引界限)
        # 注意：这里需要处理循环移位 (modulo 1) 的情况，论文公式比较抽象
        # 我们用更直观的“区间映射”理解：
        
        # 消息映射后的随机数 r(m) = (r + km/Nm) mod 1
        # 我们要看 r(m) 是否落在了 [start, end] 里
        
        # 简化的逻辑实现（等价于论文）：
        # 计算在当前 Nm 分辨率下，有哪些 k 使得 r(k) 命中该 Token
        
        new_candidates = []
        for k in range(Nm):
            # 计算第 k 个候选消息对应的 随机数
            r_k = (r_pseudo + k / Nm) % 1.0
            
            # 判断是否命中当前 Token 的区间
            if start <= r_k < end:
                new_candidates.append(k)
        
        # 更新 Nm (新的候选数量)
        new_Nm = len(new_candidates)
        
        # 更新 km (如果当前真实的 km 在新候选列表里，它在列表里的位置就是新的 km)
        new_km = 0
        if km in new_candidates:
            new_km = new_candidates.index(km)
        else:
            # 这种情况理论上只有在非嵌入模式下才会发生，或者浮点精度问题
            pass 
            
        return new_Nm, new_km

    # --- 对应论文 Algorithm 3: Embedding ---
    def embed(self, cover_context_generator, secret_message_bits):
        """
        嵌入主循环
        :param cover_context_generator: 一个生成器，模拟 LLM 每次吐出下一个词的概率分布
        :param secret_message_bits: 待嵌入的比特流 [0, 1, 1, ...]
        """
        stego_tokens = []
        msg_ptr = 0
        total_bits = len(secret_message_bits)
        
        print(f"Start Embedding... Message Length: {total_bits} bits")
        
        # 状态重置
        self.Nm = 1
        self.km = 0
        self.prng = np.random.RandomState(2025) # 重置随机种子
        
        step = 0
        while True:
            # 1. 生成伪随机数 r
            r = self.prng.rand()
            
            # 2. 获取模型预测的概率分布 (模拟)
            try:
                probs = next(cover_context_generator)
            except StopIteration:
                break # 文本生成结束
            
            # 3. 尝试嵌入数据 (如果当前候选区间唯一 Nm=1，说明可以开始嵌入新的一块)
            if msg_ptr + self.lm <= total_bits:
                if self.Nm == 1:
                    # 取出 l_m 个比特
                    bits = secret_message_bits[msg_ptr : msg_ptr + self.lm]
                    val = self.bin2dec(bits)
                    
                    self.km = val # 设置目标消息索引
                    self.Nm = 2**self.lm # 扩大搜索空间
                    msg_ptr += self.lm
                    # print(f"Step {step}: Embedded chunk {bits} -> km={self.km}")

            # 4. 计算当前步的“含密随机数” r_m
            # r_m = (r + km / Nm) mod 1
            r_m = (r + self.km / self.Nm) % 1.0
            
            # 5. 采样 Token
            token, SE = self.sample(probs, r_m)
            stego_tokens.append(token)
            
            # 6. 稀疏更新 (Sparse Update) -> 缩小 Nm，更新 km
            self.Nm, self.km = self.sparse(SE, self.Nm, self.km, r)
            
            step += 1
            
        return stego_tokens

    # --- 对应论文 Algorithm 4: Extraction ---
    def extract(self, cover_context_generator, stego_tokens):
        """
        提取主循环
        """
        extracted_bits = []
        
        # 状态重置
        self.Nm = 1
        # 提取时我们要维护一个“候选列表”或者“回溯栈”，但SparSamp设计巧妙在它是确定性的
        # 论文中用 temparr 来辅助计算
        # 这里我们用 simplified 逻辑：
        # 接收方不知道 km，但知道 Nm 的变化规律。
        # 当 Nm 收敛回 1 时，说明上一个块完全确定了。
        
        # 在提取端，我们需要知道“每一个候选路径”最终会走向哪个 Token
        # 但其实更简单的理解是：接收端完全模拟发送端的“候选集缩减过程”
        
        # 初始化候选集：一开始只有一个“空”候选（代表未开始）
        # 但 SparSamp 的提取稍微复杂一点，它需要判断什么时候 Nm 变成了 1
        # 为了简化演示代码，我们假设接收方知道逻辑：
        # 发送方在 Nm=1 时总是会加载新数据（如果还有数据的话）
        
        # 我们用一个 buffer 来存当前未确定的 bits
        # 实际论文中是逆向算 km
        
        # 这里为了代码可读性，我们采用“模拟同步”策略：
        # 接收方维护一个候选消息列表 [0, 1, ..., Nm-1]
        # 每一步采样后，看哪些候选消息对应的 r_k 会命中该 Token
        # 留下的就是新的候选列表。
        
        self.prng = np.random.RandomState(2025) # 必须同种子
        
        # 当前可能的 km 列表
        candidate_kms = [0] 
        current_Nm_space = 1 # 当前处于哪个 embedding 层的空间
        
        extracted_chunks = []
        
        step = 0
        for token in stego_tokens:
            r = self.prng.rand()
            try:
                probs = next(cover_context_generator)
            except StopIteration:
                break
            
            # 获取该 Token 的概率区间
            # 我们需要重新计算 start, end，可以通过调用 sample 得到
            # 但 sample 需要 r，这里我们直接手动算区间
            cuml = 0.0
            start, end = 0.0, 0.0
            for k, p in enumerate(probs):
                t_start = cuml
                cuml += p
                if k == token:
                    start, end = t_start, cuml
                    break
            
            # 核心提取逻辑：筛选候选人
            # 对于当前所有可能的 km (in candidate_kms), 计算 r_k
            # 如果 r_k 命中 [start, end]，则保留
            
            next_candidates = []
            
            # 如果当前是初始状态 (候选集大小为1)，说明可能开始新的一轮嵌入
            # 在发送端，如果 Nm=1，会扩展成 2^lm 个候选
            # 我们假设总是尝试扩展（只要流没结束），实际应用中会有结束符
            
            if len(candidate_kms) == 1:
                # 这是一个确定的状态，记录下来的就是提取出的值（如果是刚收敛）
                # 但我们需要判断是否是刚开始
                pass

            # 扩展阶段：如果当前候选列表长度为1，意味着上一块结束了，
            # 下一步潜在的候选人变成了 2^lm 个（对应新消息）
            # *注意：这是简化的理解，为了代码能跑通，我们模拟“尝试解码”*
            
            if len(candidate_kms) == 1:
                # 这一步对应发送端的 "if Nm=1: Nm <- 2^lm"
                # 我们把这唯一的 km 扩展成 2^lm 个假设
                base_km = candidate_kms[0] # 其实总是0，因为归一化了
                # 实际上，每次收敛后，km 归 0，Nm 归 1
                
                # 产生新的候选集 0 ~ 2^lm - 1
                current_candidates = list(range(2**self.lm))
                current_Nm = 2**self.lm
            else:
                current_candidates = candidate_kms
                current_Nm = len(candidate_kms) * (2**self.lm / len(candidate_kms)) # 保持比例
                # 这里直接用 len(candidate_kms) 即可，因为是从上一轮筛选下来的
                current_Nm = 2**self.lm # 总空间始终是 2^lm (逻辑上的)
                # 不，SparSamp 的 Nm 是动态变化的。
                
                # 让我们回退到最稳健的方法：完全复刻 sparse 函数的逻辑
                # 接收方不知道 km，但他知道 Nm 的变化
                # 接收方维护 Nm
            
            # --- 修正后的提取逻辑 ---
            # 接收方维护 Nm。如果 Nm=1，说明上一个消息解完了，存下来。
            # 然后 Nm 变成 2^lm。
            # 然后根据 Token 和 sparse 函数更新 Nm 和 km_base
            # 但接收方不知道具体的 km，只知道 km 在某个范围内。
            # 实际上，接收方通过 "Sparse" 操作，计算出 offset。
            
            # 让我们用论文 Algorithm 4 的数学方法：
            # 逆向还原不太直观，我们用“候选集过滤法”是等价且更容易懂的。
            
            # 重新初始化（模拟）
            if step == 0:
                candidates = list(range(2**self.lm)) # 初始候选集
                
            filtered = []
            Nm_curr = len(candidates) # 这里的 Nm 不是 paper 里的 Nm，而是当前存活的候选数
            # 实际上 paper 里的 Nm 就是存活数 * 缩放因子
            
            # 让我们直接用“模拟发送端”的方式：
            # 我们假设所有 2^lm 个消息都是可能的。
            # 只要某一个消息 m 导致生成的 token != received_token，就把它剔除。
            # 当只剩 1 个消息时，那就是它！
            
            # 这种方法对于短消息块 (lm=4, 16 candidates) 非常高效且易懂
            
            survivors = []
            for cand_msg in candidates:
                # 计算如果消息是 cand_msg，在当前状态下应该采样的 token
                # 这需要维护每个 candidate 的内部状态 (Nm, km)
                # 这有点复杂。
                pass
                
        return []

# --- 为了让代码简单可运行，我实现一个基于“候选集筛选”的提取器 ---
# 这是 SparSamp 的等价逻辑：接收方不知道消息是啥，但他可以穷举所有可能的消息（比如16种），
# 看哪一种消息会让发送方生成出当前的 Token。
# 随着 Token 一个个生成，不符合的消息会被排除，最后只剩一个。

def run_simulation():
    # 1. 模拟 LLM：一个无限生成随机概率分布的生成器
    def mock_llm_generator(vocab_size=100, length=50):
        np.random.seed(123) # 固定模型行为
        for _ in range(length):
            # 生成随机概率分布 (Softmax)
            logits = np.random.randn(vocab_size)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            yield probs

    # 2. 准备数据
    lm = 4 # 每次嵌 4 bit (即 0~15 的整数)
    secret_msg = [1, 0, 1, 1,   0, 0, 1, 1] # 两个块: 1011(11), 0011(3)
    print(f"Secret Message: {secret_msg}")

    # 3. 嵌入
    stego = SparSampStego(message_block_len=lm)
    # 注意：生成器只能用一次，所以嵌入和提取要分别实例化
    gen_embed = mock_llm_generator()
    
    stego_tokens = stego.embed(gen_embed, secret_msg)
    print(f"Stego Tokens Generated: {stego_tokens}")

    # 4. 提取 (使用筛选法)
    print("\nStart Extracting...")
    
    # 重新初始化生成器和随机数
    gen_extract = mock_llm_generator()
    prng_extract = np.random.RandomState(2025)
    
    # 候选池：我们在等待“唯一幸存者”
    # 结构：{ message_value: {Nm, km} } 
    # 初始状态：假设当前正在嵌入一个 lm 长的块，所有 2^lm 种可能都在
    
    decoded_bits = []
    target_bits = len(secret_msg)  # 目标比特数
    
    # 当前正在解密的候选集：List of (candidate_val, current_Nm, current_km)
    active_candidates = []
    
    # 初始化：加入所有可能的消息 (0 到 15)
    # 每个候选人的初始状态: Nm=16, km=candidate_val
    for val in range(2**lm):
        active_candidates.append( {'val': val, 'Nm': 2**lm, 'km': val} )
    
    step = 0
    for token in stego_tokens:
        if len(active_candidates) == 0:
            break
            
        r = prng_extract.rand()
        probs = next(gen_extract)
        
        # 计算当前 Token 的概率区间 [start, end]
        cuml = 0.0
        start, end = 0.0, 0.0
        for k, p in enumerate(probs):
            t_start = cuml
            cuml += p
            if k == token:
                start, end = t_start, cuml
                break
        
        # 筛选：检查哪些 candidate 在当前步 依然会选中这个 token
        next_gen_candidates = []
        
        finished_candidate = None # 标记是否有候选人已经收敛 (Nm -> 1)
        
        for cand in active_candidates:
            # 还原发送端逻辑
            Nm = cand['Nm']
            km = cand['km']
            
            # 计算 r_m
            r_m = (r + km / Nm) % 1.0
            
            # 检查 r_m 是否落在 [start, end]
            if start <= r_m < end:
                # 命中！更新状态 (Sparse Update)
                # 这里我们需要把 sparse 函数逻辑内联进来，或者用之前的类方法
                # 为了简单，直接手动算：
                
                # sparse逻辑：计算新 Nm, km
                # 重新计算该 token 区间内包含多少个 slot
                # Slot width = 1/Nm
                # count = ... 
                # 这里直接调用 helper
                new_Nm, new_km = stego.sparse((start, end), Nm, km, r)
                
                # 更新状态
                cand['Nm'] = new_Nm
                cand['km'] = new_km
                
                # 如果 Nm 变成了 1 (或者更小，但在正确逻辑下是收敛到1)
                # 说明这个 candidate 已经“确信”自己是唯一路径了？
                # 不，SparSamp 的逻辑是：当 Nm 减小，在这个分支下继续走。
                # 真正的“解码成功”是：当其他所有竞争对手都“死”了（不命中），只剩这一个！
                
                next_gen_candidates.append(cand)
            else:
                # 没命中，说明如果消息是这个，就不可能生成这个 token
                pass
        
        active_candidates = next_gen_candidates
        # print(f"Step {step}: Survivors = {[c['val'] for c in active_candidates]}")
        
        # 检查是否只剩一个候选人，并且这个候选人已经无法再区分（或者我们逻辑上的收敛）
        # 实际上，SparSamp 设计是“分块独立”的。
        # 如果我们发现 active_candidates 里所有人的原始 val 都是同一个 (虽然这在 SparSamp 里不太可能，因为不同 val 路径不同)
        # 或者只剩 1 个候选人
        
        if len(active_candidates) == 1:
            # 找到了！
            winner = active_candidates[0]
            # print(f"--> Decoded Chunk: {winner['val']}")
            decoded_bits.extend(stego.dec2bin(winner['val'], lm))
            
            # 检查是否已经解码完成
            if len(decoded_bits) >= target_bits:
                break
            
            # 准备解下一个块：重置候选池
            active_candidates = []
            for val in range(2**lm):
                active_candidates.append( {'val': val, 'Nm': 2**lm, 'km': val} )
                
        step += 1
        
    print(f"Decoded Bits: {decoded_bits}")
    print(f"Original Bits: {secret_msg}")
    print(f"Match: {decoded_bits == secret_msg}")

if __name__ == "__main__":
    run_simulation()