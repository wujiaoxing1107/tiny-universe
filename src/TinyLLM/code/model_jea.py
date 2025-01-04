import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    # 自定义超参数
    dim: int = 288  # 模型维度
    n_layers: int = 6  # Transformer层数
    n_heads: int = 6  # 注意力机制的头数
    n_kv_heads: Optional[int] = 6  # 键/值头数，如果未指定，则默认为n_heads
    vocab_size: int = 32000  # 词汇表大小
    hidden_dim: Optional[int] = None  # 隐藏层维度，如果未指定，则使用其他规则确定
    multiple_of: int = 32  # MLP隐藏层大小是这个数的倍数
    norm_eps: float = 1e-5  # 归一化层的epsilon值
    max_seq_len: int = 256  # 最大序列长度
    dropout: float = 0.0  # 丢弃率


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps是为了防止除以0的情况
        self.eps = eps
        # weight是一个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Tools:
    def __init__(self):
        pass

    def compute_fre_sin_and_cos(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        fre = 1.0 * (theta ** (torch.arange(0, dim, 2)[:].float() / dim)).reshape(-1, dim // 2)

        token = torch.arange(max_seq_len, device=fre.device).reshape((-1, 1)).to(torch.float)
        # print(fre, fre.shape)
        # print(token, token.shape)
        fre = fre.to(torch.float) * token.to(torch.float)
        fre_cos = torch.cos(fre)
        fre_sin = torch.sin(fre)
        # print(fre.shape)
        return fre_cos, fre_sin

    def repeat_kv(self, x: torch.Tensor, n_repeat: int) -> torch.Tensor:
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_repeat == 1:
            return x
        # 为什么要用expand之后再reshape而不能直接用tensor自带的repeat?
        # expand
        # 方法用于对张量进行扩展，但不实际分配新的内存。它返回的张量与原始张量共享相同的数据
        # repeat
        # 方法通过实际复制数据来扩展张量。它返回的新张量不与原始张量共享数据，扩展后的张量占用了更多的内存。
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_repeat, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_repeat, head_dim)
        )
        pass

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        # 将freqs_cis调整为新的形状，并返回
        return freqs_cis.reshape(shape)

    def apply_rotary_embedding(
            self,
            xq: torch.Tensor,
            xk: torch.Tensor,
            freq_cos: torch.Tensor,
            freq_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
        # xq.float().reshape(xq.shape[:-1] + (-1, 2))->
        # (batch_size, seq_len, num_heads, d // 2, 2)
        # unbind(-1)沿着最后一维（也就是维度值为 2 的这一维，通过参数 -1 指定）进行拆分操作。
        # 分别变成(batch_size, seq_len, num_heads, d // 2)
        xq = xq.float().reshape(xq.shape[:-1] + (-1, 2))
        print(xq.shape)
        xq_r, xq_i = xq.unbind(-1)
        print(xq_r.shape)
        xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

        # 重新塑形频率tensor以进行广播
        # (seq_len,d//2)
        freqs_cos = self.reshape_for_broadcast(freq_cos, xq_r)
        freqs_sin = self.reshape_for_broadcast(freq_sin, xq_r)

        # 应用旋转，分别计算旋转后的实部和虚部
        # (batch_size, seq_len, num_heads, d // 2)
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        # 将最后两个维度合并，并还原为原始张量的形状
        # torch.stack 它的作用是沿着一个新的维度将给定的一系列张量进行堆叠，从而创建出一个维度更高的新张量
        # flatten(3) 这里的参数 3 表示从第 3 维开始进行扁平化操作（注意维度索引从 0 开始计数）
        # (batch_size, seq_len, num_heads, d // 2)->\
        # (batch_size, seq_len, num_heads, d // 2,2)->(batch_size, seq_len, num_heads, d)
        xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
        print(f"xq_out.shape: {xq_out.shape}")
        print(f"xq_out.shape: {xq_out.shape}")
        xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
        print(f"xk_out.shape: {xk_out.shape}")

        return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头数整除。
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵。
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout。
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率。
        self.dropout = args.dropout
        self.tools = Tools()

        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 计算查询（Q）、键（K）、值（V）。
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入（RoPE）。
        xq, xk = self.tools.apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 对键和值进行扩展以适应重复次数。
        xk = self.tools.repeat_kv(xk, self.n_rep)
        xv = self.tools.repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理。
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持Flash Attention，选择实现方式。
        if self.flash:
            # 使用Flash Attention。
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            # 使用手动实现的注意力机制。
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        # F.silu是Swish激活函数，swish激活函数的表达式为y=x*sigmoid(x),sigmoid(x)=1/(1+(-x).pow(e))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义输入维度
        self.dim = args.dim
        # 定义每个头的维度，等于输入维度除以头数
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMA2Attention对象，用于进行多头注意力计算
        self.attention = Attention(args)
        # 定义LLaMAMLP对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层的ID
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        # (batch_size,seq_len,dim)->(batch_size,seq_len,vocab_size)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        # 从本质上讲，词嵌入层是将输入的词（通常以词索引的形式）转换为向量表示，
        # 而输出层在生成任务（例如预测下一个词）时，将内部的表示转换回词汇表空间中的概率分布。
        self.tok_embeddings.weight = self.output.weight
        self.tools = Tools

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = self.tools.compute_fre_sin_and_cos(self.args.dim // self.args.n_heads,
                                                                  self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入层和Dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，说明是训练模式，因此需要计算损失
            logits = self.output(h)
            # shape of targets:(bs,seq_len)
            # shape of logits:(bs,seq_len,vocab_size)
            # 将logits展成(bs*seq_len,prob_vocab_size)，target展成(bs*seq_len)，target中每个元素代表的都是真实标签
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
        # output经过最后的nn.Linear层后(input_dim,vocab_size)
        # 因此张量形状由(bs,seq_len)->(bs,seq_len,dim)->(bs,seq_len,vocab_size)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 获取所有需要更新的参数
        # self.named_parameters() 返回一个遍历模块参数的迭代器，包括参数名和参数本身
        # pn-参数名 p-参数
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # 将参数分为需要权重衰减和不需要权重衰减的两组
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 打印参数数量信息
        # numel()是一个函数，用于返回一个张量（Tensor）中元素的总数。它是 “number of elements” 的缩写
        # 对于一个形状为(2, 3)的张量，其元素总数为2 * 3 = 6
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 根据设备类型选择使用标准 AdamW 或其融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 估计模型的 FLOPs 利用率 (MFU) 单位：A100 bfloat16 的峰值 FLOPS """
        # 计算每次迭代的 FLOPs 数量（参考 PaLM 论文的附录 B）
        # PaLM: Scaling Language Modeling with Pathways: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.args
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # 将 FLOPs 吞吐量表示为 A100 bfloat16 峰值 FLOPS 的比例
        flops_achieved = flops_per_iter * (1.0 / dt)  # 每秒计算的 FLOPs
        flops_promised = 312e12  # A100 GPU bfloat16 的峰值 FLOPS 为 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # @torch.inference_mode()：PyTorch 中的一个装饰器，用于标记一个函数或方法在执行时处于推理（inference）模式
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        # idx:(batch_size,seq_len) 最开始是(1,embedded_token_size)
        print(f"len of idx: {len(idx)}")

        print(f"shape of idx: {idx.shape}\n")
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            print(f"shape of idx_cond: {idx_cond.shape}")
            # logits shape:(1,seq_len,prob_of_vocab),其中prob_of_vocab表示对应token(i)下一个token(i+1)的概率，大小是vocab_size
            # logits = self.output(h[:, [-1], :]),由于是推理阶段，没有目标token不需要做损失计算，可以只返回最后一个token预测的概率
            logits = self(idx_cond)
            print(f"shape of logits: {logits.shape}")
            logits = logits[:, -1, :]  # 只保留最后一个时间步的输出
            # logits(bs,1,vocab_size)
            # torch.topk(input, k, dim=None, largest=True, sorted=True)
            # 用于在给定的张量（Tensor）中找到指定数量（k）的最大值及其对应的索引。
            # 它返回两个张量，一个是包含最大值的张量，另一个是包含这些最大值在原始张量中索引的张量。
            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    print(f"shape of v: {v.shape}\n")
                    # v(1,top_k)，返回的top_k个概率值，从大到小排列
                    # logits(1,1,vocab_size)
                    # 对 v 张量进行索引取值,取待生成位置的最后一个即最小概率的值
                    # 使得logits中小于这个最小概率值的都设置为负无穷到，因此在后续softmax中只留下top_k个进行采样
                    logits[logits < v[:, [-1]]] = float('-Inf')
                probs = F.softmax(logits, dim=-1)
                # torch.multinomial 函数用于从给定的概率分布（这里就是 probs）中按照多项分布进行采样
                # 第一个参数是概率分布张量 probs，第二个参数 num_samples 表示要采样的样本数量，这里设置为 1，意味着每次只从概率分布中抽取一个样本。
                idx_next = torch.multinomial(probs, num_samples=1)

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class KVCache:
    def __init__(self, num_layers, max_seq_length, hidden_size):
        """
        初始化KV缓存。

        参数:
        - num_layers: 模型的层数，例如Transformer中的层数。
        - max_seq_length: 最大序列长度，决定缓存的最大容量。
        - hidden_size: 隐藏层大小，对应键值对的维度。
        """
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size

        # 初始化键值对缓存，使用字典来存储每一层的键和值缓存
        self.key_cache = [torch.zeros((max_seq_length, hidden_size)) for _ in range(num_layers)]
        self.value_cache = [torch.zeros((max_seq_length, hidden_size)) for _ in range(num_layers)]

        # 当前缓存的有效长度（已经填充了多少个位置）
        self.cache_lengths = [0] * num_layers

    def update(self, layer_idx, keys, values):
        """
        更新指定层的键值对缓存。

        参数:
        - layer_idx: 要更新的层索引（从0开始）。
        - keys: 当前生成步骤的键张量，形状通常为 (batch_size, num_heads, seq_len, head_dim)，这里简化为 (seq_len, hidden_size)。
        - values: 当前生成步骤的值张量，形状通常为 (batch_size, num_heads, seq_len, head_dim)，这里简化为 (seq_len, hidden_size)。
        """
        assert layer_idx < self.num_layers

        # 获取当前层的已有缓存长度
        current_length = self.cache_lengths[layer_idx]

        # 确定可以更新的位置范围，不能超过最大缓存长度
        end_index = min(current_length + keys.shape[0], self.max_seq_length)
        update_length = keys.shape[0] if end_index > current_length else end_index - current_length

        # 更新键缓存
        self.key_cache[layer_idx][current_length:end_index] = keys[:update_length]

        # 更新值缓存
        self.value_cache[layer_idx][current_length:end_index] = values[:update_length]

        # 更新当前层的缓存长度
        self.cache_lengths[layer_idx] = end_index

    def get(self, layer_idx):
        """
        获取指定层的键值对缓存。

        参数:
        - layer_idx: 层索引。

        返回:
        - 该层的键缓存和值缓存。
        """
        assert layer_idx < self.num_layers
        return self.key_cache[layer_idx][:self.cache_lengths[layer_idx]], \
            self.value_cache[layer_idx][:self.cache_lengths[layer_idx]]


if __name__ == '__main__':
    args = ModelArgs()
    # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    # torch.randint(low, high, size)
    x = torch.randint(0, 32000, (1, 50))  # [bs, seq_len]
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)

    out = model(x)
    print(out.shape)  # [batch_size, 1, vocab_size]
