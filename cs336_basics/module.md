# module

## RMSNorm 

使用math.sqrt(d)和torch.rsqrt(d)哪个更好？

RMSNorm的数据要先转成float32，为什么，怎么判断要不要转化？

## Linea
我才用了和torch.nn.Linear相反的存储，详见代码，因为为了方便做矩阵乘法

## FNN
略

## mha
多头注意力可以实现并行，但是实现上，只用了一个权重去表示多头注意力的权重，因为做一整个矩阵乘法就等于多个矩阵乘法拼接。

## tranformerLM
### (a) Consider a GPT-2 XL-sized model using our assignment architecture, which has the following  configuration:  
vocab_size: 50,257 
context_length: 1,024 
num_layers: 48 
d_model: 1,600 
num_heads: 25 
d_ff: 4,288 
(the nearest multiple of 64 to 8  3 × 1, 600)  

Suppose we constructed our model using this configuration. **How many trainable parameters would our model have?** 

Assuming each parameter is represented using single-precision floating point, **how much memory is required to just load this model?**

一个一个来算，先算内存，全部采用GB（十进制）而非GiB（二进制）：
- 一个单精度浮点数是 4 Bytes
- memory of embedding layer：$vocab\_size \times d\_model = 50,257 \times 1,600 = 80,411,200$, takes 80,411,200 × 4 Bytes = 321,644,800 Bytes = 0.322GB
- memory of each transformer block:
  - mha:
    - linear layers: $d\_model \times d\_model \times 3 + d\_model \times d\_model = 1,600 \times 1,600 \times 3 + 1,600 \times 1,600 = 10,240,000$, takes 10,240,000 × 4 Bytes = 40,960,000 Bytes = 0.041GB
    - rope embedding: $context\_length \times \frac{d\_model}{2} = 1,024 \times 800 = 819,200$, takes 819,200 × 4 Bytes = 3,276,800 Bytes = 0.003GB
  - fnn: $d\_model \times d\_ff + d\_ff \times d\_model + d\_ff \times d\_model = 1,600 \times 4,288 + 4,288 \times 1,600 + 4,288 \times 1,600 = 20,582,400$, takes 20,582,400 × 4 Bytes = 82,329,600 Bytes = 0.082GB
  - rmsnorm: takes 0.000013GB（可忽略不计）
- memory of transformer blocks: 48 × (0.041GB + 0.003GB + 0.082GB) = 48 × 0.127GB = 6.075GB
- memory of output layer: $d\_model \times vocab\_size = 1,600 \times 50,257 = 80,411,200$, takes 80,411,200 × 4 Bytes = 321,644,800 Bytes = 0.322GB
- total memory: 0.322GB + 6.075GB + 0.322GB = 6.719GB
- total trainable parameters：1.68B

总计算量：
为方便计算：[B, T, D]表示batch size, context length, model dimension
- input embedding layer: 这个主要是查表，所以不算计算量
- transformer block:
  - mha: 
    - 3 linear layers: QKV权重矩阵：[B * T, D] @ [D, D] =  3 * 2BTD^2 = 6BTD^2，输出权重矩阵：[B * T, D] @ [D, D] = 2BTD^2，一共8BTD^2；计算注意力：4BT^2D；总的为8BTD^2 + 4BT^2D
    - ffn: 6BTDF, 其中F是d_ff
    - rmsnorm: 暂且忽略不计
    - 总的为8BTD^2 + 4BT^2D + 6BTDF 
    - 假设batch size为1，那么就是8TD^2 + 4T^2D + 6TDF = 8 * 1024 * 1600^2 + 4 * 1024^2 * 1600 + 6 * 1024 * 1600 * 4288 = 20.97B + 20.97B + 20.97B = 69.84B FLOPs
  - transformer blocks: 48 * 69.84B = 3.35T FLOPs
- output layer: [B * T, D] @ [D, vocab_size] = 2BTDvocab_size =  2 * 1024 * 1600 * 50257 = 164.68B
- total computation: 3,415.2B + 164.68B = 3,579.88B = 3.5T FLOPs = 3.5e12 FLOPs

### A list of matrix multiplies (with descriptions), and the total number of FLOPs required.
Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped  model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.  27 Deliverable: 

答案见上题

### Based on your analysis above, which parts of the model require the most FLOPs?
FFN和Attention中的线性投影需要最多的计算，因为它们都是大矩阵乘法；因为上下文1024小于模型维度1600，所以主要的瓶颈在于$BTD^2$而不是$BT^2D$，所以主要的FLOPs集中在FFN、QKV计算以及output

### which parts of the Transformer LM take up proportionally more or less of the total FLOPs? 
Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium  (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads).

随着模型维度的增长，有些组件的参数是线性增长如llm head，attention，二次增长的占比更大

| 模型           |             总 FLOPs | Projection | Attention |   FFN | LM head |
| ------------ | ------------------: | ---------: | --------: | ----: | ------: |
| GPT-2 small  | (2.92\times10^{11}) |      19.9% |     13.3% | 39.8% |   27.1% |
| GPT-2 medium | (8.30\times10^{11}) |      24.8% |     12.4% | 50.1% |   12.7% |
| GPT-2 large  | (1.77\times10^{12}) |      **27.3%** |     10.9% | **54.3%** |    7.4% |

### Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one  forward pass change? How does the relative contribution of FLOPs of the model components change?


将 context length 从：$1024 \rightarrow 16384$即扩大16倍，attention部分会二次增长，总 FLOPs 从$3.52\times10^{12}$$增长到：
$
\boxed{
1.34\times10^{14}\text{ FLOPs}
}
$$

约增加$$
\boxed{38\times}
$

新的 FLOPs 占比：

| Component | Percentage |
|---|---:|
| Projection | 12.1% |
| Attention | 61.7% |
| FFN | 24.2% |
| LM head | 2.0% |

因此，当 sequence length 很大时：attention 的二次复杂度：
$$
O(T^2D)
$$

会成为 Transformer 的主要计算瓶颈。
	
### How much peak memory does running AdamW require?
具体答案懒得算了,感觉会很麻烦,可以问AI;
总的来说是四部分组成
1. 模型参数
2. 激活内存(中间结果)
3. 梯度内存 因为grad tensor和参数tensor一样大，所以也是模型参数的大小
4. 优化器状态 adamw需要为每个参数维护一个一阶和一个二阶矩，所以优化器状态的内存是模型参数的两倍


### What is the maximum batch size you can use and still fit within 80GB memory?
a⋅batch_size+b <= 显存

其中b是模型参数内存+梯度内存+优化器状态,因为这些和batch size无关

### How many FLOPs does running one step of AdamW take?
约为14倍模型参数(主要开销是更新模型参数,计算1,2阶矩和学习率)

### Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100?
考虑H100,the total training time is approximately 4836 hours, or about 201.5 days. 




