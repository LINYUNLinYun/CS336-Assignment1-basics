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

一个一个来算，先算内存：
- 一个单精度浮点数是4 Bytes
- memory of embedding layer：$vocab_size \times d_model = 50,257 \times 1,600 = 80,411,200$, takes 80,411,200 * 4 Bytes = 321,644,800 Bytes = 306.74MB
- memory of each transformer block: 
  - mha: 
    - linear layers: $d_model \times d_model \times 3 + d_model \times d_model = 1,600 \times 1,600 \times 3 + 1,600 \times 1,600 = 10240000$, takes 10,240,000 * 4 Bytes = 40,960,000 Bytes = 39.06MB
    - rope embedding: $context_length \times \frac{d_model}{2} = 1,024 \times 800 = 819,200$, takes 819,200 * 4 Bytes = 3,276,800 Bytes = 3.12MB
  - fnn: $d_model \times d_ff + d_ff \times d_model + d_ff \times d_model = 1600\times 4,288 + 4,288 \times 1,600 + 4,288 \times 1600 = 20582400$, takes 20,582,400 * 4 Bytes = 82,329,600 Bytes = 78.51MB
  - rmsnorm: takes 12.8KB（可忽略不计）
- memory of transformer blocks: 48 * (39.06MB + 3.12MB + 78.51MB) = 48 * 117.83MB = 5.52 GB
- memory of output layer: $d_model \times vocab_size = 1,600 \times 50,257 = 80,411,200$, takes 80,411,200 * 4 Bytes = 321,644,800 Bytes = 306.74MB
- total memory: 306.74MB + 5.52GB + 306.74MB = 6.13GB

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