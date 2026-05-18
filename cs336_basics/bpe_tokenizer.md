# BPE tokenizer 

## 内容：

预分词就是先用规则把文本粗切成不跨越空格、标点、特殊 token 等边界的小块；BPE 训练只在这些小块内部学习高频 byte 序列，从而既提高效率，又避免学出太多不合理的跨边界 token。

- 注意用encode编码字符串
- 处理正则时使用finditer而不是findall实现懒加载，节省内存
- metadata 元信息处理：如<|endoftext|>（表示文本的结尾）等特殊 token，预分词时直接当成一个 token 处理，不让 BPE 学习跨越它的 token
- 使用re.escape处理特殊 token，避免正则表达式解析错误
- 预分词的时候用multiprocessing加速，merge就不行了，因为每一次merge的输入都依赖于上次merge的结果

(a) What Unicode character does chr(0) return?

返回一个空字符（null）

(b) How does this character’s string representation (__repr__()) differ from its printed 
representation?

print(chr(0))什么也没有，而 repr(chr(0)) 输出 "'\x00'"，因为终端打印时会忽略空字符

(a) Why prefer UTF-8 bytes over UTF-16 or UTF-32 for tokenizer training?

utf-8常见字符就一个字节，节省空间；utf-16两个，utf-32四个，因为bpe是在byte序列上统计规律，所以用utf8合适，高频的字符占用更少的空间。

(b) Why is this function incorrect?

对于多字节的utf-8字符，比如汉字'牛'，它的utf-8编码是三个字节，不能分别对每个字节进行解码再拼起来，这样得到的就不是原来的自负了。

(c) Give a two-byte sequence that does not decode to any Unicode character(s).

b'\x80\x80'

## 编程实现

写完直接开始测试，在uv里面跑的连环境都不用自己配了，好评：
```shell
uv run pytest tests/test_train_bpe.py
```

碰到了个问题，服务器上下载uv需要的包太慢了，使用端口转发挂上本地的代理软件下载：

先修改一下本地的ssh配置文件（~/.ssh/config）：

```ssh
Host server_name
    HostName server_ip
    User username
    RemoteForward 7890 127.0.0.1:7890
```
这样服务器的7890端口就会开始监听，然后把通过ssh把这个端口转发到本地的7890端口上，7890端口是clash的代理端口，如此一来就能让服务器访问本地的代理软件了。再在终端运行：

```shell
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
curl -I -x http://127.0.0.1:7890 https://pypi.org
```
能看到输出就是代理成功了。


### 第一次测试 全部失败了
首先是测试没达标，要求测试时间不超过1.5秒，结果是：2.61s；然后生成的vocab和pair也不一样，后面两个也跟着错了

```shell
tests/test_train_bpe.py::test_train_bpe_speed FAILED
tests/test_train_bpe.py::test_train_bpe FAILED
tests/test_train_bpe.py::test_train_bpe_special_tokens FAILED
```
- 之前的代码结构太混乱了，重构了下就能passed了

解决超时问题，引入增量更新即可。

```shell
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

### 运行结果与分析
（a）资源占用分析：

8个进程一起处理，总占用内存130+MB，峰值143MB，训练时间2.93分钟，还是挺快的。最长的token是b' accomplishment'
```shell
100%|█████████████████████████████████████████████████████| 9743/9743 [00:32<00:00, 300.00it/s]
vocab json file have saved in /root/project/CS336-assignment/assignment1-basics/data/vocab.json
--------------------------------------------------------

longest_id: 7160
the_NO.400_longest_id: b'ound'
longest_token bytes: b' accomplishment'
longest_token length: 15
longest_token decoded:  accomplishment
--------------------------------------------------------

Training time: 176.05 seconds
Training time: 2.93 minutes
RSS memory: 130.70 MB
Peak RSS memory: 143.24 MB
```
（b）代码性能分析：

cprofile结果如下：瓶颈主要集中在几个地方，最大头的是进程的等待上，这是因为预分词的多进程处理的是很慢的，这是最可以优化的地方；其次是merge操作，总用时37.102s，其中merge里面一大的耗时操来自max()排序，然后比较值得注意的是_pair_counter调用了 615405 次，_merge_one_word是277780 次，都是高频操作。
```shell
16918174 function calls (16656656 primitive calls) in 329.790 seconds

   Ordered by: cumulative time
   List reduced from 491 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  604/600  288.822    0.478  580.483    0.967 {built-in method posix.read}
    22/18    0.000    0.000  580.479   32.249 connection.py:391(_recv)
        1    0.000    0.000  291.751  291.751 pool.py:738(__exit__)
        1    0.000    0.000  291.748  291.748 pool.py:654(terminate)
        9    0.000    0.000  291.738   32.415 util.py:276(__call__)
        1    0.000    0.000  291.738  291.738 pool.py:680(_terminate_pool)
        9    0.003    0.000  291.674   32.408 connection.py:247(recv)
      346    0.002    0.000  291.662    0.843 {method 'acquire' of '_multiprocessing.SemLock' objects}
        1    0.000    0.000  291.662  291.662 pool.py:671(_help_stuff_finish)
      3/1    0.000    0.000  291.661  291.661 threading.py:1001(_bootstrap)
      3/1    0.000    0.000  291.661  291.661 threading.py:1028(_bootstrap_inner)
     11/9    0.000    0.000  291.661   32.407 connection.py:430(_recv_bytes)
      3/1    0.000    0.000  291.661  291.661 threading.py:984(run)
        1    0.000    0.000  291.661  291.661 pool.py:573(_handle_results)
        1    0.000    0.000  291.659  291.659 pool.py:527(_handle_tasks)
      305    0.001    0.000  291.470    0.956 pool.py:500(_wait_for_updates)
      613    0.004    0.000  291.468    0.475 connection.py:1156(wait)
     2448    0.004    0.000  290.823    0.119 process.py:224(exitcode)
      305    0.002    0.000  288.731    0.947 pool.py:333(_maintain_pool)
      305    0.002    0.000  288.729    0.947 pool.py:289(_join_exited_workers)
     10/8    0.013    0.001   37.863    4.733 threading.py:641(wait)
     9743   23.959    0.002   37.032    0.004 BPE_tokenizer.py:142(merge)
      8/7    0.040    0.005   30.016    4.288 threading.py:327(wait)
      2/1    0.010    0.005    7.834    7.834 BPE_tokenizer.py:90(train)
    20181    7.119    0.000    7.119    0.000 {built-in method builtins.max}
        1    0.000    0.000    2.956    2.956 pool.py:369(starmap)
   615405    1.908    0.000    2.896    0.000 BPE_tokenizer.py:135(_pair_counter)
      613    0.001    0.000    2.622    0.004 selectors.py:385(select)
      613    2.620    0.004    2.620    0.004 {method 'poll' of 'select.poll' objects}
   277780    1.026    0.000    1.543    0.000 BPE_tokenizer.py:119(_merge_one_word)
```



### 在OpenWebText数据集上训练BPE tokenizer
先看下server有多少cpu: 物理核心64，逻辑核心128，试试32个进程训练2G文本——时间缩短到了2分钟以内，果然核心越多越强但我也不打算开更多了以免cpu之间竞争得不偿失，文档种有提到**Suppose we want to tokenize a large text file that we cannot fit in memory.**但是我的server有500G内存不用进行分块，所以我就不做文档分块了：
```shell
lscpu
Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             43 bits physical, 48 bits virtual
  Byte Order:                Little Endian
CPU(s):                      128
  On-line CPU(s) list:       0-127
Thread(s) per core:          2
Core(s) per socket:          32
Socket(s):                   2

free -h
               total        used        free      shared  buff/cache   available
Mem:           503Gi        63Gi        47Gi       187Mi       393Gi       437Gi
Swap:          8.0Gi       3.6Gi       4.4Gi
```
然后直接开始在owt数据集上训练，这两个数据集的差异还是比较大的，一个是由一段段儿童故事组成的，最长的token是b' accomplishment'，儿童故事中的常用词汇；另一个是互联网上的文本，有些网络文本出现比较多。

一个神奇的发现是，由于vocab size一跃而至32000，且预分词阶段可以用多进程加速，所以训练的瓶颈反而集中在了merge阶段了，这是因为merge阶段是单线程的，且每次merge都要对整个vocab进行一次排序，时间复杂度较高。训练预计要十几个小时

```shell
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 31743/31743
 [2:09:15<00:00,  4.09it/s]
vocab json file have saved in /root/project/CS336-assignment/assignment1-basics/data/vocab.json
         2668751558 function calls (2667870292 primitive calls) in 8229.492 seconds

   Ordered by: cumulative time
   List reduced from 491 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  775/773   16.936    0.022 9054.370   11.713 threading.py:641(wait)
  773/772   85.631    0.111 7151.316    9.263 threading.py:327(wait)
    31743 5064.448    0.160 6997.096    0.220 BPE_tokenizer.py:142(merge)
   121711 1114.518    0.009 1114.518    0.009 {built-in method builtins.max}
   166720    0.148    0.000  869.124    0.005 process.py:224(exitcode)
 78664700  349.052    0.000  532.549    0.000 BPE_tokenizer.py:135(_pair_counter)
3099/3090  265.539    0.086  474.118    0.153 {method 'acquire' of '_thread.lock' objects}
        1    0.000    0.000  442.138  442.138 pool.py:738(__exit__)
        1    0.000    0.000  442.135  442.135 pool.py:654(terminate)
       33    0.000    0.000  442.100   13.397 util.py:276(__call__)
        1    0.000    0.000  442.099  442.099 pool.py:680(_terminate_pool)
    29876    0.059    0.000  441.480    0.015 {method 'acquire' of '_multiprocessing.SemLock' objects}
        1    0.000    0.000  441.194  441.194 pool.py:671(_help_stuff_finish)
       64    0.000    0.000  441.191    6.894 connection.py:203(send)
       69    0.000    0.000  441.188    6.394 connection.py:407(_send_bytes)
       69    0.000    0.000  441.187    6.394 connection.py:382(_send)
       69    0.002    0.000  441.186    6.394 {built-in method posix.write}
      3/1    0.000    0.000  441.186  441.186 threading.py:1001(_bootstrap)
      3/1    0.000    0.000  441.186  441.186 threading.py:1028(_bootstrap_inner)


--------------------------------------------------------

longest_id: 25822
the_NO.400_longest_id: b' ne'
longest_token bytes: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82
\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x8
3\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
longest_token length: 64
longest_token decoded: ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ
--------------------------------------------------------

Training time: 8229.51 seconds
Training time: 137.16 minutes
RSS memory: 10601.76 MB
Peak RSS memory: 12260.70 MB
```
实际只用了两个多小时，12G内存。还行。最长的token是一个乱码字符，也很正常毕竟网页本身就存在很多的乱码

### encoding和decoding
分别从两个数据集中抽取10个文本，然后看看两个数据集在不同的vocab size下计算出来的vocab.json的压缩比差异：
```shell
========== Compression ratio ==========

TinyStories docs with TinyStories tokenizer:
total bytes:  8482
total tokens: 2785
bytes/token:  3.0456

OpenWebText docs with OpenWebText tokenizer:
total bytes:  39375
total tokens: 9275
bytes/token:  4.2453

========== OWT encoded by TinyStories tokenizer ==========

OpenWebText docs with TinyStories tokenizer:
total bytes:  39375
total tokens: 18593
bytes/token:  2.1177

Comparison on the same OpenWebText sample:
OWT tokenizer bytes/token:         4.2453
TinyStories tokenizer bytes/token: 2.1177
absolute difference: 2.1276 bytes/token
relative difference: 50.12%

Result: TinyStories tokenizer gives worse compression on OpenWebText because it produces fewer bytes per token, meaning more tokens are needed.
```
显而易见，owt数据集的vocab size更大，token更细粒度，所以compression ratio更高，且owt数据中有更多重复的长模式，比如 URL、网页片段、常见互联网表达，更容易被压缩

用tinystories的tokenizer去编码owt数据集，ratio从4.2453降到了2.1177，这也是可以预测到的，毕竟前者只是个儿童数据集训练出来的分词器，不太适用于网页文本。

**结论：**：TinyStories tokenizer 编码 OWT 时压缩率明显变差，说明 tokenizer 强依赖训练语料分布；用儿童故事训练出来的 tokenizer 不适合互联网网页文本。

### 在owt_valid数据集上计算encode的吞吐量
使用传入迭代器的encode iterable函数，因为考虑到900+GB的Pile数据集根本不可能一次性加载到内存中，所以encode函数必须支持流式处理。
```shell
total bytes: 289998753
total tokens: 135074968
elapsed time: 494.7200 seconds
throughput: 586187.63 bytes/second
throughput: 0.56 MB/s
estimated Pile time: 419.77 hours
estimated Pile time: 17.49 days
```
结果是OpenWebText验证集上的吞吐量大约是0.56 MB/s。按照这个速度，tokenize 825GB的Pile 数据集大约需要 419.77小时……感觉有什么地方不对劲；实测换成encode函数，吞吐量提升到0.62MB/s，依旧不理想原因有待查明。



