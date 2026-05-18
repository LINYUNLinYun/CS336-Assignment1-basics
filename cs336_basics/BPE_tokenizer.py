import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import defaultdict,Counter
from multiprocessing import Pool
import json
import os
from tqdm import tqdm
from typing import Iterator,Iterable


class BPE_tokenizer_trainer():
    def __init__(self, input_path: str = 'data/TinyStoriesV2-GPT4-valid.txt', vocab_size: int = 0, special_tokens: list[str] = ['<|endoftext|>']):
        """
            参数: 语料库路径、词表大小(这决定要做多少次merge)、特殊token
        """
        # 输入参数
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        # 输出参数
        self.vocab = {}
        self.merges = []

        # 中间结果
        # self._bytes_pair_freq :dict[tuple[bytes, bytes], list] = {}
        self.bytes_pair_freq = defaultdict(int)
        self.bytes_pair_to_word = defaultdict(set)
        self.new_words = []
        self.pre_tokenized :dict[tuple[bytes, ...], int] =  None

        for i in range(256):
            self.vocab[i] = bytes([i])

        # 特殊token的处理
        if(special_tokens):
            self.pattern = "|".join(re.escape(tok) for tok in special_tokens)
            vocab_id = len(self.vocab)
            for tok in special_tokens:
                self.vocab[vocab_id] = tok.encode("utf-8")
                vocab_id+=1
        else:
            self.pattern = None
        
    @staticmethod
    def pre_tokenize(self,start,end):
        pre_tokenized :dict[tuple[bytes, ...], int] = {}
        with open(self.input_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # 把一个chunk的按eof分成多个文本
            docs = re.split(self.pattern, chunk)
            for doc in docs:
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for m in re.finditer(PAT,doc):
                    # pre_tokenized.append(m.group())
                    word = m.group()
                    bytes_seq = word.encode("utf-8")
                    key = tuple(bytes([b]) for b in bytes_seq)
                    
                        
                    if(key in pre_tokenized):
                        pre_tokenized[key]+=1
                    else:
                        pre_tokenized[key]=1
        return pre_tokenized
    
    def _save_vocab(self, output_path = "vocab.json"):
        root_dir = os.getcwd()
        output_path = root_dir + '/data/' + output_path
        serializable_vocab = {
            str(token_id): list(token_bytes)
            for token_id, token_bytes in self.vocab.items()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

        print(f"vocab json file have saved in {output_path}")

    def _save_merges(self, output_path = "merges.json"):
        root_dir = os.getcwd()
        output_path = root_dir + '/data/' + output_path
        serializable_merges = [
            [list(left), list(right)]
            for left, right in self.merges
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f, ensure_ascii=False, indent=2)
            

    def train(self, num_processes :int = 8) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
        
        with open(self.input_path, "rb") as f:
            # 分块是用于并行处理
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            with Pool(processes=num_processes) as pool:
                dicts = pool.starmap(
                    self.pre_tokenize,
                    [
                        (self,start, end)
                        for start, end in zip(boundaries[:-1], boundaries[1:])
                    ],
                )
        total_counter = Counter()
        for d in dicts:
            total_counter.update(Counter(d))
        self.pre_tokenized = dict(total_counter)


        vocab_n = len(self.vocab)
        for i in tqdm(range(self.vocab_size - vocab_n)):
            merged_pair = self.merge()
            self.merges.append(merged_pair)
            self.vocab[vocab_n] = merged_pair[0] + merged_pair[1]
            vocab_n+=1
        self._save_vocab()
        self._save_merges()
        return self.vocab, self.merges
    
    @staticmethod
    def _merge_one_word(
            word: tuple[bytes,...],
            best_pair:tuple[bytes, bytes],
            ) -> tuple[bytes, ...]:
        new_word = []
        i = 0
        while(i < len(word)):
            if(i+1< len(word) and best_pair[0] == word[i] and best_pair[1] == word[i+1]):
                new_word.append(best_pair[0]+best_pair[1])
                i+=1
            else:
                new_word.append(word[i])
            i+=1
        return tuple(new_word)
    
    @staticmethod
    def _pair_counter(word:tuple[bytes,...]):
        c = Counter()
        for i in range(1, len(word)):
            c[(word[i - 1], word[i])] += 1
        return c

    def merge(
            self, 
            # pre_tokenized :dict[tuple[bytes, ...], int] | None = None,
            # last_merged_pair: tuple[bytes, bytes] | None = None,
            ) -> tuple[bytes,bytes]:
    
        """
            传入预分词后的字典 返回合并了一个pair的字典 和 合并的pair
        """
        # 第一轮先建表
        if len(self.bytes_pair_freq) == 0 and len(self.bytes_pair_to_word) == 0:
            for word,freq in self.pre_tokenized.items():
                if(len(word) < 2):
                    # 跳过不构成pair的
                    continue
                for pair, count in self._pair_counter(word).items():
                    # 第一个元素是频数
                    self.bytes_pair_freq[pair] += freq*count
                    # 记录一下哪个word让这个频数增加了
                    self.bytes_pair_to_word[pair].add(word)


        # 提取出频数最大的一个或几个pair
        max_freq = max(self.bytes_pair_freq.values())
        # python 奇怪的语法糖让这个变得很简单
        candidates = [pair for pair, freq in self.bytes_pair_freq.items() if freq == max_freq] 
        # 然后选出字典序最大的key即可
        best_pair = max(candidates)
        
        # new_pre_tokenized :dict[tuple[bytes, ...], int] = {}
        # 清空上一轮相关的词
        self.new_words = []
        words_inlcude_best_pair = self.bytes_pair_to_word[best_pair].copy()
    
        for word in words_inlcude_best_pair:
            freq = self.pre_tokenized[word]
            # 由于即将要合并 先移除old word的统计
            for pair, count in self._pair_counter(word).items():
                self.bytes_pair_freq[pair] -= freq*count
                self.bytes_pair_to_word[pair].remove(word)


            new_word = self._merge_one_word(word,best_pair)
            # 更新新词的统计
            for pair, count in self._pair_counter(new_word).items():
                self.bytes_pair_freq[pair] = self.bytes_pair_freq.get(pair,0) +freq*count
                self.bytes_pair_to_word[pair].add(new_word)
            
            self.pre_tokenized[new_word] = self.pre_tokenized[word]
            self.pre_tokenized.pop(word)

        self.bytes_pair_freq.pop(best_pair)
        self.bytes_pair_to_word.pop(best_pair)

        return best_pair

class BPE_tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.special_tokens = special_tokens
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }
        self.token_to_id: dict[bytes, int] = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = cls._load_vocab_from_json(vocab_filepath)
        merges = cls._load_merges_from_json(merges_filepath)
        return BPE_tokenizer(vocab,merges,special_tokens)
    @staticmethod
    def _load_vocab_from_json(path: str) -> dict[int, bytes]:
        with open(path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            int(token_id): bytes(byte_list)
            for token_id, byte_list in raw_vocab.items()
        }
        return vocab
    
    @staticmethod
    def _load_merges_from_json(path: str) -> list[tuple[bytes, bytes]]:
        with open(path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges = [
            (bytes(left), bytes(right))
            for left, right in raw_merges
        ]
        return merges
    def _merge_one_pair(self,bytes_list:list)->list | None:
        if(len(bytes_list) < 2):
            return None
        # 找最先合并的pair
        best_pair = None
        best_rank = float("inf")

        for i in range(len(bytes_list) - 1):
            pair = (bytes_list[i], bytes_list[i + 1])
            rank = self.merge_ranks.get(pair)

            if rank is not None and rank < best_rank:
                best_pair = pair
                best_rank = rank

        # 没有pair可以merge
        if best_pair is None:
            return None
        i = 0
        new_bytes_list = []

        while i < len(bytes_list):
            if (
                i < len(bytes_list) - 1
                and bytes_list[i] == best_pair[0]
                and bytes_list[i + 1] == best_pair[1]
            ):
                new_bytes_list.append(bytes_list[i] + bytes_list[i + 1])
                i += 2
            else:
                new_bytes_list.append(bytes_list[i])
                i += 1

        return new_bytes_list
        
    def _encode_text(self,  text: str):
        # bytes_str = str.encode()
        tokens_seq = self.pre_tokenize(text)
        # print(tokens_seq)
        encoded_tokens_seq = []
        for token in tokens_seq:
            temp_token = token
            # 合并到没有pair能合并为止
            while True:
                result =  self._merge_one_pair(temp_token)
                if(result!= None):
                    temp_token = result
                else:
                    # print(temp_token)
                    encoded_tokens_seq.extend(
                        self.token_to_id[token_bytes]
                        for token_bytes in temp_token
                    )
                    break
            # merged_tokens_seq.append(temp_token)
            # break
        return encoded_tokens_seq
    
    def encode(self,  text: str):
        if(self.special_tokens == None):
            return self._encode_text(text)

        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_token_set = set(self.special_tokens)
        # 带括号的正则 会在分割后保留special toks
        pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        parts = re.split(pattern, text)

        encoded_tokens_seq = []

        for part in parts:
            if part == "":
                continue
            if part in special_token_set:
                encoded_tokens_seq.append(self.token_to_id[part.encode("utf-8")])
            else:
                encoded_tokens_seq.extend(self._encode_text(part))

        return encoded_tokens_seq
        


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """内存高效的"""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_seq = bytes()
        for id in ids:
            bytes_seq+= self.vocab[id]
        return bytes_seq.decode(errors="replace")

    @staticmethod
    def pre_tokenize(text):
        tokens_seq = []
        
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for m in re.finditer(PAT,text):
            # pre_tokenized.append(m.group())
            word = m.group()
            bytes_seq = word.encode("utf-8")
            bytes_list = [bytes([b]) for b in bytes_seq]
            tokens_seq.append(bytes_list)
            
        return tokens_seq


"""
以下为测试用函数 
"""
def test_bpe_tokenizer():
    import random
    SPECIAL_TOKEN = "<|endoftext|>"
    def compute_compression_ratio(tokenizer: BPE_tokenizer,docs: list[str],
        ) -> tuple[float, int, int]:
        """
        计算 bytes/token。

        返回：
        ratio: bytes per token
        total_bytes: UTF-8 bytes 总数
        total_tokens: token 总数
        """
        total_bytes = 0
        total_tokens = 0

        for doc in docs:
            byte_len = len(doc.encode("utf-8"))
            ids = tokenizer.encode(doc)
            total_bytes += byte_len
            total_tokens += len(ids)

        if total_tokens == 0:
            raise ValueError("Encoded token count is zero.")

        ratio = total_bytes / total_tokens
        return ratio, total_bytes, total_tokens
    
    def sample_documents(input_path: str,num_docs: int = 10,
                         special_token: str = SPECIAL_TOKEN,seed: int = 42,
        ) -> list[str]:
        """
        从数据文件中随机采样 num_docs 篇文档。
        TinyStories / OpenWebText 文件中一般用 <|endoftext|> 分隔文档。
        """
        random.seed(seed)

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = text.split(special_token)
        # 去掉空文档
        docs = [doc for doc in docs if doc.strip()]

        if len(docs) < num_docs:
            raise ValueError(f"Only found {len(docs)} documents, but need {num_docs}.")

        return random.sample(docs, num_docs)
    
    
    tinystories_data_path = "data/TinyStoriesV2-GPT4-valid.txt"
    owt_data_path = "data/owt_valid.txt"
    tinystories_vocab_path = "data/vocab.json"
    tinystories_merges_path = "data/merges.json"

    owt_vocab_path = "data/owt_vocab/vocab.json"
    owt_merges_path = "data/owt_vocab/merges.json"

    special_tokens = [SPECIAL_TOKEN]
    tinystories_tokenizer = BPE_tokenizer.from_files(
        tinystories_vocab_path,tinystories_merges_path,special_tokens)
    owt_tokenizer = BPE_tokenizer.from_files(
        owt_vocab_path,owt_merges_path,special_tokens)

    tinystories_docs = sample_documents(
        input_path=tinystories_data_path,
        num_docs=10,
        special_token=SPECIAL_TOKEN,
        seed=42,
    )

    owt_docs = sample_documents(
        input_path=owt_data_path,
        num_docs=10,
        special_token=SPECIAL_TOKEN,
        seed=42,
    )

    tinystories_ratio, tinystories_bytes, tinystories_tokens = compute_compression_ratio(
        tokenizer=tinystories_tokenizer,
        docs=tinystories_docs,
    )

    owt_ratio, owt_bytes, owt_tokens = compute_compression_ratio(
        tokenizer=owt_tokenizer,
        docs=owt_docs,
    )

    print("========== Compression ratio ==========")

    print("\nTinyStories docs with TinyStories tokenizer:")
    print(f"total bytes:  {tinystories_bytes}")
    print(f"total tokens: {tinystories_tokens}")
    print(f"bytes/token:  {tinystories_ratio:.4f}")

    print("\nOpenWebText docs with OpenWebText tokenizer:")
    print(f"total bytes:  {owt_bytes}")
    print(f"total tokens: {owt_tokens}")
    print(f"bytes/token:  {owt_ratio:.4f}")
    


    owt_with_tinystories_ratio, owt_with_tinystories_bytes, owt_with_tinystories_tokens = (
        compute_compression_ratio(
            tokenizer=tinystories_tokenizer,
            docs=owt_docs,
        )
    )

    print("\n========== OWT encoded by TinyStories tokenizer ==========")

    print("\nOpenWebText docs with TinyStories tokenizer:")
    print(f"total bytes:  {owt_with_tinystories_bytes}")
    print(f"total tokens: {owt_with_tinystories_tokens}")
    print(f"bytes/token:  {owt_with_tinystories_ratio:.4f}")

    print("\nComparison on the same OpenWebText sample:")
    print(f"OWT tokenizer bytes/token:         {owt_ratio:.4f}")
    print(f"TinyStories tokenizer bytes/token: {owt_with_tinystories_ratio:.4f}")

    diff = owt_ratio - owt_with_tinystories_ratio
    relative = diff / owt_ratio * 100

    print(f"absolute difference: {diff:.4f} bytes/token")
    print(f"relative difference: {relative:.2f}%")

    if owt_with_tinystories_ratio < owt_ratio:
        print(
            "\nResult: TinyStories tokenizer gives worse compression on OpenWebText "
            "because it produces fewer bytes per token, meaning more tokens are needed."
        )
    else:
        print(
            "\nResult: TinyStories tokenizer did not compress worse on this small sample, "
            "but this may be due to sampling noise."
        )
    


def test_bpe_trainer():
    import time
    import psutil
    import resource
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()



    start_time = time.perf_counter()

    bpe_trainer = BPE_tokenizer_trainer(input_path= 'data/TinyStoriesV2-GPT4-valid.txt', vocab_size=1000)
    vocab,_ = bpe_trainer.train(num_processes=32)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    longest_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1])
    )
    print("--------------------------------------------------------\n")
    print("longest_id:", longest_id)
    print("the_NO.400_longest_id:", vocab[400])
    print("longest_token bytes:", longest_token)
    print("longest_token length:", len(longest_token))
    print("longest_token decoded:", longest_token.decode("utf-8", errors="replace"))
    print("--------------------------------------------------------\n")
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Training time: {elapsed / 60:.2f} minutes")

    process = psutil.Process(os.getpid())

    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"RSS memory: {memory_mb:.2f} MB")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_memory_mb = usage.ru_maxrss / 1024

    print(f"Peak RSS memory: {peak_memory_mb:.2f} MB")
    # vocab = list(vocab)
    # print(_)

def estimate_tokenizer_throughput_with_encode_iterable(tokenizer:BPE_tokenizer, input_path: str):
    import time
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    total_bytes = len(text.encode("utf-8"))

    start_time = time.perf_counter()

    # total_tokens = len(tokenizer.encode(text))
    # tok = 

    for _ in tokenizer.encode_iterable([text]):
        total_tokens += 1

    

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    throughput = total_bytes / elapsed

    print("total bytes:", total_bytes)
    print("total tokens:", total_tokens)
    print(f"elapsed time: {elapsed:.4f} seconds")
    print(f"throughput: {throughput:.2f} bytes/second")
    print(f"throughput: {throughput / 1024 / 1024:.2f} MB/s")

    pile_bytes = 825 * 1024 ** 3
    pile_seconds = pile_bytes / throughput

    print(f"estimated Pile time: {pile_seconds / 3600:.2f} hours")
    print(f"estimated Pile time: {pile_seconds / 3600 / 24:.2f} days")

if __name__ == '__main__':
    # test_encoding_decoding()
    # exit(0)
    # test_bpe_trainer()
    # test_bpe_tokenizer()
    tinystories_vocab_path = "data/vocab.json"
    tinystories_merges_path = "data/merges.json"

    owt_vocab_path = "data/owt_vocab/vocab.json"
    owt_merges_path = "data/owt_vocab/merges.json"

    special_tokens = ["<|endoftext|>"]

    corpus = 'data/TinyStoriesV2-GPT4-train.txt','data/TinyStoriesV2-GPT4-valid.txt'
    corpus = 'data/owt_train.txt','data/owt_valid.txt'

    tinystories_tokenizer = BPE_tokenizer.from_files(
        tinystories_vocab_path,tinystories_merges_path,special_tokens)
    owt_tokenizer = BPE_tokenizer.from_files(
        owt_vocab_path,owt_merges_path,special_tokens)
    
    import numpy as np
    for c in corpus:
        with open(c, "r", encoding="utf-8") as f:
            text = f.read()
        ids = owt_tokenizer.encode(text)
        ids = np.array(ids, dtype=np.uint16)
        # print(tinystories_valid_ids)
        np.save(c.split('.')[0]+".npy", ids)
    

    # estimate_tokenizer_throughput_with_encode_iterable(tinystories_tokenizer,'data/TinyStoriesV2-GPT4-valid.txt')
    


    

    









