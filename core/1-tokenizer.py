class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _get_stats(self, ids):
        counts={}
        for i in range(len(ids)-1):
            pair=(ids[i], ids[i+1])
            counts[pair]=counts.get(pair, 0)+1
        return counts
    
    def _merge(self, ids, pair, idx):
        new_ids=[]
        i=0
        while i < len(ids):
            if i < len(ids) -1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        return new_ids
    
    def train(self, text, vocab_size):
        tokens = list(text.encode('utf-8'))
        num_merges = vocab_size-256

        for i in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats: break
            pair = max(stats, key=stats.get)
            idx=256+i

            tokens = self._merge(tokens, pair, idx)
            self.merges[pair]=idx
            self.vocab[idx]=self.vocab[pair[0]]+self.vocab[pair[1]]
            print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}")

    def decode(self, ids):
        part_bytes = [self.vocab[idx] for idx in ids]
        full_bytes = b"".join(part_bytes)
        return full_bytes.decode("utf-8", errors="replace")
        #当解码过程中遇到无法识别的、不符合 UTF-8 规则的字节时，不抛出错误崩溃，
        # 而是用一个 “替代字符”（默认是 �，读作 “替换符”）来代替这些错误的字节。
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, idx in self.merges.items():
            tokens = self._merge(tokens, pair, idx)
        return tokens
    
    def save(self, file_prefix):
        with open(f"{file_prefix}_vocab.txt", "w", encoding="utf-8") as f:
            for idx, byte_seq in self.vocab.items():
                f.write(f"{idx}\t{byte_seq.decode('utf-8', errors='replace')}\n")
        with open(f"{file_prefix}_merges.txt", "w", encoding="utf-8") as f:
            for pair, idx in self.merges.items():
                f.write(f"{idx}\t{pair[0]}\t{pair[1]}\n")

if __name__ == "__main__":
    tokenizer = BasicTokenizer()
    text = "hello world! this is a test. hello world!"
    tokenizer.train(text, vocab_size=300)
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)