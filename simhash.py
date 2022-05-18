import hashlib

def near_match_hashes(hash1, hash2, threshold = 0.9):
    similarity = 1 - '{0:80b}'.format(hash1 ^ hash2).count("1") / 128.0
    return similarity > threshold  # 95% similar

def near_match_tokens(tokens1, tokens2, threshold = 0.9):
    hash1 = compute_simhash(tokens1)
    hash2 = compute_simhash(tokens2)
    return near_match_hashes(hash1, hash2, threshold)

def compute_simhash(tokens):
    hash_ls = []
    for token in tokens:
        hash_ls.append(hashlib.md5(token.encode()))

    hash_int_ls = []
    for hash in hash_ls:
        hash_int_ls.append(int(hash.hexdigest(), 16))

    res = 0
    for i in range(128):
        sum_ = 0
        for h in hash_int_ls:
            if h >> i & 1 == 1:
                sum_ += 1
            else:
                sum_ += -1
        if sum_ > 1:
            sum_ = 1
        else:
            sum_ = 0

        res += sum_ * 2 ** i
    return res