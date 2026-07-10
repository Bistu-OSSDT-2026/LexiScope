def get_word_freq(text):
    if not text or not isinstance(text, str):
        return {}

    words = jieba.lcut(text)
    valid_words = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if w in STOPWORDS:
            continue
        if re.fullmatch(r'[^\u4e00-\u9fa5a-zA-Z0-9]+', w):
            continue
        valid_words.append(w)

    if not valid_words:
        return {}

    freq = {}
    for w in valid_words:
        freq[w] = freq.get(w, 0) + 1

    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    return dict(sorted_items)
