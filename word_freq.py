import jieba

def get_word_freq(text):
    if not text or not isinstance(text, str):
        return {}
    
    stopwords = {
        '的', '了', '是', '我', '你', '他', '她', '它', '们', '在', '有', '和',
        '与', '或', '但', '而', '等', '个', '把', '被', '让', '给', '从', '去',
        '上', '下', '不', '也', '都', '就', '还', '要', '可以', '能', '会', '来',
        '说', '看', '听', '想', '做', '吃', '喝', '玩', '乐', '走', '跑', '跳',
        '坐', '站', '躺', '开', '关', '大', '小', '多', '少', '很', '太', '更',
        '最'
    }
    
    words = jieba.lcut(text)
    valid_words = [w for w in words if w.strip() and w not in stopwords]
    
    if not valid_words:
        return {}
    
    freq = {}
    for w in valid_words:
        freq[w] = freq.get(w, 0) + 1
    
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    return dict(sorted_items)