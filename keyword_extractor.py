import re
import math
import collections

# ============================================
# === 队员C：关键词提取模块（函数：extract_keywords）===
# ============================================

def extract_keywords(text):
       try:
        if not text or not text.strip():
            return {}

        text = text.strip()
        sentences_raw = re.split(r'[。！？；.!?;]+', text)
        sentences = [s.strip() for s in sentences_raw if s.strip()]
        if not sentences:
            return {}

        def tokenize(sentence):
            tokens_raw = re.split(r'[\s\u3000\uff0c,，、：:""''《》【】（）()—/\\@#$%^&*+=|~`-]+', sentence)
            tokens = []
            for t in tokens_raw:
                t = t.strip()
                if len(t) &lt; 2:
                    continue
                if re.fullmatch(r'[\d\s]+', t):
                    continue
                if re.fullmatch(r'[^\u4e00-\u9fa5a-zA-Z0-9]+', t):
                    continue
                tokens.append(t)
            return tokens

        sentence_tokens = [tokenize(s) for s in sentences]

        all_words = set()
        for tokens in sentence_tokens:
            all_words.update(tokens)
        if not all_words:
            return {}

        total_sentences = len(sentence_tokens)

        idf_map = {}
        for word in all_words:
            doc_count = sum(1 for tokens in sentence_tokens if word in tokens)
            idf_map[word] = math.log(total_sentences / (doc_count + 1)) + 1

        max_tfidf = collections.defaultdict(float)
        for tokens in sentence_tokens:
            if not tokens:
                continue
            total_words_in_sent = len(tokens)
            word_count = collections.Counter(tokens)
            for word, count in word_count.items():
                tf = count / total_words_in_sent
                tfidf = tf * idf_map[word]
                if tfidf &gt; max_tfidf[word]:
                    max_tfidf[word] = tfidf

        sorted_words = sorted(max_tfidf.items(), key=lambda x: x[1], reverse=True)
        return {word: round(score, 3) for word, score in sorted_words[:10]}

    except Exception:
        # 任何异常均返回空字典
        return {}

# ============================================
# === 队员C 模块结束 ===
# ============================================