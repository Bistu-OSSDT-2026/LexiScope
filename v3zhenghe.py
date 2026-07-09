# ============================================================
# 导入语句
# ============================================================

import re
import math
import collections
import os
import zipfile
from io import BytesIO
from flask import Flask, request, jsonify
import jieba
from transformers import pipeline

# ============================================================
# 全局停用词表（B和C共用，统一标准）
# ============================================================

STOPWORDS = {
    '的', '了', '是', '我', '你', '他', '她', '它', '们', '在', '有', '和',
    '与', '或', '但', '而', '等', '个', '把', '被', '让', '给', '从', '去',
    '上', '下', '不', '也', '都', '就', '还', '要', '可以', '能', '会',
    '来', '说', '看', '听', '想', '做', '吃', '喝', '玩', '乐', '走', '跑',
    '跳', '坐', '站', '躺', '开', '关', '大', '小', '多', '少', '很', '太',
    '更', '最', '这', '那', '将', '使', '让', '从', '到', '在', '又',
    '还', '可', '以', '之', '于', '及', '因为', '所以', '但是', '而且',
    '然后', '如果', '虽然', '因此', '于是', '那么', '这样', '那样',
    '什么', '为什么', '怎么', '如何', '难道', '吧', '呢', '吗', '啊',
    '呀', '哦', '嗯'
}


# ============================================================
# === 队员B：词频统计模块 ===
# ============================================================

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


# ============================================================
# === 队员B 模块结束 ===
# ============================================================


# ============================================================
# === 队员C：关键词提取模块 ===
# ============================================================

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
            tokens = jieba.lcut(sentence)
            result = []
            for t in tokens:
                t = t.strip()
                if len(t) < 2:
                    continue
                if t in STOPWORDS:
                    continue
                if re.fullmatch(r'[\d\s]+', t):
                    continue
                if re.fullmatch(r'[^\u4e00-\u9fa5a-zA-Z0-9]+', t):
                    continue
                result.append(t)
            return result

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
                if tfidf > max_tfidf[word]:
                    max_tfidf[word] = tfidf

        sorted_words = sorted(max_tfidf.items(), key=lambda x: x[1], reverse=True)
        return {word: round(score, 3) for word, score in sorted_words[:10]}

    except Exception:
        return {}


# ============================================================
# === 队员C 模块结束 ===
# ============================================================


# ============================================================
# === 队员D：情感分析模块（升级版：文学多标签情感模型） ===
# ============================================================

# 定义积极和消极的情感标签（基于 GoEmotions 中文版）
POSITIVE_LABELS = {
    'admiration', 'amusement', 'approval', 'caring', 'excitement',
    'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'
}
NEGATIVE_LABELS = {
    'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
}

# 全局加载模型（仅启动时加载一次）
print("⏳ 正在加载文学情感分析模型（首次约 1.5GB，请耐心等待）...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="liudev/roberta-multilabel-28-3-classes",
    device=-1,          # 使用 CPU
    top_k=None          # 返回所有标签的概率
)
print("✅ 模型加载完成！")


def get_sentiment(text):
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {
            "score": 0.0,
            "label": "中性",
            "pos": 0,
            "neg": 0,
            "score_int": 0,
            "comment": "文本为空，无法分析"
        }

    try:
        # 模型输入长度限制（约 512 token）
        truncated = text[:400] if len(text) > 400 else text

        # 模型返回 list of dicts，每个 dict 包含 label 和 score
        results = sentiment_pipeline(truncated)[0]  # 取第一个样本的结果
        # results 是 list of dict，例如 [{'label': 'anger', 'score': 0.1}, ...]

        positive_sum = 0.0
        negative_sum = 0.0

        for item in results:
            label = item['label']
            score = item['score']
            if label in POSITIVE_LABELS:
                positive_sum += score
            elif label in NEGATIVE_LABELS:
                negative_sum += score
            # 忽略中性标签（如 neutral, confusion 等）

        # 计算最终得分
        total = positive_sum + negative_sum
        if total < 1e-8:
            raw_score = 0.0
        else:
            raw_score = (positive_sum - negative_sum) / (total + 1e-8)

        # 映射到 -100 ~ 100
        score_int = int(round(raw_score * 100))
        score_int = max(-100, min(100, score_int))

        # 确定标签
        if score_int >= 20:
            label_map = "积极"
        elif score_int <= -20:
            label_map = "消极"
        else:
            label_map = "中性"

        # 生成点评（复用原有逻辑）
        if score_int >= 60:
            comment = "整体基调昂扬向上，充满积极与肯定的情绪。"
        elif score_int >= 20:
            comment = "整体情绪偏向积极，表达出正面、向上的态度。"
        elif score_int > -20:
            comment = "整体较为理性、客观，未表现出明显的情感倾向。"
        elif score_int > -60:
            comment = "整体情绪偏于低沉，流露出一定程度的消极或忧虑。"
        else:
            comment = "整体基调较为沉重，带有明显的消极与负面情绪。"

        # 返回时 pos 和 neg 给出概率和（原前端显示的是词数，这里保留，数值仅作示意）
        return {
            "score": round(raw_score, 3),
            "score_int": score_int,
            "label": label_map,
            "pos": round(positive_sum, 3),   # 保留为浮点数，前端会显示
            "neg": round(negative_sum, 3),
            "comment": comment
        }

    except Exception as e:
        print(f"情感分析出错: {e}")
        return {
            "score": 0.0,
            "label": "中性",
            "pos": 0,
            "neg": 0,
            "score_int": 0,
            "comment": "情感分析服务暂时不可用，请稍后重试"
        }


# ============================================================
# === 队员D 模块结束 ===
# ============================================================


# ============================================================
# === 队员A：Flask服务端模块 ===
# ============================================================

def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

    home_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LexiScope</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px;
            background: rgba(255,255,255,0.95);
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .header h1 { font-size: 3.5rem; color: #4a5568; margin-bottom: 15px; font-weight: 700; }
        .header p { color: #718096; font-size: 1.3rem; max-width: 800px; margin: 0 auto; line-height: 1.6; }
        .upload-section {
            background: white;
            padding: 40px;
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 30px;
        }
        .input-methods {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .method-btn {
            padding: 16px 32px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            max-width: 350px;
        }
        .method-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }
        .method-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .file-input-container, .text-input-container {
            display: none;
            flex-direction: column;
            align-items: center;
            margin-bottom: 20px;
        }
        .file-input-container.active, .text-input-container.active { display: flex; }
        .file-input-wrapper { display: inline-block; position: relative; margin-bottom: 15px; }
        .file-input { display: none; }
        .file-label {
            display: inline-block;
            padding: 16px 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            font-size: 1.2rem;
            width: 100%;
            max-width: 350px;
            text-align: center;
        }
        .file-label:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102,126,234,0.4); }
        .file-name { margin-top: 15px; color: #718096; font-size: 1.2rem; }
        .text-input {
            width: 100%;
            max-width: 350px;
            min-height: 56px;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1.1rem;
            resize: vertical;
            font-family: inherit;
            transition: border-color 0.3s ease;
        }
        .text-input:focus { outline: none; border-color: #667eea; }
        .analyze-btn {
            padding: 16px 48px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.3rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }
        .analyze-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(245,87,108,0.4); }
        .analyze-btn:disabled { background: #cbd5e0; cursor: not-allowed; transform: none; box-shadow: none; }
        .error { color: #e53e3e; margin-top: 30px; font-size: 1.1rem; display: none; }
        .result-section { display: none; animation: fadeIn 0.5s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .result-header { text-align: center; margin-bottom: 30px; }
        .result-header h2 { font-size: 2rem; color: white; margin-bottom: 10px; }
        .result-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; }
        .result-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 30px;
        }
        .card {
            background: white;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            height: 100%;
        }
        .card:hover { transform: translateY(-5px); }
        .card-header { display: flex; align-items: center; margin-bottom: 25px; }
        .card-icon { font-size: 2.2rem; margin-right: 15px; }
        .card-title { font-size: 1.5rem; font-weight: 600; color: #4a5568; }
        .card-content { min-height: 220px; }
        .result-list { list-style: none; }
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 14px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        .result-item:last-child { border-bottom: none; }
        .result-word { font-weight: 500; color: #2d3748; font-size: 1.2rem; }
        .result-value { color: #718096; font-weight: 600; font-size: 1.2rem; }
        .sentiment-result { display: flex; flex-direction: column; gap: 12px; }
        .sentiment-score {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            padding: 12px;
            border-radius: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .sentiment-label {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 600;
            padding: 12px;
            border-radius: 15px;
            background: #f7fafc;
        }
        .sentiment-positive { color: #38a169; background: #f0fff4; }
        .sentiment-negative { color: #e53e3e; background: #fff5f5; }
        .sentiment-neutral { color: #718096; background: #f7fafc; }
        .sentiment-stats {
            display: flex;
            justify-content: space-around;
            text-align: center;
            padding: 16px;
            background: #f7fafc;
            border-radius: 18px;
        }
        .stat-item { display: flex; flex-direction: column; }
        .stat-value { font-size: 1.6rem; font-weight: 700; color: #4a5568; }
        .stat-label { font-size: 1.1rem; color: #718096; }
        .sentiment-bar-container { margin-top: 8px; padding-top: 12px; border-top: 1px solid #e2e8f0; }
        .sentiment-bar-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: #718096; }
        .sentiment-bar {
            position: relative;
            height: 12px;
            background: linear-gradient(to right, #3b82f6, #ef4444);
            border-radius: 6px;
            margin: 4px 0;
        }
        .sentiment-bar-marker {
            position: absolute;
            top: -4px;
            width: 3px;
            height: 20px;
            background: #1a202c;
            border-radius: 2px;
            transform: translateX(-50%);
            transition: left 0.3s ease;
        }
        .sentiment-bar-score { text-align: center; font-size: 0.9rem; font-weight: 600; color: #2d3748; }
        .sentiment-comment {
            margin-top: 8px;
            padding: 10px 14px;
            background: #f7fafc;
            border-radius: 12px;
            font-size: 0.95rem;
            color: #4a5568;
            line-height: 1.6;
            text-align: center;
            border-left: 3px solid #667eea;
        }
        @media (max-width: 768px) {
            .header h1 { font-size: 2.8rem; }
            .header p { font-size: 1.1rem; }
            .input-methods { flex-direction: column; gap: 10px; }
            .result-cards { grid-template-columns: 1fr; }
            .card { padding: 30px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>LexiScope</h1>
            <p>智能文本洞察者 - 深度分析文本内容，提取关键信息，洞察情感倾向</p>
        </div>
        <div class="upload-section">
            <div class="input-methods">
                <button class="method-btn active" data-method="text">直接输入</button>
                <button class="method-btn" data-method="file">上传文档</button>
            </div>
            <div class="file-input-container">
                <div class="file-input-wrapper">
                    <input type="file" id="fileInput" class="file-input" accept=".txt,.docx">
                    <label for="fileInput" class="file-label">选择文件</label>
                </div>
                <span class="file-name" id="fileName">未选择文件</span>
            </div>
            <div class="text-input-container active">
                <textarea id="textInput" class="text-input" placeholder="请在此输入或粘贴文本内容..."></textarea>
            </div>
            <div id="errorMsg" class="error"></div>
        </div>
        <div style="text-align: center; margin-bottom: 30px;">
            <button id="analyzeBtn" class="analyze-btn">开始分析</button>
        </div>
        <div id="resultSection" class="result-section">
            <div class="result-header">
                <h2>分析结果</h2>
                <p>以下是您的文本深度分析报告</p>
            </div>
            <div class="result-cards">
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">📊</span>
                        <h2 class="card-title">词频统计</h2>
                    </div>
                    <div class="card-content"><ul id="wordFreqList" class="result-list"></ul></div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">🏷️</span>
                        <h2 class="card-title">关键词提取</h2>
                    </div>
                    <div class="card-content"><ul id="keywordsList" class="result-list"></ul></div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">❤️</span>
                        <h2 class="card-title">情感分析</h2>
                    </div>
                    <div class="card-content"><div id="sentimentResult" class="sentiment-result"></div></div>
                </div>
            </div>
        </div>
    </div>
    <script>
        var methodBtns = document.querySelectorAll('.method-btn');
        var fileInputContainer = document.querySelector('.file-input-container');
        var textInputContainer = document.querySelector('.text-input-container');
        var fileInput = document.getElementById('fileInput');
        var textInput = document.getElementById('textInput');
        var fileName = document.getElementById('fileName');
        var errorMsg = document.getElementById('errorMsg');
        var analyzeBtn = document.getElementById('analyzeBtn');
        var resultSection = document.getElementById('resultSection');

        for (var i = 0; i < methodBtns.length; i++) {
            methodBtns[i].addEventListener('click', function() {
                for (var j = 0; j < methodBtns.length; j++) {
                    methodBtns[j].classList.remove('active');
                }
                this.classList.add('active');
                if (this.dataset.method === 'file') {
                    fileInputContainer.classList.add('active');
                    textInputContainer.classList.remove('active');
                } else {
                    fileInputContainer.classList.remove('active');
                    textInputContainer.classList.add('active');
                }
            });
        }

        fileInput.addEventListener('change', function(e) {
            var selectedFile = e.target.files[0];
            fileName.textContent = selectedFile ? selectedFile.name : '未选择文件';
        });

        function renderWordFreq(wordFreq) {
            var list = document.getElementById('wordFreqList');
            list.innerHTML = '';
            if (!wordFreq || Object.keys(wordFreq).length === 0) {
                list.innerHTML = '<li class="result-item">暂无数据</li>';
                return;
            }
            var keys = Object.keys(wordFreq);
            for (var i = 0; i < keys.length; i++) {
                var word = keys[i];
                var count = wordFreq[word];
                var li = document.createElement('li');
                li.className = 'result-item';
                li.innerHTML = '<span class="result-word">' + word + '</span><span class="result-value">' + count + '</span>';
                list.appendChild(li);
            }
        }

        function renderKeywords(keywords) {
            var list = document.getElementById('keywordsList');
            list.innerHTML = '';
            if (!keywords || Object.keys(keywords).length === 0) {
                list.innerHTML = '<li class="result-item">暂无数据</li>';
                return;
            }
            var keys = Object.keys(keywords);
            for (var i = 0; i < keys.length; i++) {
                var word = keys[i];
                var weight = keywords[word];
                var displayWeight = typeof weight === 'number' ? weight.toFixed(3) : weight;
                var li = document.createElement('li');
                li.className = 'result-item';
                li.innerHTML = '<span class="result-word">' + word + '</span><span class="result-value">' + displayWeight + '</span>';
                list.appendChild(li);
            }
        }

        function renderSentiment(sentiment) {
            var result = document.getElementById('sentimentResult');
            result.innerHTML = '';
            if (!sentiment || sentiment.score_int === undefined) {
                result.innerHTML = '<div class="sentiment-score">暂无数据</div>';
                return;
            }
            var scoreInt = sentiment.score_int;
            var label = sentiment.label || '中性';
            var pos = sentiment.pos || 0;
            var neg = sentiment.neg || 0;
            var comment = sentiment.comment || '';

            var labelClass = 'sentiment-neutral';
            if (label === '积极') labelClass = 'sentiment-positive';
            if (label === '消极') labelClass = 'sentiment-negative';

            var leftPercent = ((scoreInt + 100) / 200) * 100;
            if (leftPercent < 0) leftPercent = 0;
            if (leftPercent > 100) leftPercent = 100;

            result.innerHTML = 
                '<div class="sentiment-score">得分: ' + scoreInt + '</div>' +
                '<div class="sentiment-label ' + labelClass + '">' + label + '</div>' +
                '<div class="sentiment-stats">' +
                    '<div class="stat-item"><span class="stat-value">' + pos.toFixed(2) + '</span><span class="stat-label">积极概率和</span></div>' +
                    '<div class="stat-item"><span class="stat-value">' + neg.toFixed(2) + '</span><span class="stat-label">消极概率和</span></div>' +
                '</div>' +
                '<div class="sentiment-bar-container">' +
                    '<div class="sentiment-bar-labels"><span>-100</span><span>0</span><span>+100</span></div>' +
                    '<div class="sentiment-bar">' +
                        '<div class="sentiment-bar-marker" style="left:' + leftPercent.toFixed(1) + '%;"></div>' +
                    '</div>' +
                    '<div class="sentiment-bar-score">当前得分: ' + scoreInt + ' 分</div>' +
                '</div>' +
                (comment ? '<div class="sentiment-comment">📝 ' + comment + '</div>' : '');
        }

        function performAnalysis(text) {
            errorMsg.style.display = 'none';
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '分析中...';

            fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            })
            .then(function(res) { return res.json(); })
            .then(function(data) {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = '开始分析';
                if (data.error) {
                    errorMsg.textContent = data.error;
                    errorMsg.style.display = 'block';
                } else {
                    renderWordFreq(data.word_freq);
                    renderKeywords(data.keywords);
                    renderSentiment(data.sentiment);
                    resultSection.style.display = 'block';
                    resultSection.scrollIntoView({ behavior: 'smooth' });
                }
            })
            .catch(function(err) {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = '开始分析';
                errorMsg.textContent = '请求失败，请检查网络或服务器';
                errorMsg.style.display = 'block';
            });
        }

        analyzeBtn.addEventListener('click', function() {
            var text = '';
            if (fileInputContainer.classList.contains('active')) {
                if (fileName.textContent === '未选择文件' || !fileInput.files[0]) {
                    errorMsg.textContent = '请上传文件';
                    errorMsg.style.display = 'block';
                    return;
                }
                var reader = new FileReader();
                reader.onload = function(e) {
                    performAnalysis(e.target.result);
                };
                reader.readAsText(fileInput.files[0], 'UTF-8');
            } else {
                text = textInput.value.trim();
                if (!text) {
                    errorMsg.textContent = '请输入文本内容';
                    errorMsg.style.display = 'block';
                    return;
                }
                performAnalysis(text);
            }
        });
    </script>
</body>
</html>
    """

    @app.route('/')
    def index():
        return home_html

    @app.route('/analyze', methods=['POST'])
    def analyze():
        text = None

        if request.is_json:
            data = request.get_json()
            text = data.get('text', '') if data else ''
        else:
            if 'file' not in request.files:
                return jsonify({"error": "请上传文件"}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "请选择一个文件"}), 400

            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()

            try:
                if ext == '.txt':
                    text = file.read().decode('utf-8')
                elif ext == '.docx':
                    file_bytes = BytesIO(file.read())
                    with zipfile.ZipFile(file_bytes, 'r') as zf:
                        xml_content = zf.read('word/document.xml').decode('utf-8')
                        text_parts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', xml_content)
                        text = ''.join(text_parts)
                    if not text or not text.strip():
                        return jsonify({"error": "文档中未读取到文字内容"}), 400
                else:
                    return jsonify({"error": "仅支持 .txt 和 .docx 格式的文件"}), 400
            except Exception as e:
                return jsonify({"error": f"文件读取失败：{str(e)}"}), 400

        if not text or not text.strip():
            return jsonify({"error": "文本内容为空"}), 400

        try:
            word_freq_result = get_word_freq(text)
            keywords_result = extract_keywords(text)
            sentiment_result = get_sentiment(text)
        except Exception as e:
            return jsonify({"error": f"分析过程出错：{str(e)}"}), 500

        return jsonify({
            "word_freq": word_freq_result,
            "keywords": keywords_result,
            "sentiment": sentiment_result
        })

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=5000)