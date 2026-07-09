#============================================================
# 导入语句（所有依赖统一放在顶部）
# ============================================================

import re
import math
import collections
from flask import Flask, request, jsonify
import jieba


# ============================================================
# === 队员B：词频统计模块（函数：get_word_freq）===
# ============================================================

def get_word_freq(text):
    if not text or not isinstance(text, str):
        return {}

    stopwords = {
        '的', '了', '是', '我', '你', '他', '她', '它', '们',
        '在', '有', '和', '与', '或', '但', '而', '等', '个', '把', '被', '让',
        '给', '从', '去', '上', '下', '不', '也', '都', '就', '还', '要',
        '可以', '能', '会', '来', '说', '看', '听', '想', '做', '吃', '喝', '玩', '乐',
        '走', '跑', '跳', '坐', '站', '躺', '开', '关', '大', '小', '多', '少',
        '很', '太', '更', '最'
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


# ============================================================
# === 队员B 模块结束 ===
# ============================================================


# ============================================================
# === 队员C：关键词提取模块（函数：extract_keywords）===
# ============================================================

def extract_keywords(text):
    try:
        if not text or not text.strip():
            return {}

        text = text.strip()

        # 按句子拆分
        sentences_raw = re.split(r'[。！？；.!?;]+', text)
        sentences = [s.strip() for s in sentences_raw if s.strip()]

        if not sentences:
            return {}

        # 简单切词函数
        def tokenize(sentence):
            tokens_raw = re.split(
                r'[\s\u3000\uff0c,，、：:""''《》【】（）()---/\\@#$%\^&*+=\|~`-]+',
                sentence
            )
            tokens = []
            for t in tokens_raw:
                t = t.strip()
                if len(t) < 2:
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

        # 计算IDF
        total_sentences = len(sentence_tokens)
        idf_map = {}
        for word in all_words:
            doc_count = sum(1 for tokens in sentence_tokens if word in tokens)
            idf_map[word] = math.log(total_sentences / (doc_count + 1)) + 1

        # 计算TF-IDF（每个词取所有句子中的最大值）
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

        # 排序取前10
        sorted_words = sorted(max_tfidf.items(), key=lambda x: x[1], reverse=True)
        return {word: round(score, 3) for word, score in sorted_words[:10]}

    except Exception:
        # 任何异常均返回空字典
        return {}


# ============================================================
# === 队员C 模块结束 ===
# ============================================================


# ============================================================
# === 队员D：情感分析模块（函数：get_sentiment）===
# ============================================================

def get_sentiment(text):
    # 1. 空文本处理
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {"score": 0.0, "label": "中性", "pos": 0, "neg": 0}

    # 2. 情感词库（硬编码）
    positive_words = {
        "好", "棒", "优秀", "出色", "满意", "喜欢", "爱",
        "开心", "快乐", "幸福", "美好", "精彩", "卓越", "完美", "靠谱", "真诚",
        "善良", "温暖", "阳光", "积极", "上进", "努力", "奋斗", "成功",
        "希望", "光明", "自由", "和平", "团结", "友好", "赞", "棒极了",
        "佩服", "给力", "厉害", "松弛感", "水灵灵", "双向奔赴", "多巴胺"
    }
    negative_words = {
        "差", "坏", "糟糕", "失望", "讨厌", "恨", "伤心",
        "痛苦", "难过", "悲惨", "恶心", "虚伪", "冷漠", "消极", "懒惰", "失败",
        "绝望", "黑暗", "压抑", "焦虑", "紧张", "害怕", "恐惧", "愤怒",
        "仇恨", "抱怨", "麻烦", "困难", "危险", "浪费", "烂", "糟透了",
        "悲哀", "可恶", "无奈", "班味", "内耗", "蚌埠住了"
    }

    # 3. 按标点切分为短句片段
    fragments = re.split(r'[，。！？、；：\s]+', text)

    # 4. 统计每个片段中是否包含情感词
    pos_count = 0
    neg_count = 0

    for frag in fragments:
        # 检查片段中是否包含正面词
        for word in positive_words:
            if word in frag:
                pos_count += 1
                break  # 每个片段只计一次正面匹配
        # 检查片段中是否包含负面词
        for word in negative_words:
            if word in frag:
                neg_count += 1
                break  # 每个片段只计一次负面匹配

    # 5. 计算得分
    score = (pos_count - neg_count) / (pos_count + neg_count + 1)
    score = round(score, 3)

    # 6. 判定标签
    if score > 0.1:
        label = "积极"
    elif score < -0.1:
        label = "消极"
    else:
        label = "中性"

    # 7. 返回结果
    return {
        "score": score,
        "label": label,
        "pos": pos_count,
        "neg": neg_count
    }


# ============================================================
# === 队员D 模块结束 ===
# ============================================================


# ============================================================
# === 队员A：Flask服务端模块 ===
# ============================================================

def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 限制5MB

    # ============================================================
    # 首页HTML（来自E同学的完整页面）
    # ============================================================

    home_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LexiScope</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        .header h1 {
            font-size: 3.5rem;
            color: #4a5568;
            margin-bottom: 15px;
            font-weight: 700;
        }
        .header p {
            color: #718096;
            font-size: 1.3rem;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .upload-section {
            background: white;
            padding: 40px;
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 30px;
            position: relative;
        }
        .input-methods {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
            width: 100%;
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
        .method-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .file-input-container, .text-input-container {
            display: none;
            flex-direction: column;
            align-items: center;
            margin-bottom: 20px;
        }
        .file-input-container.active, .text-input-container.active {
            display: flex;
        }
        .file-input-wrapper {
            display: inline-block;
            position: relative;
            margin-bottom: 15px;
        }
        .file-input {
            display: none;
        }
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
        .file-label:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        .file-name {
            margin-top: 15px;
            color: #718096;
            font-size: 1.2rem;
        }
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
        .text-input:focus {
            outline: none;
            border-color: #667eea;
        }
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
        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(245, 87, 108, 0.4);
        }
        .analyze-btn:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .error {
            color: #e53e3e;
            margin-top: 30px;
            font-size: 1.1rem;
            display: none;
        }
        .result-section {
            display: none;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .result-header h2 {
            font-size: 2rem;
            color: white;
            margin-bottom: 10px;
        }
        .result-header p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
        }
        .result-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 30px;
        }
        .card {
            background: white;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            height: 100%;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 25px;
        }
        .card-icon {
            font-size: 2.2rem;
            margin-right: 15px;
        }
        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #4a5568;
        }
        .card-content {
            min-height: 220px;
        }
        .result-list {
            list-style: none;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 14px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        .result-item:last-child {
            border-bottom: none;
        }
        .result-word {
            font-weight: 500;
            color: #2d3748;
            font-size: 1.2rem;
        }
        .result-value {
            color: #718096;
            font-weight: 600;
            font-size: 1.2rem;
        }
        .sentiment-result {
            display: flex;
            flex-direction: column;
            gap: 25px;
        }
        .sentiment-score {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            padding: 25px;
            border-radius: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .sentiment-label {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 600;
            padding: 18px;
            border-radius: 15px;
            background: #f7fafc;
        }
        .sentiment-positive {
            color: #38a169;
            background: #f0fff4;
        }
        .sentiment-negative {
            color: #e53e3e;
            background: #fff5f5;
        }
        .sentiment-neutral {
            color: #718096;
            background: #f7fafc;
        }
        .sentiment-stats {
            display: flex;
            justify-content: space-around;
            text-align: center;
            padding: 25px;
            background: #f7fafc;
            border-radius: 18px;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #4a5568;
        }
        .stat-label {
            font-size: 1.1rem;
            color: #718096;
        }
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2.8rem;
            }
            .header p {
                font-size: 1.1rem;
            }
            .input-methods {
                flex-direction: column;
                gap: 10px;
            }
            .result-cards {
                grid-template-columns: 1fr;
            }
            .card {
                padding: 30px;
            }
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
                    <input type="file" id="fileInput" class="file-input" accept=".txt,.doc,.docx">
                    <label for="fileInput" class="file-label">选择文件</label>
                </div>
                <span class="file-name" id="fileName">未选择文件</span>
            </div>
            <div class="text-input-container">
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
                    <div class="card-content">
                        <ul id="wordFreqList" class="result-list"></ul>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">🏷️</span>
                        <h2 class="card-title">关键词提取</h2>
                    </div>
                    <div class="card-content">
                        <ul id="keywordsList" class="result-list"></ul>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">❤️</span>
                        <h2 class="card-title">情感分析</h2>
                    </div>
                    <div class="card-content">
                        <div id="sentimentResult" class="sentiment-result"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        // 切换输入方式
        const methodBtns = document.querySelectorAll('.method-btn');
        const fileInputContainer = document.querySelector('.file-input-container');
        const textInputContainer = document.querySelector('.text-input-container');
        const fileInput = document.getElementById('fileInput');
        const textInput = document.getElementById('textInput');
        const fileName = document.getElementById('fileName');
        const uploadSection = document.querySelector('.upload-section');

        methodBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                methodBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const method = btn.dataset.method;
                if (method === 'file') {
                    fileInputContainer.classList.add('active');
                    textInputContainer.classList.remove('active');
                } else {
                    fileInputContainer.classList.remove('active');
                    textInputContainer.classList.add('active');
                }
            });
        });

        // 文件选择处理
        fileInput.addEventListener('change', function(e) {
            const selectedFile = e.target.files[0];
            if (selectedFile) {
                fileName.textContent = selectedFile.name;
            } else {
                fileName.textContent = '未选择文件';
            }
        });

        // 点击空白区域回到初始状态
        uploadSection.addEventListener('click', function(e) {
            const isButton = e.target.classList.contains('method-btn');
            const isInputArea = e.target.classList.contains('text-input') ||
                e.target.classList.contains('file-label') ||
                e.target.classList.contains('file-input');
            if (!isButton && !isInputArea) {
                methodBtns.forEach(b => b.classList.remove('active'));
                fileInputContainer.classList.remove('active');
                textInputContainer.classList.remove('active');
            }
        });

        // 分析按钮点击事件
        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            const btn = document.getElementById('analyzeBtn');
            const errorMsg = document.getElementById('errorMsg');
            const resultSection = document.getElementById('resultSection');
            let text = '';

            // 获取文本内容
            if (fileInputContainer.classList.contains('active')) {
                // 文件上传方式
                if (fileName.textContent === '未选择文件') {
                    errorMsg.textContent = '请上传文件';
                    errorMsg.style.display = 'block';
                    return;
                }
                const file = fileInput.files[0];
                if (!file) {
                    errorMsg.textContent = '请选择文件';
                    errorMsg.style.display = 'block';
                    return;
                }
                try {
                    const reader = new FileReader();
                    reader.onload = async (e) => {
                        text = e.target.result;
                        await performAnalysis(text);
                    };
                    reader.readAsText(file, 'UTF-8');
                } catch (err) {
                    errorMsg.textContent = '文件读取失败，请检查文件格式';
                    errorMsg.style.display = 'block';
                }
            } else {
                // 文本输入方式
                text = textInput.value.trim();
                if (!text) {
                    errorMsg.textContent = '请输入文本内容';
                    errorMsg.style.display = 'block';
                    return;
                }
                await performAnalysis(text);
            }

            btn.disabled = true;
            btn.textContent = '分析中...';
            errorMsg.style.display = 'none';
        });

        // 执行分析
        async function performAnalysis(text) {
            const btn = document.getElementById('analyzeBtn');
            const errorMsg = document.getElementById('errorMsg');
            const resultSection = document.getElementById('resultSection');

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

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
            } catch (err) {
                errorMsg.textContent = '请求失败，请检查网络或服务器';
                errorMsg.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = '开始分析';
            }
        }

        function renderWordFreq(wordFreq) {
            const list = document.getElementById('wordFreqList');
            list.innerHTML = '';
            if (Object.keys(wordFreq).length === 0) {
                list.innerHTML = '<li class="result-item">暂无数据</li>';
                return;
            }
            for (const [word, count] of Object.entries(wordFreq)) {
                const item = document.createElement('li');
                item.className = 'result-item';
                item.innerHTML = `
                    <span class="result-word">${word}</span>
                    <span class="result-value">${count}</span>
                `;
                list.appendChild(item);
            }
        }

        function renderKeywords(keywords) {
            const list = document.getElementById('keywordsList');
            list.innerHTML = '';
            if (Object.keys(keywords).length === 0) {
                list.innerHTML = '<li class="result-item">暂无数据</li>';
                return;
            }
            for (const [word, weight] of Object.entries(keywords)) {
                const item = document.createElement('li');
                item.className = 'result-item';
                item.innerHTML = `
                    <span class="result-word">${word}</span>
                    <span class="result-value">${weight.toFixed(3)}</span>
                `;
                list.appendChild(item);
            }
        }

        function renderSentiment(sentiment) {
            const result = document.getElementById('sentimentResult');
            result.innerHTML = '';
            if (Object.keys(sentiment).length === 0) {
                result.innerHTML = '<div class="sentiment-score">暂无数据</div>';
                return;
            }
            const score = sentiment.score.toFixed(3);
            const label = sentiment.label;
            const pos = sentiment.pos;
            const neg = sentiment.neg;
            let labelClass = 'sentiment-neutral';
            if (label === '积极') labelClass = 'sentiment-positive';
            if (label === '消极') labelClass = 'sentiment-negative';
            result.innerHTML = `
                <div class="sentiment-score">得分: ${score}</div>
                <div class="sentiment-label ${labelClass}">${label}</div>
                <div class="sentiment-stats">
                    <div class="stat-item">
                        <span class="stat-value">${pos}</span>
                        <span class="stat-label">正面词</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">${neg}</span>
                        <span class="stat-label">负面词</span>
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>
    """

    # ============================================================
    # 首页路由
    # ============================================================

    @app.route('/')
    def index():
        return home_html

    # ============================================================
    # 分析路由（同时支持文件上传和JSON传文本两种方式）
    # ============================================================

    @app.route('/analyze', methods=['POST'])
    def analyze():
        text = None

        # 判断请求类型
        if request.is_json:
            # 方式一：JSON传文本（E同学直接输入方式）
            data = request.get_json()
            text = data.get('text', '') if data else ''
        else:
            # 方式二：文件上传（E同学文件上传方式）
            if 'file' not in request.files:
                return jsonify({"error": "请上传文件"}), 400

            file = request.files['file']

            if file.filename == '':
                return jsonify({"error": "请选择一个文件"}), 400

            # 检查文件类型
            if not file.filename.endswith('.txt'):
                return jsonify({"error": "仅支持 .txt 格式的文件"}), 400

            try:
                text = file.read().decode('utf-8')
            except UnicodeDecodeError:
                return jsonify({"error": "文件编码不支持，请保存为UTF-8格式"}), 400

        # 检查文本是否为空
        if not text or not text.strip():
            return jsonify({"error": "文本内容为空"}), 400

        # 调用三个分析函数
        try:
            word_freq_result = get_word_freq(text)
            keywords_result = extract_keywords(text)
            sentiment_result = get_sentiment(text)
        except Exception as e:
            return jsonify({"error": f"分析过程出错：{str(e)}"}), 500

        # 打包返回
        return jsonify({
            "word_freq": word_freq_result,
            "keywords": keywords_result,
            "sentiment": sentiment_result
        })

    return app


# ============================================================
# 启动服务
# ============================================================

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=5000)


# ============================================================
# === 队员A 模块结束 ===
# ============================================================
