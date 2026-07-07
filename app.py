import re
import math
import collections
from flask import Flask, request, jsonify

# ============================================================
# 【占位函数】等B/C/D同学写好后替换
# 现在这些是假数据，用来测试你的服务器能不能跑通
# ============================================================

def get_word_freq(text):
    """B同学写好后替换我"""
    return {"测试词1": 5, "测试词2": 3, "测试词3": 2}

def extract_keywords(text):
    """C同学写好后替换我"""
    return {"测试关键词1": 0.856, "测试关键词2": 0.721, "测试关键词3": 0.534}

def get_sentiment(text):
    """D同学写好后替换我"""
    return {"score": 0.350, "label": "积极", "pos": 12, "neg": 5}

# ============================================================
# 【A同学的核心代码】Flask服务器 + 路由
# 这部分你不需要修改，除非报错
# ============================================================

def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    
    # ---------- 首页HTML（E同学最终会替换这个）----------
    home_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>文脉 · 智能文本洞察者</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f7fa; }
            .card { background: white; border-radius: 16px; padding: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 24px; }
            h1 { margin: 0; font-size: 28px; }
            .upload-area { border: 2px dashed #ccc; border-radius: 12px; padding: 30px; text-align: center; }
            .upload-area input[type="file"] { margin: 12px 0; }
            button { background: #2563eb; color: white; border: none; padding: 12px 40px; border-radius: 8px; font-size: 16px; cursor: pointer; }
            button:disabled { background: #94a3b8; cursor: not-allowed; }
            .error { background: #fee2e2; color: #dc2626; padding: 12px 20px; border-radius: 8px; display: none; margin-bottom: 16px; }
            .results { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 16px; }
            .result-card { background: white; border-radius: 12px; padding: 20px; border: 1px solid #e5e7eb; }
            .result-card h3 { margin-top: 0; font-size: 16px; color: #4b5563; }
            .result-card ul { padding-left: 0; list-style: none; max-height: 300px; overflow-y: auto; }
            .result-card li { padding: 4px 0; border-bottom: 1px solid #f3f4f6; font-size: 14px; }
            .sentiment-score { font-size: 32px; font-weight: bold; }
            .sentiment-label { display: inline-block; padding: 4px 16px; border-radius: 20px; font-size: 14px; }
            .label-positive { background: #d1fae5; color: #065f46; }
            .label-negative { background: #fee2e2; color: #991b1b; }
            .label-neutral { background: #e5e7eb; color: #4b5563; }
            @media (max-width: 700px) { .results { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>📄 文脉 · 智能文本洞察者</h1>
            <p style="color: #6b7280;">上传TXT文件，自动分析词频、关键词和情感倾向</p>
        </div>

        <div class="card">
            <div class="upload-area">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".txt" id="fileInput">
                    <br><br>
                    <button type="submit" id="analyzeBtn">开始分析</button>
                </form>
            </div>
        </div>

        <div id="errorBox" class="error"></div>

        <div id="resultArea" style="display: none;">
            <div class="results">
                <div class="result-card" id="freqCard"><h3>📊 词频统计 (Top20)</h3><ul id="freqList"></ul></div>
                <div class="result-card" id="keywordCard"><h3>🏷️ 关键词提取 (Top10)</h3><ul id="keywordList"></ul></div>
                <div class="result-card" id="sentimentCard"><h3>❤️ 情感分析</h3><div id="sentimentContent"></div></div>
            </div>
        </div>

        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();
                const btn = document.getElementById('analyzeBtn');
                const errorBox = document.getElementById('errorBox');
                const resultArea = document.getElementById('resultArea');
                
                errorBox.style.display = 'none';
                errorBox.textContent = '';
                resultArea.style.display = 'none';
                btn.disabled = true;
                btn.textContent = '⏳ 分析中...';

                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files || fileInput.files.length === 0) {
                    errorBox.textContent = '请选择一个文件';
                    errorBox.style.display = 'block';
                    btn.disabled = false;
                    btn.textContent = '开始分析';
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const resp = await fetch('/analyze', { method: 'POST', body: formData });
                    const data = await resp.json();

                    if (data.error) {
                        errorBox.textContent = '❌ ' + data.error;
                        errorBox.style.display = 'block';
                        btn.disabled = false;
                        btn.textContent = '开始分析';
                        return;
                    }

                    // 渲染词频
                    const freqList = document.getElementById('freqList');
                    freqList.innerHTML = '';
                    const freqData = data.word_freq || {};
                    const freqEntries = Object.entries(freqData).slice(0, 20);
                    if (freqEntries.length === 0) {
                        freqList.innerHTML = '<li style="color:#9ca3af;">暂无数据</li>';
                    } else {
                        freqEntries.forEach(([word, count]) => {
                            const li = document.createElement('li');
                            li.textContent = word + '：' + count + ' 次';
                            freqList.appendChild(li);
                        });
                    }

                    // 渲染关键词
                    const keywordList = document.getElementById('keywordList');
                    keywordList.innerHTML = '';
                    const keywordData = data.keywords || {};
                    const keywordEntries = Object.entries(keywordData).slice(0, 10);
                    if (keywordEntries.length === 0) {
                        keywordList.innerHTML = '<li style="color:#9ca3af;">暂无数据</li>';
                    } else {
                        keywordEntries.forEach(([word, weight]) => {
                            const li = document.createElement('li');
                            li.textContent = word + '：' + weight.toFixed(3);
                            keywordList.appendChild(li);
                        });
                    }

                    // 渲染情感
                    const sentiment = data.sentiment || { score: 0, label: '中性', pos: 0, neg: 0 };
                    const labelClass = sentiment.label === '积极' ? 'label-positive' : 
                                       sentiment.label === '消极' ? 'label-negative' : 'label-neutral';
                    document.getElementById('sentimentContent').innerHTML = `
                        <div class="sentiment-score">${sentiment.score.toFixed(3)}</div>
                        <span class="sentiment-label ${labelClass}">${sentiment.label}</span>
                        <p style="font-size:14px;color:#6b7280;margin-top:12px;">
                            👍 正面词 ${sentiment.pos} 个 &nbsp;|&nbsp; 👎 负面词 ${sentiment.neg} 个
                        </p>
                    `;

                    resultArea.style.display = 'block';

                } catch (err) {
                    errorBox.textContent = '❌ 网络错误，请重试';
                    errorBox.style.display = 'block';
                }

                btn.disabled = false;
                btn.textContent = '开始分析';
            };
        </script>
    </body>
    </html>
    """
    # ---------- 首页HTML结束 ----------
    
    @app.route('/')
    def index():
        """首页路由：返回HTML页面"""
        return home_html
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        """
        分析路由：接收上传的txt文件，调用三个分析函数，返回JSON结果
        """
        # 1. 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"error": "请上传文件"}), 400
        
        file = request.files['file']
        
        # 2. 检查文件名是否为空
        if file.filename == '':
            return jsonify({"error": "请选择一个文件"}), 400
        
        # 3. 读取文件内容（UTF-8解码）
        try:
            text = file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({"error": "文件编码不支持，请保存为UTF-8格式"}), 400
        
        # 4. 检查文件内容是否为空
        if not text or text.strip() == '':
            return jsonify({"error": "文件内容为空"}), 400
        
        # 5. 调用三个分析函数
        word_freq_result = get_word_freq(text)
        keywords_result = extract_keywords(text)
        sentiment_result = get_sentiment(text)
        
        # 6. 打包成规定的JSON格式返回
        result = {
            "word_freq": word_freq_result,
            "keywords": keywords_result,
            "sentiment": sentiment_result
        }
        
        return jsonify(result)
    
    return app

# ============================================================
# 启动代码（运行这个文件就会启动服务器）
# ============================================================

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)