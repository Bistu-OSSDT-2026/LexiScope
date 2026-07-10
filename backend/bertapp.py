def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

    # 注意：home_html 前端变量由队员E提供并定义
    @app.route('/')
    def index():
        return home_html

    @app.route('/analyze', methods=['POST'])
    def analyze():
        text = None

        # 1. 解析请求（支持 JSON 和 文件上传）
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

        # 2. 调用各模块（B, C, D）
        try:
            word_freq_result = get_word_freq(text)      # 队员B
            keywords_result = extract_keywords(text)    # 队员C
            sentiment_result = get_sentiment(text)      # 队员D（新模型）
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
