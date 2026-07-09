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

        # 模型推理
        results = sentiment_pipeline(truncated)[0]

        positive_sum = 0.0
        negative_sum = 0.0

        for item in results:
            label = item['label']
            score = item['score']
            if label in POSITIVE_LABELS:
                positive_sum += score
            elif label in NEGATIVE_LABELS:
                negative_sum += score

        total = positive_sum + negative_sum
        if total < 1e-8:
            raw_score = 0.0
        else:
            raw_score = (positive_sum - negative_sum) / (total + 1e-8)

        score_int = int(round(raw_score * 100))
        score_int = max(-100, min(100, score_int))

        if score_int >= 20:
            label_map = "积极"
        elif score_int <= -20:
            label_map = "消极"
        else:
            label_map = "中性"

        # 复用原有的分档点评逻辑
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

        return {
            "score": round(raw_score, 3),
            "score_int": score_int,
            "label": label_map,
            "pos": round(positive_sum, 3),   # 返回概率和（浮点数）
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