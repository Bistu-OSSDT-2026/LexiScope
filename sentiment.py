# ============================================
# === 队员D：情感分析模块（函数：get_sentiment）===
# ============================================

import re

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


# ============================================
# === 队员D 模块结束 ===
# ============================================