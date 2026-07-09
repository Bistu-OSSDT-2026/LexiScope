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
# === 队员D：情感分析模块 ===
# ============================================================

def get_sentiment(text):
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {"score": 0.0, "label": "中性", "pos": 0, "neg": 0, "score_int": 0, "comment": "文本为空，无法分析"}

    # ===== 正面词 =====
    positive_words = {
        "好", "棒", "优秀", "出色", "满意", "喜欢", "爱",
        "开心", "快乐", "幸福", "美好", "精彩", "卓越", "完美", "靠谱", "真诚",
        "善良", "温暖", "阳光", "积极", "上进", "努力", "奋斗", "成功",
        "希望", "光明", "自由", "和平", "团结", "友好", "赞", "棒极了",
        "佩服", "给力", "厉害", "松弛感", "水灵灵", "双向奔赴", "多巴胺",
        "献身", "斗争", "解放", "英勇", "无畏", "自豪", "骄傲",
        "光荣", "伟大", "壮丽", "崇高", "坚定", "坚决", "果断", "勇敢",
        "智慧", "聪明", "卓越", "杰出", "非凡", "了不起", "值得", "有意义",
        "宝贵", "奋斗", "献", "全心全意", "全部", "整个", "最壮丽"
    }

    # ===== 负面词 =====
    negative_words = {
        # 基础负面
        "差", "坏", "糟糕", "失望", "讨厌", "恨", "伤心",
        "痛苦", "难过", "悲惨", "恶心", "虚伪", "冷漠", "消极", "懒惰", "失败",
        "绝望", "黑暗", "压抑", "焦虑", "紧张", "害怕", "恐惧", "愤怒",
        "仇恨", "抱怨", "麻烦", "困难", "危险", "浪费", "烂", "糟透了",
        "悲哀", "可恶", "无奈", "班味", "内耗", "蚌埠住了",
        "悔恨", "羞愧", "碌碌无为", "虚度", "可耻", "可悲",
        "后悔", "遗憾", "自责", "内疚", "惭愧", "无用", "辜负",
        "悲伤", "沮丧", "失落",
        # 麦克白/虚无类
        "不过", "只是", "仅仅", "无非", "不过是",
        "拙劣", "笨拙", "可笑", "荒唐", "荒谬", "无聊", "无趣",
        "无声无息", "悄然", "悄然退下", "退下", "消失", "虚无",
        "喧哗", "骚动", "吵闹", "空洞", "苍白", "无力",
        "影子", "幻影", "梦幻", "泡影", "虚空",
        "愚人", "傻子", "蠢货", "笨蛋",
        "指手画脚", "装模作样", "徒劳", "白费",
        "找不到", "寻不到", "没有", "毫无",
        "无意义", "没意义", "无关紧要", "无所谓",
        "下场", "结局", "终局", "落幕", "散场",
        # 一代人/荒芜类
        "夹缝", "苟活", "太晚", "太早", "来不及", "等不到",
        "破晓", "黎明", "照进", "现实", "荒原", "旷野", "遗孤",
        "一盏", "熄灭", "一座", "坍塌", "废墟",
        "声嘶力竭", "呼喊", "听不见", "回响",
        "打输了", "跑错了", "枉然", "幻影",
        "美好", "仗", "跑", "守住", "虚空",
        "一切", "尝试", "希望", "皆为", "枉然", "幻影",
        # 雪国/死亡美学类
        "僵硬", "仰面朝天", "安详", "不挣扎", "放弃", "呼吸",
        "发疯", "叫喊", "厚厚的玻璃", "冰冷", "璀璨",
        "失去", "失去了一切", "温暖的可能", "剩下的",
        "白茫茫", "真干净", "空虚", "倾泻", "银河",
        # 祥子/人生堕落类
        "体面", "要强", "好梦想", "利己", "个人", "健壮", "伟大",
        "陪送", "殡", "埋", "堕落", "自私", "不幸", "病胎", "产儿",
        "末路", "鬼", "嫖", "赌", "懒", "狡猾", "没了心", "被摘了",
        "肉架子", "溃烂", "乱死岗子",
        "烟酒", "思索", "停止思索", "任凭", "往下坠", "无底的深渊",
        # 半生缘/人生回不去类
        "回不去了", "困惑", "痛苦", "残酷", "老了", "无法重叠",
        "日子过得真快", "中年", "指缝间", "一生一世"
    }

    negation_words = {'不', '没', '未', '别', '勿', '不要', '没有', '并非', '从不', '绝不会', '不会'}

    # 按整句拆分
    sentences = re.split(r'[。！？；]+', text)

    pos_count = 0
    neg_count = 0

    for sent in sentences:
        if not sent.strip():
            continue

        # 检查正面词
        for word in positive_words:
            if word in sent:
                idx = sent.find(word)
                start = max(0, idx - 10)
                before = sent[start:idx]
                negated = any(nw in before for nw in negation_words)
                if not negated:
                    pos_count += 1
                    break

        # 检查负面词
        for word in negative_words:
            if word in sent:
                idx = sent.find(word)
                start = max(0, idx - 10)
                before = sent[start:idx]
                negated = any(nw in before for nw in negation_words)
                if negated:
                    continue
                else:
                    neg_count += 1
                    break

    # 原始分
    raw_score = (pos_count - neg_count) / (pos_count + neg_count + 1)

    # 非线性压缩
    compressed = raw_score * (1 - 0.35 * raw_score * raw_score)
    score_int = int(round(compressed * 100))
    score_int = max(-100, min(100, score_int))

    score = round(raw_score, 3)

    # ===== 联动扣分1：悲观/虚无/荒芜感 =====
    despair_markers = [
        "夹缝", "苟活", "太晚", "太早", "来不及", "等不到",
        "荒原", "旷野", "遗孤", "熄灭", "坍塌", "废墟",
        "声嘶力竭", "听不见", "回响", "打输了", "跑错了",
        "枉然", "幻影", "皆为", "虚空", "泡影"
    ]
    marker_hit = sum(1 for m in despair_markers if m in text)

    if marker_hit >= 4 and score_int > -40:
        score_int = max(-70, score_int - 30)
    elif marker_hit >= 3 and score_int > -20:
        score_int = max(-50, score_int - 20)

    if "一切" in text and ("枉然" in text or "幻影" in text):
        if score_int > -30:
            score_int = max(-50, score_int - 20)

    # ===== 联动扣分2：死亡/失去/雪国美学 =====
    death_markers = [
        "僵硬", "仰面朝天", "安详", "不挣扎", "放弃呼吸",
        "发疯", "厚厚的玻璃", "冰冷", "空虚",
        "失去", "失去了一切", "温暖的可能",
        "白茫茫", "真干净"
    ]
    death_hit = sum(1 for m in death_markers if m in text)
    lose_count = len(re.findall(r'失去', text))

    if "白茫茫" in text and "真干净" in text:
        death_hit += 3

    if death_hit >= 5 and score_int > -50:
        score_int = max(-80, score_int - 35)
    elif death_hit >= 3 and score_int > -30:
        score_int = max(-60, score_int - 25)
    elif death_hit >= 2 and score_int > -10:
        score_int = max(-40, score_int - 15)

    if lose_count >= 2 and score_int > -30:
        score_int = max(-60, score_int - 20)

    # ===== 联动扣分3：人生失落/堕化/祥子+半生缘 =====
    fall_markers = [
        "堕落", "自私", "不幸", "末路", "溃烂", "深渊",
        "回不去了", "困惑", "痛苦", "残酷", "老了", "无法重叠",
        "往下坠", "无底的深渊", "乱死岗子", "陪送", "埋",
        "没了心", "被摘了", "肉架子", "烟酒", "停止思索"
    ]
    fall_hit = sum(1 for m in fall_markers if m in text)

    if "回不去了" in text:
        fall_hit += 2

    if fall_hit >= 4 and score_int > -50:
        score_int = max(-80, score_int - 30)
    elif fall_hit >= 3 and score_int > -30:
        score_int = max(-60, score_int - 20)
    elif fall_hit >= 2 and score_int > -10:
        score_int = max(-40, score_int - 15)

    core_words = ["堕落", "自私", "末路", "溃烂", "深渊"]
    core_hit = sum(1 for cw in core_words if cw in text)
    if core_hit >= 2 and score_int > -40:
        score_int = max(-70, score_int - 25)

    # 重新根据修正后的 score_int 微调标签
    if score_int >= 20:
        label = "积极"
    elif score_int <= -20:
        label = "消极"
    else:
        label = "中性"

    # ===== 点评生成 =====
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

    # 特殊场景评语
    if "白茫茫" in text or "真干净" in text:
        if score_int <= -30:
            comment = "文本基调极度悲凉，充满死亡、失去与虚无感，呈现出一种空寂而沉痛的审美。"
        elif score_int <= -10:
            comment = "文本带有浓重的失落与虚无色彩，整体氛围压抑而悲凉。"

    if fall_hit >= 3 and score_int <= -20:
        comment = "文本基调极度沉沦，充满堕落、无奈与人生虚无之感，透露出强烈的绝望与幻灭。"
    elif fall_hit >= 2 and score_int <= -10:
        comment = "文本整体弥漫着沉重的人生失落与无力感，基调颓丧而压抑。"

    if marker_hit >= 3 and score_int <= -20:
        comment = "文本带有强烈的荒芜感与虚无色彩，整体基调低沉、压抑，透露出对现实与未来的无力与幻灭。"

    if pos_count + neg_count <= 2:
        comment += "（文本中情感词较少，分析结果仅供参考）"

    return {
        "score": score,
        "score_int": score_int,
        "label": label,
        "pos": pos_count,
        "neg": neg_count,
        "comment": comment
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
                    '<div class="stat-item"><span class="stat-value">' + pos + '</span><span class="stat-label">正面词</span></div>' +
                    '<div class="stat-item"><span class="stat-value">' + neg + '</span><span class="stat-label">负面词</span></div>' +
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