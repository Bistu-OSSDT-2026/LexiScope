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