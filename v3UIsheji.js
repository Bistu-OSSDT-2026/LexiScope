// ============================================================
// === 队员E-前端：情感展示卡片适配深度学习模型输出 ===
// ============================================================

function renderSentiment(sentiment) {
    var result = document.getElementById('sentimentResult');
    result.innerHTML = '';
    if (!sentiment || sentiment.score_int === undefined) {
        result.innerHTML = '<div class="sentiment-score">暂无数据</div>';
        return;
    }
    var scoreInt = sentiment.score_int;
    var label = sentiment.label || '中性';
    var pos = sentiment.pos || 0;      // 新模型返回浮点数（概率和）
    var neg = sentiment.neg || 0;      // 新模型返回浮点数（概率和）
    var comment = sentiment.comment || '';

    var labelClass = 'sentiment-neutral';
    if (label === '积极') labelClass = 'sentiment-positive';
    if (label === '消极') labelClass = 'sentiment-negative';

    var leftPercent = ((scoreInt + 100) / 200) * 100;
    if (leftPercent < 0) leftPercent = 0;
    if (leftPercent > 100) leftPercent = 100;

    // 关键修改：显示“积极概率和”与“消极概率和”，并保留两位小数
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