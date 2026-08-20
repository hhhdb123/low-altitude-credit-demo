"""
低空企业信贷风险评估 Demo（熵权 TOPSIS 综合评分）
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# 1. 全局配置
# ---------------------------------------------------------------------------

INDICATORS = [
    ("营业利润率",     "盈利能力", "positive", "%"),
    ("毛利率",         "盈利能力", "positive", "%"),
    ("净利率",         "盈利能力", "positive", "%"),
    ("ROA",            "盈利能力", "positive", "%"),
    ("营业利润增长率", "成长性",   "positive", "%"),
    ("研发投入增长率", "成长性",   "positive", "%"),
    ("流动比率",       "偿债能力", "positive", "倍"),
    ("速动比率",       "偿债能力", "positive", "倍"),
    ("资产负债率",     "偿债能力", "negative", "%"),
    ("应收账款周转率", "运营效率", "positive", "次"),
    ("存货周转率",     "运营效率", "positive", "次"),
    ("总资产周转率",   "运营效率", "positive", "次"),
    ("无形资产占比",   "低空投入", "positive", "%"),
    ("研发费用占比",   "低空投入", "positive", "%"),
    ("资本支出占比",   "低空投入", "positive", "%"),
]

DIM_ORDER = ["盈利能力", "成长性", "偿债能力", "运营效率", "低空投入"]

FEATURE_IMPORTANCE = {
    "速动比率": 0.4550, "毛利率": 0.0852, "营业利润率": 0.0849, "资本支出占比": 0.0805,
    "营业利润增长率": 0.0434, "资产负债率": 0.0405, "研发费用占比": 0.0339, "ROA": 0.0329,
    "总资产周转率": 0.0321, "存货周转率": 0.0256, "净利率": 0.0219, "研发投入增长率": 0.0178,
    "无形资产占比": 0.0177, "流动比率": 0.0175, "应收账款周转率": 0.0110,
}

RISK_THRESHOLDS = {"low": 0.3021, "high": 0.2131}

# 低饱和度 slate 同色系
PALETTE = {
    "primary":   "#475569",  # slate-600
    "secondary": "#64748b",  # slate-500
    "light":     "#94a3b8",  # slate-400
    "lighter":   "#cbd5e1",  # slate-300
    "bg":        "#f8fafc",  # slate-50
    "dark":      "#1e293b",  # slate-800
    "darker":    "#0f172a",  # slate-900
    "rule":      "#e2e8f0",  # slate-200
    # 红黄绿 - 风险等级色
    "risk_high": "#dc2626",  # red-600   高风险
    "risk_mid":  "#d97706",  # amber-600 中风险
    "risk_low":  "#16a34a",  # green-600 低风险
}


# ---------------------------------------------------------------------------
# 2. 字体注入 + 配色样式（必须在 set_page_config 之后、st.title 之前）
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="低空企业信贷风险评估",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 字体注入（用 st.html 避免 markdown 把 <link> 当作文本）
st.html(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/noto-sans-sc/index.css">'
)

# CSS 注入：用 st.html 直接插入 <style>，保证被浏览器解析
st.html(
    """
<style>
html, body,
[data-testid="stAppViewContainer"], [data-testid="stHeader"],
.stApp, .main, .block-container,
h1, h2, h3, h4, h5, h6,
p, span, div, label, button, input, select, textarea,
td, th, li,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stMarkdownContainer"] {
    font-family: 'Noto Sans SC', 'Source Han Sans SC', 'PingFang SC',
                 'Hiragino Sans GB', 'Microsoft YaHei', system-ui, sans-serif !important;
    font-weight: 400;
}

/* === 配色 === */
.stApp { background: #ffffff; }
h1 { color: #0f172a !important; font-weight: 700 !important; font-size: 1.75rem !important; }
h2 { color: #1e293b !important; font-weight: 500 !important; font-size: 1.1rem !important; }
h3 { color: #64748b !important; font-weight: 500 !important; font-size: 0.95rem !important; }
p, span, label, td, th, li { color: #1e293b; }
[data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 500 !important; font-size: 0.85rem !important; }
[data-testid="stMetricDelta"] { color: #64748b !important; }

/* 输入控件 */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #475569 !important;
    border-color: #475569 !important;
}
.stSlider [data-baseweb="slider"] > div > div {
    background: #cbd5e1 !important;
}
.stSlider label p { color: #1e293b !important; font-weight: 500 !important; }

/* 按钮 */
.stButton > button,
.stDownloadButton > button {
    background: #475569 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 500 !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #1e293b !important; color: #ffffff !important;
}

/* 折叠面板 */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background: #f8fafc !important;
}

/* summary 容器：flex 让 ::before 和子元素水平排列 */
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    padding: 8px 12px !important;
    list-style: none !important;
    color: #1e293b !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    visibility: visible !important;
}

/* 抑制浏览器默认 disclosure 三角 */
[data-testid="stExpander"] summary::-webkit-details-marker,
[data-testid="stExpander"] summary::marker {
    display: none !important;
    content: "";
}

/* 自定义 CSS 三角箭头（展开时旋转） */
[data-testid="stExpander"] summary::before {
    content: "▸";
    color: #64748b;
    font-size: 14px;
    width: 18px;
    height: 18px;
    line-height: 18px;
    margin-right: 10px;
    flex-shrink: 0;
    text-align: center;
    transition: transform 0.2s ease;
    display: inline-block;
    visibility: visible !important;
}
[data-testid="stExpander"][open] > summary::before {
    transform: rotate(90deg);
}

/* summary 内任何元素都强制可见（JS 会把真正的 chevron 图标置 display:none） */
[data-testid="stExpander"] summary * {
    visibility: visible !important;
}

/* 提示框 */
.stAlert, [data-testid="stAlert"] {
    background: #f8fafc !important;
    color: #1e293b !important;
    border-left: 3px solid #475569 !important;
}
[data-testid="stAlert"] div, [data-testid="stAlert"] p { color: #1e293b !important; }

/* 表格 */
.stDataFrame { border: 1px solid #e2e8f0 !important; }

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
    min-width: 320px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1e293b !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #1e293b !important; }

/* 收起按钮仍可点击，但侧边栏不被自动收起 */
[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0 !important;
}

/* 侧边栏收起/展开按钮加文字标签 */
[data-testid="stSidebarCollapsedControl"]::before,
[data-testid="stSidebarCollapseButton"]::before {
    content: "财务指标";
    font-size: 11px !important;
    color: #475569 !important;
    font-family: 'Noto Sans SC', sans-serif !important;
    font-weight: 500 !important;
    margin-right: 4px !important;
    writing-mode: horizontal-tb !important;
}

/* 分割线 */
hr { border-color: #e2e8f0 !important; margin: 0.5rem 0 !important; }

/* 顶部菜单与装饰栏隐藏 */
#MainMenu, [data-testid="stToolbar"], footer { visibility: hidden; }
</style>
"""
)

# 清理 Streamlit expander 默认 chevron 图标的英文文字 (keyboard_arrow_right 等)
components.html(
    """<script>
(function() {
    // 匹配所有 Material Symbols 风格的图标名（chevron、arrow、expand、keyboard_double 等）
    var PATTERNS = /^(keyboard_arrow|keyboard_double|arrow_drop|arrow_back|arrow_forward|expand_more|expand_less|chevron_|menu)/;
    function isPureIconNode(el) {
        // 只含一个文本节点的 span/div/i 才视为图标容器
        if (el.childNodes.length !== 1) return false;
        var only = el.childNodes[0];
        if (only.nodeType !== 3) return false;
        var t = (only.nodeValue || '').trim();
        return PATTERNS.test(t);
    }
    function clean() {
        try {
            // 扫描整个页面（不只是 summary），覆盖 sidebar 折叠按钮等所有 Material Symbols 图标
            var all = parent.document.querySelectorAll('span, div, i');
            Array.from(all).forEach(function(el) {
                if (isPureIconNode(el)) {
                    el.style.display = 'none';
                }
            });
        } catch(e) {}
    }
    clean();
    setTimeout(clean, 100);
    setTimeout(clean, 500);
    setTimeout(clean, 1500);
    setTimeout(clean, 3000);
    try {
        new MutationObserver(clean).observe(parent.document.body, {childList: true, subtree: true});
    } catch(e) {}
})();
</script>""",
    height=0,
)


# ---------------------------------------------------------------------------
# 3. 数据加载与算法
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("company_features.csv", encoding="utf-8-sig")


@st.cache_data
def compute_reference(df: pd.DataFrame) -> dict:
    cols = [t[0] for t in INDICATORS]
    X = df[cols].values.astype(float)

    raw_min_arr, raw_max_arr = X.min(axis=0), X.max(axis=0)

    F = X.copy()
    for j, (_, _, polarity, _) in enumerate(INDICATORS):
        if polarity == "negative":
            F[:, j] = X[:, j].max() - X[:, j]

    f_min, f_max = F.min(axis=0), F.max(axis=0)
    rng = f_max - f_min
    rng[rng == 0] = 1.0
    R = (F - f_min) / rng

    Z = R + 1e-12
    p = Z / Z.sum(axis=0, keepdims=True)
    n = Z.shape[0]
    e = -np.sum(p * np.log(p), axis=0) / np.log(n)
    d = 1 - e
    w = d / d.sum()

    V = R * w
    return {
        "weights":         dict(zip(cols, w)),
        "raw_min":         dict(zip(cols, raw_min_arr)),
        "raw_max":         dict(zip(cols, raw_max_arr)),
        "f_min":           dict(zip(cols, f_min)),
        "f_max":           dict(zip(cols, f_max)),
        "weights_arr":     w,
        "R_sample":        R,
        "V_plus_per_col":  V.max(axis=0),
        "V_minus_per_col": V.min(axis=0),
        "V_sample":        V,
    }


def score_one(user_input: dict, ref: dict) -> dict:
    cols = [t[0] for t in INDICATORS]
    raw = np.array([user_input[c] for c in cols], dtype=float)

    forward = raw.copy()
    for j, (_, _, polarity, _) in enumerate(INDICATORS):
        if polarity == "negative":
            forward[j] = ref["raw_max"][cols[j]] - raw[j]

    norm = np.zeros_like(forward)
    for j in range(len(cols)):
        lo, hi = ref["f_min"][cols[j]], ref["f_max"][cols[j]]
        if hi <= lo:
            norm[j] = 0.5
        else:
            norm[j] = float(np.clip((forward[j] - lo) / (hi - lo), 0.0, 1.0))

    w = ref["weights_arr"]
    V = norm * w
    V_plus  = ref["V_plus_per_col"]
    V_minus = ref["V_minus_per_col"]
    D_plus  = float(np.sqrt(((V - V_plus)  ** 2).sum()))
    D_minus = float(np.sqrt(((V - V_minus) ** 2).sum()))
    score   = D_minus / (D_plus + D_minus + 1e-12)

    dim_score = {}
    for dim in DIM_ORDER:
        idx = [j for j, (_, dim_j, _, _) in enumerate(INDICATORS) if dim_j == dim]
        dim_score[dim] = float(norm[idx].mean())

    risks = []
    for j, (name, dim, polarity, _) in enumerate(INDICATORS):
        gap = 1.0 - norm[j]
        importance = FEATURE_IMPORTANCE.get(name, 0.0)
        risks.append({
            "指标": name,
            "维度": dim,
            "极性": "极小型" if polarity == "negative" else "极大型",
            "原始值": float(raw[j]),
            "归一化值": float(norm[j]),
            "重要性": float(importance),
            "风险贡献": float(gap * importance),
        })
    risks_sorted = sorted(risks, key=lambda r: r["风险贡献"], reverse=True)

    return {
        "score":         score,
        "dim_score":     dim_score,
        "dim_score_T":   {k: v * 100 for k, v in dim_score.items()},
        "risks":         risks_sorted,
        "norm_values":   dict(zip(cols, norm)),
    }


def risk_level(score: float) -> tuple[str, str, str]:
    # 红 / 黄 / 绿 区分风险等级
    if score >= RISK_THRESHOLDS["low"]:
        return "低风险", PALETTE["risk_low"], "得分位于安全区间，违约概率较低。"
    elif score >= RISK_THRESHOLDS["high"]:
        return "中风险", PALETTE["risk_mid"], "得分位于观察区间，存在一定违约风险。"
    else:
        return "高风险", PALETTE["risk_high"], "得分位于高风险区间，违约概率显著上升。"


def default_prob(score: float) -> float:
    z = 15.0 * (score - RISK_THRESHOLDS["high"])
    return float(1.0 / (1.0 + np.exp(z)))


# ---------------------------------------------------------------------------
# 4. 页面
# ---------------------------------------------------------------------------

st.title("低空企业信贷风险评估")

df_ref = load_data()
ref    = compute_reference(df_ref)

user_input: dict = {t[0]: float(df_ref[t[0]].mean()) for t in INDICATORS}

with st.sidebar:
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:500;color:#1e293b;'
        'margin:0 0 12px 0;display:flex;align-items:center;gap:8px;">'
        '<span style="color:#64748b;font-size:0.95rem;">▾</span>'
        '<span>财务指标输入</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    for dim in DIM_ORDER:
        with st.expander(dim, expanded=True):
            for name, dim_j, polarity, unit in INDICATORS:
                if dim_j != dim:
                    continue
                lo = float(ref["raw_min"][name])
                hi = float(ref["raw_max"][name])
                if hi <= lo:
                    hi = lo + 1.0
                default = float(np.clip(df_ref[name].mean(), lo, hi))
                step = (hi - lo) / 200 if hi > lo else 0.01
                user_input[name] = st.slider(
                    label=f"{name} ({unit})",
                    min_value=lo, max_value=hi, value=default, step=step,
                    key=f"in_{name}",
                )

# 主区
result   = score_one(user_input, ref)
score    = result["score"]
level, color, desc = risk_level(score)
pd_val   = default_prob(score)
score_pct = score * 100

col1, col2, col3 = st.columns(3)
col1.metric("综合得分", f"{score:.4f}", f"{score_pct:.1f} / 100")
col2.metric("风险等级", level)
col3.metric("违约概率", f"{pd_val*100:.1f}%", f"{(0.5-pd_val)*100:+.1f}%")

# 风险等级条（红→黄→绿，对应高/中/低风险）
st.markdown(
    f"""
<div style="background:linear-gradient(90deg,
    {PALETTE['risk_high']} 0%, {PALETTE['risk_high']} 25%,
    {PALETTE['risk_mid']} 25%, {PALETTE['risk_mid']} 55%,
    {PALETTE['risk_low']} 55%, {PALETTE['risk_low']} 100%);
    height:22px;border-radius:11px;position:relative;margin:8px 0 12px 0;">
  <div style="position:absolute;left:{min(max(score_pct,0),100)}%;top:-4px;
    transform:translateX(-50%);">
    <div style="background:{color};color:#ffffff;
        padding:2px 10px;border-radius:6px;font-size:12px;white-space:nowrap;">
      {level} · {score:.4f}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(desc)

st.markdown("<hr>", unsafe_allow_html=True)

# 雷达图 + 风险因子
left, right = st.columns(2)

with left:
    st.subheader("5 维度得分雷达图")
    radar_df = pd.DataFrame({
        "维度": list(result["dim_score_T"].keys()),
        "得分": list(result["dim_score_T"].values()),
    })
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_df["得分"].tolist() + [radar_df["得分"].iloc[0]],
        theta=radar_df["维度"].tolist() + [radar_df["维度"].iloc[0]],
        fill="toself", name="本企业",
        line=dict(color=color, width=2),
        fillcolor="rgba(71, 85, 105, 0.28)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=10, color=PALETTE["secondary"]),
                gridcolor=PALETTE["lighter"], linecolor=PALETTE["lighter"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=PALETTE["dark"]),
                gridcolor=PALETTE["lighter"], linecolor=PALETTE["lighter"],
            ),
        ),
        showlegend=False, height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("核心风险因子 Top 5")
    risk_df = pd.DataFrame(result["risks"][:5])
    risk_df["归一化值"] = risk_df["归一化值"].round(3)
    risk_df["风险贡献"] = risk_df["风险贡献"].round(4)
    risk_df["重要性"] = risk_df["重要性"].round(4)
    st.dataframe(
        risk_df[["指标", "维度", "极性", "原始值", "归一化值", "重要性", "风险贡献"]],
        use_container_width=True, hide_index=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("15 项指标归一化明细"):
    detail = []
    for name, dim, polarity, _ in INDICATORS:
        detail.append({
            "维度":     dim,
            "指标":     name,
            "极性":     "极小型" if polarity == "negative" else "极大型",
            "原始值":   round(user_input[name], 4),
            "权重":     round(ref["weights"][name], 4),
            "归一化值": round(result["norm_values"][name], 4),
        })
    st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

with st.expander("模型说明"):
    st.markdown(
        """
**综合评分 5 步流程**

1. **标准化处理**：资产负债率极小型用 max−x 正向化；其余 14 项极大型。Min-Max 归一化到 [0,1]。
2. **熵权法**：w_j = (1 − e_j) / Σ(1 − e_i)，权重反映指标在样本间变异度。
3. **加权矩阵**：V = R · w。
4. **TOPSIS 距离**：D⁺ = √Σ(V − V⁺)²，D⁻ = √Σ(V − V⁻)²。
5. **综合得分**：Score = D⁻ / (D⁺ + D⁻)。

**风险等级阈值**

- 低风险：Score ≥ 0.3021
- 中风险：0.2131 ≤ Score < 0.3021
- 高风险：Score < 0.2131

**违约概率**：Sigmoid 函数 PD = 1 / (1 + exp(15·(Score − 0.2131)))，
在中/高边界处 PD = 0.5，向两端平滑收敛。
"""
    )