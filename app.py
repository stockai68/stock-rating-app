import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="纳斯达克AI评分仪表盘", layout="wide")

st.title("纳斯达克 AI 股票评分系统（实时数据）")

tickers_input = st.text_input(
    "请输入股票代码（最多100只，用英文逗号分隔，例如：AAPL,TSLA,NVDA）",
    "AAPL,TSLA,NVDA,AMZN,MSFT,META"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")[:100]]

@st.cache_data(ttl=60)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="7d")
    if hist.empty:
        return None
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE", np.nan),
        "ps": info.get("priceToSalesTrailing12Months", np.nan),
        "eps_growth": info.get("earningsQuarterlyGrowth", 0),
        "rsi": compute_rsi(hist['Close']),
        "moneyflow": hist["Volume"].iloc[-1] * hist["Close"].iloc[-1],
        "news_sentiment": np.random.uniform(-1, 1)  # 模拟情绪
    }

def compute_rsi(prices, window=14):
    delta = prices.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def score(row):
    score = 0
    score += (1 - min(row['pe'] / 80, 1)) * 10
    score += (1 - min(row['ps'] / 20, 1)) * 10
    score += min(max((row['eps_growth'] + 0.1) / 0.6, 0), 1) * 20
    if 30 <= row['rsi'] <= 70:
        score += 20
    elif row['rsi'] < 30:
        score += 10
    else:
        score += 5
    score += min(max(row['moneyflow'] / 2e9, 0), 1) * 20
    score += min(max((row['news_sentiment'] + 1) / 2, 0), 1) * 20
    return round(score, 2)

with st.spinner("正在抓取数据并打分..."):
    rows = []
    for t in tickers:
        try:
            row = get_stock_data(t)
            if row:
                rows.append(row)
        except:
            continue
    df = pd.DataFrame(rows)

if not df.empty:
    df['score'] = df.apply(score, axis=1)
    df['recommend'] = df['score'].apply(lambda x: 'Buy' if x >= 75 else ('Hold' if x >= 50 else 'Sell'))

    filter_buy = st.checkbox("✅ 只显示推荐为 Buy 的股票")
    df_display = df[['ticker', 'score', 'recommend', 'pe', 'ps', 'rsi', 'eps_growth', 'moneyflow', 'news_sentiment']]

    if filter_buy:
        df_display = df_display[df_display['recommend'] == 'Buy']

    st.dataframe(df_display.sort_values("score", ascending=False), use_container_width=True)

else:
    st.warning("未获取到有效数据，请检查股票代码是否正确。")

   
