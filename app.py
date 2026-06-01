import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# 網頁基本設定
st.set_page_config(page_title="AI動能妖股雷達 (全市場降載版)", page_icon="🚀", layout="wide")

# ==========================================
# 核心功能模組
# ==========================================
@st.cache_data(ttl=3600)
def get_tw_stock_list():
    stock_dict = {}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for m in [2, 4]:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
            res = requests.get(url, headers=headers, verify=False, timeout=15)
            df = pd.read_html(StringIO(res.text))[0].iloc[1:]
            for _, row in df.iterrows():
                try:
                    code_name = str(row[0]).split()
                    if len(code_name) == 2:
                        code, name = code_name
                        cat = str(row[4])
                        if len(code) == 4:
                            suffix = ".TW" if m == 2 else ".TWO"
                            stock_dict[f"{code}{suffix}"] = {"name": name, "sector": cat}
                except Exception: 
                    continue
    except Exception as e: 
        print(f"獲取台股清單發生錯誤: {e}")
    return stock_dict

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0'}
        res = requests.get(url, headers=headers, timeout=15)
        df = pd.read_html(StringIO(res.text))[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        names = df['Security'].tolist()
        sectors = df['GICS Sector'].tolist()
        return {t: {"name": n, "sector": s} for t, n, s in zip(tickers, names, sectors)}
    except Exception: 
        return {}

def check_market(symbol):
    try:
        data = yf.download(symbol, period="50d", progress=False)
        close_series = data['Close']
        close = float(close_series.iloc[-1])
        ma20 = float(close_series.rolling(20).mean().iloc[-1])
        return close >= ma20, close, ma20
    except Exception as e:
        print(f"大盤檢查失敗: {e}")
        return True, 0, 0

# 將計算邏輯獨立，統一由 yfinance 餵入資料
def process_stock(ticker, df, stock_dict, market_name, vol_label, mkt_ret_20, records):
    try:
        if df.empty or len(df) < 120: return 
        df = df.dropna()
        
        close = df['Close']
        vol = df['Volume']
        high = df['High']
        low = df['Low']
        open_p = df['Open']
        
        c_close = float(close.iloc[-1])
        c_open = float(open_p.iloc[-1])
        c_high = float(high.iloc[-1])
        c_low = float(low.iloc[-1])
        
        # 避雷針過濾
        k_len = c_high - c_low
        if k_len > 0:
            upper_shadow = (c_high - max(c_open, c_close)) / k_len
            if upper_shadow > 0.5: return 
                
        # 爆量倍數
        vol_20_mean = float(vol.tail(20).mean())
        vol_ratio = float(vol.iloc[-1]) / (vol_20_mean + 1e-9)

        # 爆量黑K過濾
        is_black_k = c_close < c_open
        if vol_ratio > 2.5 and is_black_k: return 
        
        roll_5 = close.rolling(5)
        roll_20 = close.rolling(20)
        roll_60 = close.rolling(60)
        
        ma5 = float(roll_5.mean().iloc[-1])
        if c_close < ma5: return  # 必須站在 5MA 之上
        
        ma5_bias = ((c_close - ma5) / (ma5 + 1e-9)) * 100
        ma20 = float(roll_20.mean().iloc[-1])
        ma60 = float(roll_60.mean().iloc[-1])
        
        # 流動性計算 (yfinance台股回傳的是股數，需要除以 1000 變為張)
        if "台股" in market_name:
            avg_vol = float((vol.tail(5).mean()) / 1000)
        else:
            avg_vol = float(vol.tail(5).mean())
        
        stock_ret_20 = float((c_close / float(close.iloc[-21]) - 1) * 100)
        rs_20 = stock_ret_20 - mkt_ret_20
        
        past_120_max = float(close.iloc[-121:-1].max())
        dist_120_high = float((c_close / past_120_max - 1) * 100) if past_120_max > 0 else 0

        daily_ret = close.pct_change()
        hist_vol = float(daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
        std20 = float(roll_20.std().iloc[-1])
        bb_upper = float(ma20 + 2 * std20)
        bb_width = float((bb_upper - (ma20 - 2 * std20)) / (ma20 + 1e-9) * 100)
        
        trend_str = (ma5 / (ma60 + 1e-9) - 1) * 100
        p_to_ma20 = (c_close / (ma20 + 1e-9) - 1) * 100
        p_to_bbupper = (c_close / (bb_upper + 1e-9) - 1) * 100
        roc_10 = float((c_close - close.iloc[-11]) / (close.iloc[-11] + 1e-9) * 100)
        
        if np.isnan(hist_vol) or np.isnan(roc_10): return

        records.append({
            'ID': ticker.replace(".TW", "").replace(".TWO", ""),
            '股名': stock_dict[ticker]['name'],
            '板塊產業': stock_dict[ticker]['sector'],
            '收盤價': round(c_close, 2),
            'MA5 (防守線)': round(ma5, 2),
            '5MA乖離率(%)': round(ma5_bias, 2),
            '爆量倍數': round(vol_ratio, 2),
            'RS相對強度': round(rs_20, 2),       
            '120日高距離(%)': round(dist_120_high, 2), 
            'Avg_Vol': avg_vol, 
            vol_label: round(avg_vol / 1000000, 2) if "美股" in market_name else int(avg_vol),
            'F_RS': rs_20, 'F_120_High': dist_120_high, 'F_Vol_Ratio': vol_ratio, 
            'F_Hist_Vol': hist_vol, 'F_BB_Width': bb_width, 'F_Trend_Strength': trend_str, 
            'F_P_to_MA20': p_to_ma20, 'F_P_to_BBUpper': p_to_bbupper, 'F_ROC_10': roc_10
        })
    except Exception as e:
        pass # 遇到計算錯誤靜默略過，不干擾主畫面

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_calculate_features(market_name):
    records = []
    
    # 決定抓取母體清單
    if "台股" in market_name:
        stock_dict = get_tw_stock_list()
        vol_label = "5日均量(張)"
        market_ticker = "^TWII"
    else:
        stock_dict = get_sp500_tickers()
        vol_label = "5日均量(M)"
        market_ticker = "^GSPC"

    if not stock_dict:
        return pd.DataFrame(), vol_label

    # 抓取大盤資訊
    try:
        mkt_data = yf.download(market_ticker, period="1y", auto_adjust=True, progress=False)['Close']
        mkt_ret_20 = float((mkt_data.iloc[-1] / mkt_data.iloc[-21]) - 1) * 100
    except:
        mkt_ret_20 = 0.0

    all_tickers = list(stock_dict.keys())
    batch_size = 50
    
    # 全面採用 yfinance 引擎批次抓取
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            time.sleep(0.5) # 稍微喘息，避免被 yfinance 阻擋 IP
            data = yf.download(batch, period="1y", interval="1d", group_by='ticker', auto_adjust=True, progress=False, threads=True)
            for ticker in batch:
                df = data[ticker] if len(batch) > 1 else data
                process_stock(ticker, df, stock_dict, market_name, vol_label, mkt_ret_20, records)
        except Exception as e:
            print(f"yfinance 抓取批次失敗: {e}")
            continue
                
    return pd.DataFrame(records), vol_label

# ==========================================
# 網頁介面設計
# ==========================================
st.title("🚀 AI 動能妖股雷達 (全市場降載版)")
st.markdown("全面回歸 `yfinance` 雙引擎，無抓取次數限制，內建【相對強度 RS】與【半年新高突破】精準狙擊。")

st.sidebar.header("⚙️ 雷達設定")
market = st.sidebar.radio("選擇掃描市場", ["🇹🇼 台股 (API: yfinance)", "🇺🇸 美股 (API: yfinance)"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 策略微調")
user_vol_limit = st.sidebar.number_input("最小均量限制 (台:張 / 美:百萬股)", min_value=100, max_value=20000, value=1000, step=100)
user_bias_limit = st.sidebar.slider("乖離率扣分門檻 (%)", min_value=1, max_value=15, value=5)
user_penalty = st.sidebar.number_input("超過門檻每 1% 扣幾分?", min_value=1, max_value=20, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.info("💡 **教練實戰紀律提醒**\n\n進場後若收盤跌破 5MA，請無條件執行停損。")

if st.button("開始全面掃描", type="primary"):
    market_ticker = "^TWII" if "台股" in market else "^GSPC"
    is_bull, idx_close, idx_ma = check_market(market_ticker)

    if is_bull:
        st.success(f"🟢 【大盤偏多】目前指數 ({idx_close:.2f}) 站上月線 ({idx_ma:.2f})，適合動能策略！")
    else:
        st.error(f"🔴 【大盤偏空】目前指數 ({idx_close:.2f}) 跌破月線 ({idx_ma:.2f})，極易假突破，建議空手觀望！")

    with st.status(f"🔍 啟動 {market} 運算中 (包含 RS 大盤比對，約需 1-2 分鐘)...", expanded=True) as status:
        df_all, vol_label = fetch_and_calculate_features(market)
        
        if df_all.empty:
            status.update(label="❌ 掃描失敗或無符合標的", state="error", expanded=False)
            st.error("目前無法取得數據，請確認網路連線或稍後再試。")
            st.stop()
            
        df_records = df_all[df_all['Avg_Vol'] >= user_vol_limit].copy()
        
        if df_records.empty:
            status.update(label="❌ 無符合條件的標的", state="error", expanded=False)
            st.warning(f"目前沒有任何標的的成交量大於 {user_vol_limit}，請嘗試調低標準。")
            st.stop()

        features = ['F_RS', 'F_Vol_Ratio', 'F_Hist_Vol', 'F_120_High', 'F_BB_Width', 'F_Trend_Strength', 'F_P_to_MA20', 'F_P_to_BBUpper', 'F_ROC_10']
        weights =  [20.0,  15.0,          15.0,         10.0,         10.0,         10.0,               10.0,          5.0,              5.0] 

        for f in features: df_records[f + '_Rank'] = df_records[f].rank(pct=True)
        
        df_records['AI 總分'] = 0.0
        for f, w in zip(features, weights): df_records['AI 總分'] += df_records[f + '_Rank'] * w
        
        df_records['乖離懲罰分'] = df_records['5MA乖離率(%)'].apply(lambda x: (x - user_bias_limit) * -user_penalty if x > user_bias_limit else 0)
        df_records['AI 總分'] = df_records['AI 總分'] + df_records['乖離懲罰分']
        df_records['AI 總分'] = df_records['AI 總分'].round(2)
        
        top20 = df_records.sort_values(by='AI 總分', ascending=False).head(20)
        
        status.update(label="✅ 掃描與運算完成！", state="complete", expanded=False)

    display_cols = ['ID', '股名', '板塊產業', '收盤價', 'MA5 (防守線)', '5MA乖離率(%)', '爆量倍數', 'RS相對強度', '120日高距離(%)', vol_label, 'AI 總分']
    st.dataframe(top20[display_cols], width="stretch", hide_index=True)
    
    st.info(f"💡 **乖離率實戰指南**：🟢 0% - 3% 首選試單 ｜ 🟡 3% - {user_bias_limit}% 注意追高｜ 🔴 >{user_bias_limit}% 已自動扣分處罰。")
    
    st.markdown("---")
    st.markdown("### 🔥 今日資金匯聚熱區 (前 20 名板塊統計)")
    sector_counts = top20['板塊產業'].value_counts().reset_index()
    sector_counts.columns = ['板塊產業', '進榜檔數']
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(sector_counts, hide_index=True, width="stretch")
    with col2:
        st.bar_chart(sector_counts.set_index('板塊產業'))

    csv = top20.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下載完整 CSV 報表",
        data=csv,
        file_name=f"Radar_Top20_Pro_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )