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
    err_msg = ""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        for m in [2, 4]:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={m}"
            res = requests.get(url, headers=headers, verify=False, timeout=15)
            res.raise_for_status() 
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
        err_msg = f"台股清單抓取失敗: {str(e)}"
    return stock_dict, err_msg

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    stock_dict = {}
    err_msg = ""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        df = pd.read_html(StringIO(res.text))[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        names = df['Security'].tolist()
        sectors = df['GICS Sector'].tolist()
        stock_dict = {t: {"name": n, "sector": s} for t, n, s in zip(tickers, names, sectors)}
    except Exception as e: 
        err_msg = f"美股清單抓取失敗: {str(e)}"
    return stock_dict, err_msg

def check_market(symbol):
    try:
        data = yf.Ticker(symbol).history(period="50d")
        if data.empty:
            return True, 0, 0
            
        close_vals = data['Close'].dropna().values
        close = float(close_vals[-1])
        ma20 = float(data['Close'].rolling(20).mean().dropna().values[-1])
        return close >= ma20, close, ma20
    except Exception as e:
        print(f"大盤檢查失敗: {e}")
        return True, 0, 0

def process_stock(ticker, df, stock_dict, market_name, vol_label, mkt_ret_20, records, debug_errors):
    try:
        df.columns = [str(c).capitalize() for c in df.columns]
        
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return
            
        df = df.dropna(subset=required_cols)
        if len(df) < 120: 
            return 
            
        c_vals = df['Close'].values
        o_vals = df['Open'].values
        h_vals = df['High'].values
        l_vals = df['Low'].values
        v_vals = df['Volume'].values
        
        c_close = float(c_vals[-1])
        c_open = float(o_vals[-1])
        c_high = float(h_vals[-1])
        c_low = float(l_vals[-1])
        c_vol = float(v_vals[-1])
        
        # 🔥 新增：計算今日與昨日漲跌幅 (加入保護避免除以 0)
        today_ret = ((c_vals[-1] / c_vals[-2]) - 1) * 100 if c_vals[-2] > 0 else 0
        yest_ret = ((c_vals[-2] / c_vals[-3]) - 1) * 100 if c_vals[-3] > 0 else 0
        
        # 避雷針過濾
        k_len = c_high - c_low
        if k_len > 0:
            upper_shadow = (c_high - max(c_open, c_close)) / k_len
            if upper_shadow > 0.5: return 
                
        # 爆量倍數
        vol_20_mean = float(np.mean(v_vals[-20:]))
        vol_ratio = c_vol / (vol_20_mean + 1e-9)

        # 爆量黑K過濾
        if vol_ratio > 2.5 and c_close < c_open: return 
        
        c_series = df['Close']
        ma5 = float(c_series.rolling(5).mean().dropna().values[-1])
        if c_close < ma5: return 
        
        ma5_bias = ((c_close - ma5) / (ma5 + 1e-9)) * 100
        ma20 = float(c_series.rolling(20).mean().dropna().values[-1])
        ma60 = float(c_series.rolling(60).mean().dropna().values[-1])
        
        avg_vol = float(np.mean(v_vals[-5:]))
        if "台股" in market_name:
            avg_vol = avg_vol / 1000.0
        
        c_close_21 = float(c_vals[-21])
        stock_ret_20 = ((c_close / c_close_21) - 1) * 100
        rs_20 = stock_ret_20 - mkt_ret_20
        
        past_120_max = float(np.max(c_vals[-121:-1]))
        dist_120_high = float((c_close / past_120_max - 1) * 100) if past_120_max > 0 else 0

        daily_ret = c_series.pct_change().dropna()
        std20_ret = daily_ret.rolling(20).std().dropna().values
        if len(std20_ret) == 0: return
        hist_vol = float(std20_ret[-1] * np.sqrt(252) * 100)
        
        std20_price = float(c_series.rolling(20).std().dropna().values[-1])
        bb_upper = float(ma20 + 2 * std20_price)
        bb_width = float((bb_upper - (ma20 - 2 * std20_price)) / (ma20 + 1e-9) * 100)
        
        trend_str = (ma5 / (ma60 + 1e-9) - 1) * 100
        p_to_ma20 = (c_close / (ma20 + 1e-9) - 1) * 100
        p_to_bbupper = (c_close / (bb_upper + 1e-9) - 1) * 100
        
        c_close_11 = float(c_vals[-11])
        roc_10 = float((c_close - c_close_11) / (c_close_11 + 1e-9) * 100)

        records.append({
            'ID': ticker.replace(".TW", "").replace(".TWO", ""),
            '股名': stock_dict[ticker]['name'],
            '板塊產業': stock_dict[ticker]['sector'],
            '收盤價': round(c_close, 2),
            '今日漲幅(%)': round(today_ret, 2), # 🔥 新增欄位
            '昨日漲幅(%)': round(yest_ret, 2), # 🔥 新增欄位
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
        debug_errors.append(f"標的 {ticker} 計算錯誤: {str(e)}") 

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_calculate_features(market_name):
    records = []
    debug_errors = [] 
    
    if "台股" in market_name:
        stock_dict, list_err = get_tw_stock_list()
        vol_label = "5日均量(張)"
        market_ticker = "^TWII"
    else:
        stock_dict, list_err = get_sp500_tickers()
        vol_label = "5日均量(M)"
        market_ticker = "^GSPC"

    if not stock_dict:
        return pd.DataFrame(), vol_label, [f"❌ 獲取股票母體清單失敗 ({list_err})"]

    try:
        mkt_data = yf.Ticker(market_ticker).history(period="1y")['Close']
        mkt_ret_20 = float((mkt_data.iloc[-1] / mkt_data.iloc[-21]) - 1) * 100
    except Exception:
        mkt_ret_20 = 0.0

    all_tickers = list(stock_dict.keys())
    batch_size = 50
    
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            time.sleep(0.5) 
            data = yf.download(batch, period="1y", interval="1d", group_by='ticker', auto_adjust=True, progress=False, threads=True)
            
            if data is None or data.empty: 
                continue
            
            for ticker in batch:
                try:
                    df = None
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.get_level_values(1):
                            df = data.xs(ticker, level=1, axis=1)
                        elif ticker in data.columns.get_level_values(0):
                            df = data[ticker]
                    else:
                        if len(batch) == 1: df = data
                            
                    if df is not None and not df.empty:
                        process_stock(ticker, df, stock_dict, market_name, vol_label, mkt_ret_20, records, debug_errors)
                except Exception as e:
                    debug_errors.append(f"{ticker} 抽取失敗: {str(e)}")
                    
        except Exception as e:
            debug_errors.append(f"批次 {i} 請求異常: {str(e)}")
            continue
                
    return pd.DataFrame(records), vol_label, debug_errors

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
        df_all, vol_label, debug_errors = fetch_and_calculate_features(market)
        
        if df_all.empty:
            status.update(label="❌ 掃描失敗或無符合標的", state="error", expanded=False)
            error_msg = "目前無法取得有效數據，可能是 `yfinance` 暫時阻擋了 IP 或資料結構異常。\n\n"
            if debug_errors:
                error_msg += "**系統除錯日誌 (前 5 筆)：**\n"
                for err in debug_errors[:5]: error_msg += f"- {err}\n"
            st.error(error_msg)
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

    # 🔥 新增：將今日與昨日漲幅加入顯示清單中
    display_cols = ['ID', '股名', '板塊產業', '收盤價', '今日漲幅(%)', '昨日漲幅(%)', 'MA5 (防守線)', '5MA乖離率(%)', '爆量倍數', 'RS相對強度', '120日高距離(%)', vol_label, 'AI 總分']
    st.dataframe(top20[display_cols], use_container_width=True, hide_index=True)
    
    st.info(f"💡 **乖離率實戰指南**：🟢 0% - 3% 首選試單 ｜ 🟡 3% - {user_bias_limit}% 注意追高｜ 🔴 >{user_bias_limit}% 已自動扣分處罰。")
    
    st.markdown("---")
    st.markdown("### 🔥 今日資金匯聚熱區 (前 20 名板塊統計)")
    sector_counts = top20['板塊產業'].value_counts().reset_index()
    sector_counts.columns = ['板塊產業', '進榜檔數']
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(sector_counts, hide_index=True, use_container_width=True)
    with col2:
        st.bar_chart(sector_counts.set_index('板塊產業'))

    csv = top20.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下載完整 CSV 報表",
        data=csv,
        file_name=f"Radar_Top20_Pro_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
