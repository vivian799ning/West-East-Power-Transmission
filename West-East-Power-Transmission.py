
# -*- coding: utf-8 -*-
"""
云南省河流水位与发电量相关性分析 - Streamlit应用
author:pwy
"""

import streamlit as st
import pandas as pd
import numpy as np
import pymysql
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 直流配置 - 【修改点1】在这里添加或修改直流信息
# ============================================================================
DC_CONFIG = {
    '楚穗直流': {
        'file_path': 'E:/pwy/work/word_data/云南/chushui_20251223140853.xlsx',
        'column_name': '云南-楚穗直流-西电通道（实际）',
        'color': '#1f77b4'
    },
    '昆柳龙直流': {
        'file_path': 'E:/pwy/work/word_data/云南/kunliulong_20251223140957.xlsx',
        'column_name': '云南-昆柳龙直流-西电通道（实际）',
        'color': '#ff7f0e'
    },
    '牛从直流': {
        'file_path': 'E:/pwy/work/word_data/云南/niucong_20251223141142.xlsx',
        'column_name': '云南-牛从直流-西电通道（实际）',
        'color': '#2ca02c'
    },
    '新东直流': {
        'file_path': 'E:/pwy/work/word_data/云南/xinodng_20251223141052.xlsx',
        'column_name': '云南-新东直流-西电通道（实际）',
        'color': '#d62728'
    },
    '普侨直流': {
        'file_path': 'E:/pwy/work/word_data/云南/puqiao_20251223141219.xlsx',
        'column_name': '云南-普侨直流-西电通道（实际）',
        'color': '#9467bd'
    }
}

# ============================================================================
# 页面配置 - 必须是第一个Streamlit命令
# ============================================================================
st.set_page_config(
    page_title="云南省河流水位与发电量分析",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 自定义CSS样式
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .dc-info {
        background-color: #e8f4f8;
        border-left: 4px solid #1E88E5;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 数据加载函数 - 使用缓存提高性能
# ============================================================================
@st.cache_data(ttl=3600)  # 缓存1小时
def load_river_data():
    """从数据库加载河流水位数据"""
    try:
        conn = pymysql.connect(
            host=st.secrets["db_host"],
            port=3306,
            database=st.secrets["db_name"],
            charset="utf8",
            user=st.secrets["db_user"],
            passwd=st.secrets["db_pass"]
            )
        
        sql = "SELECT * FROM water_rain_river"
        df = pd.read_sql(sql, conn)
        conn.close()
        
        # 处理时间字段
        df['time'] = pd.to_datetime(df['time'])
        
        # 筛选云南省数据（包含'云南'和'云南省'两种标记）
        df = df[df['region'].isin(['云南', '云南省'])]
        
        return df
    except Exception as e:
        st.error(f"数据库连接失败: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_power_data(dc_name):
    """加载指定直流的发电数据
    
    参数:
        dc_name: 直流名称，如 '楚穗直流'、'昆柳龙直流' 等
    """
    try:
        config = DC_CONFIG[dc_name]
        file_path = config['file_path']
        column_name = config['column_name']
        
        df = pd.read_excel(file_path)
        df['datetime'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['时点'].astype(str))
        df = df[['datetime', column_name]]
        df = df.rename(columns={column_name: 'power_actual'})
        
        # 聚合到日级别
        df['date'] = df['datetime'].dt.date
        df_daily = df.groupby('date').agg({'power_actual': 'sum'}).reset_index()
        df_daily.columns = ['date', 'power_sum']
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily = df_daily[df_daily['power_sum'].notna()]
        
        return df_daily
    except Exception as e:
        st.error(f"发电数据加载失败 ({dc_name}): {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_all_power_data():
    """加载所有直流的发电数据"""
    all_data = {}
    for dc_name in DC_CONFIG.keys():
        df = load_power_data(dc_name)
        if df is not None:
            all_data[dc_name] = df
    return all_data

def process_water_data(df_river, start_date, end_date):
    """处理水位数据"""
    df_water = df_river[
        (df_river['time'] >= start_date) & 
        (df_river['time'] <= end_date)
    ][['time', 'river_name', 'water_level']].copy()
    
    df_water = df_water.dropna(subset=['water_level'])
    df_water['water_level'] = df_water['water_level'].astype(float)
    df_water['date'] = pd.to_datetime(df_water['time'].dt.date)
    df_water = df_water.sort_values(by=['river_name', 'time']).reset_index(drop=True)
    
    return df_water

def calculate_correlation(water_values, power_values):
    """计算相关性指标"""
    if len(water_values) < 10:
        return None
    
    try:
        pearson_r, pearson_p = pearsonr(water_values, power_values)
        spearman_r, spearman_p = spearmanr(water_values, power_values)
        
        model = LinearRegression()
        model.fit(water_values.reshape(-1, 1), power_values)
        r2 = model.score(water_values.reshape(-1, 1), power_values)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r2': r2,
            'slope': slope,
            'intercept': intercept,
            'n': len(water_values)
        }
    except Exception as e:
        return None

# ============================================================================
# 绘图函数
# ============================================================================
def plot_timeseries(df_merged, river_name, dc_name, stats=None):
    """绘制时序图"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    dc_color = DC_CONFIG.get(dc_name, {}).get('color', '#1f77b4')
    
    # 发电量曲线
    fig.add_trace(
        go.Scatter(
            x=df_merged['date'],
            y=df_merged['power_sum'],
            name=f'{dc_name}日发电量(MWh)',
            line=dict(color=dc_color, width=1.5),
            opacity=0.8
        ),
        secondary_y=False
    )
    
    # 水位曲线
    fig.add_trace(
        go.Scatter(
            x=df_merged['date'],
            y=df_merged['water_level'],
            name=f'{river_name}水位(m)',
            line=dict(color='#ff7f0e', width=1.5),
            opacity=0.8
        ),
        secondary_y=True
    )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=f'{river_name} 水位 vs {dc_name}日发电量时间序列',
            font=dict(size=16)
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    fig.update_xaxes(title_text="日期")
    fig.update_yaxes(title_text=f"{dc_name}日发电量(MWh)", secondary_y=False, color=dc_color)
    fig.update_yaxes(title_text=f"{river_name}水位(m)", secondary_y=True, color='#ff7f0e')
    
    # 添加统计信息注释
    if stats:
        annotation_text = f"r = {stats['pearson_r']:.4f}<br>R² = {stats['r2']:.4f}<br>n = {stats['n']}"
        fig.add_annotation(
            x=0.98, y=0.98,
            xref="paper", yref="paper",
            text=annotation_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12)
        )
    
    return fig

def plot_scatter(df_merged, river_name, dc_name, stats=None):
    """绘制散点回归图"""
    fig = go.Figure()
    
    dc_color = DC_CONFIG.get(dc_name, {}).get('color', '#3498db')
    
    # 散点
    fig.add_trace(
        go.Scatter(
            x=df_merged['water_level'],
            y=df_merged['power_sum'],
            mode='markers',
            name='数据点',
            marker=dict(
                color=dc_color,
                size=6,
                opacity=0.5,
                line=dict(color='white', width=0.5)
            )
        )
    )
    
    # 回归线
    if stats:
        x_range = np.linspace(df_merged['water_level'].min(), df_merged['water_level'].max(), 100)
        y_pred = stats['slope'] * x_range + stats['intercept']
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name='回归线',
                line=dict(color='red', width=2.5)
            )
        )
    
    fig.update_layout(
        title=dict(
            text=f'{river_name} 水位 vs {dc_name}日发电量散点回归图',
            font=dict(size=16)
        ),
        xaxis_title=f'{river_name}水位(m)',
        yaxis_title=f'{dc_name}日发电量(MWh)',
        height=500,
        showlegend=True
    )
    
    # 添加统计信息
    if stats:
        annotation_text = (
            f"n = {stats['n']}<br>"
            f"r = {stats['pearson_r']:.4f}<br>"
            f"R² = {stats['r2']:.4f}<br>"
            f"y = {stats['slope']:.4f}x + {stats['intercept']:.2f}"
        )
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=annotation_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12),
            align="left"
        )
    
    return fig

def plot_multi_dc_comparison(df_water, all_power_data, selected_rivers, start_date, end_date):
    """绘制多直流对比图"""
    fig = go.Figure()
    
    # 处理选中河流的水位数据
    df_water_selected = df_water[df_water['river_name'].isin(selected_rivers)].copy()
    df_water_sum = df_water_selected.groupby('date').agg({'water_level': 'sum'}).reset_index()
    
    results = []
    
    for dc_name, df_power in all_power_data.items():
        # 筛选时间范围
        df_power_filtered = df_power[
            (df_power['date'] >= str(start_date)) & 
            (df_power['date'] <= str(end_date))
        ].copy()
        
        # 合并数据
        df_merged = pd.merge(df_power_filtered, df_water_sum, on='date', how='inner')
        df_merged = df_merged.dropna()
        
        if len(df_merged) >= 10:
            stats = calculate_correlation(
                df_merged['water_level'].values,
                df_merged['power_sum'].values
            )
            if stats:
                results.append({
                    '直流名称': dc_name,
                    'Pearson_r': stats['pearson_r'],
                    'R²': stats['r2'],
                    '样本量': stats['n']
                })
                
                # 添加散点
                fig.add_trace(
                    go.Scatter(
                        x=df_merged['water_level'],
                        y=df_merged['power_sum'],
                        mode='markers',
                        name=dc_name,
                        marker=dict(
                            color=DC_CONFIG[dc_name]['color'],
                            size=6,
                            opacity=0.6
                        )
                    )
                )
    
    fig.update_layout(
        title='各直流与河流水位相关性对比',
        xaxis_title='河流水位总和(m)',
        yaxis_title='日发电量(MWh)',
        height=500
    )
    
    return fig, pd.DataFrame(results)

# ============================================================================
# 主应用
# ============================================================================
def main():
    # 标题
    st.markdown('<p class="main-header">🌊 云南省河流水位与发电量相关性分析</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">西电通道数据分析平台</p>', unsafe_allow_html=True)
    
    # ========== 侧边栏 - 参数设置 ==========
    st.sidebar.header("📊 参数设置")
    
    # 【核心功能】直流选择
    st.sidebar.subheader("⚡ 直流选择")
    selected_dc = st.sidebar.selectbox(
        "选择直流线路",
        options=list(DC_CONFIG.keys()),
        index=0,
        help="选择要分析的直流输电线路"
    )
    
    # 显示当前直流信息
    st.sidebar.markdown(f"""
    <div class="dc-info">
        <b>当前直流：</b>{selected_dc}<br>
        <b>数据列：</b>{DC_CONFIG[selected_dc]['column_name']}
    </div>
    """, unsafe_allow_html=True)
    
    # 时间范围选择
    st.sidebar.subheader("📅 时间范围")
    
    # 动态获取当前日期
    today = datetime.now().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=datetime(2021, 1, 1),
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=today,  # 默认为当前日期
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
    
    # 加载数据
    with st.spinner("正在加载数据..."):
        df_river = load_river_data()
        df_power = load_power_data(selected_dc)
    
    if df_river is None or df_power is None:
        st.error("数据加载失败，请检查数据源连接")
        return
    
    # 处理水位数据
    df_water = process_water_data(df_river, str(start_date), str(end_date))
    
    # 筛选发电数据时间范围
    df_power_filtered = df_power[
        (df_power['date'] >= str(start_date)) & 
        (df_power['date'] <= str(end_date))
    ].copy()
    
    # 获取所有河流列表
    all_rivers = sorted(df_water['river_name'].unique().tolist())
    
    # 侧边栏显示数据概览
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 数据概览")
    st.sidebar.metric("当前直流", selected_dc)
    st.sidebar.metric("河流总数", f"{len(all_rivers)} 条")
    st.sidebar.metric("水位数据量", f"{len(df_water):,} 条")
    st.sidebar.metric("发电数据天数", f"{len(df_power_filtered):,} 天")
    
    # ========== 主页面标签页 ==========
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 单条河流分析", 
        "🔗 多河流组合分析", 
        "📈 所有河流汇总",
        "📋 数据总览",
        "🔄 多直流对比"
    ])
    
    # ========== Tab 1: 单条河流分析 ==========
    with tab1:
        st.header(f"单条河流与{selected_dc}发电量相关性分析")
        
        # 河流选择
        selected_river = st.selectbox(
            "选择河流",
            options=all_rivers,
            index=0 if len(all_rivers) > 0 else None,
            key="tab1_river"
        )
        
        if selected_river:
            # 获取该河流数据
            df_river_single = df_water[df_water['river_name'] == selected_river].copy()
            df_river_daily = df_river_single.groupby('date').agg({
                'water_level': 'mean'
            }).reset_index()
            
            # 合并数据
            df_merged = pd.merge(
                df_power_filtered[['date', 'power_sum']], 
                df_river_daily, 
                on='date', 
                how='inner'
            )
            df_merged = df_merged.dropna()
            
            # 计算相关性
            if len(df_merged) >= 10:
                stats = calculate_correlation(
                    df_merged['water_level'].values,
                    df_merged['power_sum'].values
                )
                
                # 显示统计指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pearson相关系数", f"{stats['pearson_r']:.4f}")
                with col2:
                    st.metric("R²决定系数", f"{stats['r2']:.4f}")
                with col3:
                    st.metric("样本量", f"{stats['n']}")
                with col4:
                    p_str = f"{stats['pearson_p']:.2e}" if stats['pearson_p'] < 0.001 else f"{stats['pearson_p']:.4f}"
                    st.metric("P值", p_str)
                
                # 绘制图表
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_timeseries(df_merged, selected_river, selected_dc, stats), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_scatter(df_merged, selected_river, selected_dc, stats), use_container_width=True)
                
                # 显示数据表格
                with st.expander("查看原始数据"):
                    st.dataframe(df_merged, use_container_width=True)
            else:
                st.warning(f"⚠️ 该河流有效数据点不足10个（当前: {len(df_merged)}），无法进行相关性分析")
    
    # ========== Tab 2: 多河流组合分析 ==========
    with tab2:
        st.header(f"多河流组合与{selected_dc}发电量相关性分析")
        st.info("💡 选择多条河流，系统将计算它们水位总和与发电量的相关性")
        
        # 多选河流
        selected_rivers = st.multiselect(
            "选择河流（可多选）",
            options=all_rivers,
            default=all_rivers[:5] if len(all_rivers) >= 5 else all_rivers,
            key="tab2_rivers"
        )
        
        # 预设河流组
        st.markdown("**快捷选择：**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        # 预设河流组定义 - 【修改点2】可以在这里添加更多预设河流组
        lancang_rivers = ['硕多岗河', '漾弓江', '龙川江', '白水河', '万马河']
        nanpan_rivers = ['南盘江', '牛栏江', '大汶溪', '关河', '螳螂川', '宁蒗河', '落漏河', '马过河', '五郎河']
        
        with preset_col1:
            if st.button("澜沧江水系", key="btn_lancang"):
                st.session_state['selected_rivers_tab2'] = [r for r in lancang_rivers if r in all_rivers]
                st.rerun()
        
        with preset_col2:
            if st.button("南盘江水系", key="btn_nanpan"):
                st.session_state['selected_rivers_tab2'] = [r for r in nanpan_rivers if r in all_rivers]
                st.rerun()
        
        with preset_col3:
            if st.button("全部河流", key="btn_all"):
                st.session_state['selected_rivers_tab2'] = all_rivers
                st.rerun()
        
        if len(selected_rivers) > 0:
            # 筛选选中河流的数据
            df_water_selected = df_water[df_water['river_name'].isin(selected_rivers)].copy()
            
            # 按日期求和
            df_water_sum = df_water_selected.groupby('date').agg({
                'water_level': 'sum'
            }).reset_index()
            
            # 合并数据
            df_merged = pd.merge(
                df_power_filtered[['date', 'power_sum']], 
                df_water_sum, 
                on='date', 
                how='inner'
            )
            df_merged = df_merged.dropna()
            
            if len(df_merged) >= 10:
                stats = calculate_correlation(
                    df_merged['water_level'].values,
                    df_merged['power_sum'].values
                )
                
                # 显示统计指标
                st.markdown(f"**已选择 {len(selected_rivers)} 条河流：** {', '.join(selected_rivers)}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pearson相关系数", f"{stats['pearson_r']:.4f}")
                with col2:
                    st.metric("R²决定系数", f"{stats['r2']:.4f}")
                with col3:
                    st.metric("样本量", f"{stats['n']}")
                with col4:
                    p_str = f"{stats['pearson_p']:.2e}" if stats['pearson_p'] < 0.001 else f"{stats['pearson_p']:.4f}"
                    st.metric("P值", p_str)
                
                # 绘制图表
                river_name = f"选中{len(selected_rivers)}条河流总和"
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_timeseries(df_merged, river_name, selected_dc, stats), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_scatter(df_merged, river_name, selected_dc, stats), use_container_width=True)
            else:
                st.warning(f"⚠️ 有效数据点不足10个（当前: {len(df_merged)}）")
        else:
            st.info("请选择至少一条河流")
    
    # ========== Tab 3: 所有河流汇总 ==========
    with tab3:
        st.header(f"所有河流水位总和与{selected_dc}发电量相关性")
        
        # 按日期汇总所有河流水位
        df_water_all = df_water.groupby('date').agg({
            'water_level': 'sum'
        }).reset_index()
        
        # 合并数据
        df_merged_all = pd.merge(
            df_power_filtered[['date', 'power_sum']], 
            df_water_all, 
            on='date', 
            how='inner'
        )
        df_merged_all = df_merged_all.dropna()
        
        if len(df_merged_all) >= 10:
            stats = calculate_correlation(
                df_merged_all['water_level'].values,
                df_merged_all['power_sum'].values
            )
            
            # 显示统计指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pearson相关系数", f"{stats['pearson_r']:.4f}")
            with col2:
                st.metric("R²决定系数", f"{stats['r2']:.4f}")
            with col3:
                st.metric("河流总数", f"{len(all_rivers)} 条")
            with col4:
                st.metric("样本量", f"{stats['n']}")
            
            # 绘制图表
            river_name = f"云南省所有河流（共{len(all_rivers)}条）"
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_timeseries(df_merged_all, river_name, selected_dc, stats), use_container_width=True)
            with col2:
                st.plotly_chart(plot_scatter(df_merged_all, river_name, selected_dc, stats), use_container_width=True)
    
    # ========== Tab 4: 数据总览 ==========
    with tab4:
        st.header(f"河流与{selected_dc}相关性排名")
        
        # 计算所有河流的相关性
        correlation_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, river in enumerate(all_rivers):
            status_text.text(f"正在计算: {river} ({idx+1}/{len(all_rivers)})")
            progress_bar.progress((idx + 1) / len(all_rivers))
            
            df_river_single = df_water[df_water['river_name'] == river].copy()
            df_river_daily = df_river_single.groupby('date').agg({
                'water_level': 'mean'
            }).reset_index()
            
            df_merged = pd.merge(
                df_power_filtered[['date', 'power_sum']], 
                df_river_daily, 
                on='date', 
                how='inner'
            )
            df_merged = df_merged.dropna()
            
            if len(df_merged) >= 10:
                stats = calculate_correlation(
                    df_merged['water_level'].values,
                    df_merged['power_sum'].values
                )
                if stats:
                    correlation_results.append({
                        '河流名称': river,
                        '样本量': stats['n'],
                        'Pearson_r': stats['pearson_r'],
                        'R²': stats['r2'],
                        'P值': stats['pearson_p'],
                        '回归斜率': stats['slope'],
                        '数据起始': df_merged['date'].min().strftime('%Y-%m-%d'),
                        '数据结束': df_merged['date'].max().strftime('%Y-%m-%d')
                    })
        
        progress_bar.empty()
        status_text.empty()
        
        if correlation_results:
            df_results = pd.DataFrame(correlation_results)
            df_results = df_results.sort_values('Pearson_r', key=abs, ascending=False).reset_index(drop=True)
            df_results.insert(0, '排名', range(1, len(df_results) + 1))
            
            # 格式化显示（去掉background_gradient避免需要matplotlib）
            st.dataframe(
                df_results.style.format({
                    'Pearson_r': '{:.4f}',
                    'R²': '{:.4f}',
                    'P值': '{:.2e}',
                    '回归斜率': '{:.4f}'
                }),
                use_container_width=True,
                height=600
            )
            
            # 下载按钮
            csv = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"📥 下载{selected_dc}相关性分析结果 (CSV)",
                data=csv,
                file_name=f"river_correlation_{selected_dc}.csv",
                mime="text/csv"
            )
        else:
            st.warning("没有足够的数据进行相关性分析")
    
    # ========== Tab 5: 多直流对比 ==========
    with tab5:
        st.header("多直流与河流水位相关性对比")
        st.info("💡 对比不同直流与选定河流水位的相关性")
        
        # 选择要对比的直流
        compare_dcs = st.multiselect(
            "选择要对比的直流",
            options=list(DC_CONFIG.keys()),
            default=list(DC_CONFIG.keys()),
            key="compare_dcs"
        )
        
        # 选择河流
        compare_rivers = st.multiselect(
            "选择河流（可多选）",
            options=all_rivers,
            default=all_rivers[:10] if len(all_rivers) >= 10 else all_rivers,
            key="compare_rivers"
        )
        
        if len(compare_dcs) > 0 and len(compare_rivers) > 0:
            # 加载所有选中直流的数据
            with st.spinner("正在加载各直流数据..."):
                all_power_data = {}
                for dc_name in compare_dcs:
                    df = load_power_data(dc_name)
                    if df is not None:
                        all_power_data[dc_name] = df
            
            # 处理选中河流的水位数据
            df_water_selected = df_water[df_water['river_name'].isin(compare_rivers)].copy()
            df_water_sum = df_water_selected.groupby('date').agg({'water_level': 'sum'}).reset_index()
            
            # 计算各直流的相关性
            comparison_results = []
            
            for dc_name, df_power_dc in all_power_data.items():
                # 筛选时间范围
                df_power_filtered_dc = df_power_dc[
                    (df_power_dc['date'] >= str(start_date)) & 
                    (df_power_dc['date'] <= str(end_date))
                ].copy()
                
                # 合并数据
                df_merged = pd.merge(df_power_filtered_dc, df_water_sum, on='date', how='inner')
                df_merged = df_merged.dropna()
                
                if len(df_merged) >= 10:
                    stats = calculate_correlation(
                        df_merged['water_level'].values,
                        df_merged['power_sum'].values
                    )
                    if stats:
                        comparison_results.append({
                            '直流名称': dc_name,
                            'Pearson_r': stats['pearson_r'],
                            'R²': stats['r2'],
                            'P值': stats['pearson_p'],
                            '样本量': stats['n']
                        })
            
            if comparison_results:
                # 显示对比表格
                df_comparison = pd.DataFrame(comparison_results)
                df_comparison = df_comparison.sort_values('Pearson_r', key=abs, ascending=False)
                
                st.subheader("各直流相关性对比")
                st.dataframe(
                    df_comparison.style.format({
                        'Pearson_r': '{:.4f}',
                        'R²': '{:.4f}',
                        'P值': '{:.2e}'
                    }),
                    use_container_width=True
                )
                
                # 绘制对比柱状图
                fig_bar = go.Figure()
                colors = [DC_CONFIG[dc]['color'] for dc in df_comparison['直流名称']]
                
                fig_bar.add_trace(go.Bar(
                    x=df_comparison['直流名称'],
                    y=df_comparison['Pearson_r'],
                    marker_color=colors,
                    text=df_comparison['Pearson_r'].apply(lambda x: f'{x:.4f}'),
                    textposition='outside'
                ))
                
                fig_bar.update_layout(
                    title='各直流与河流水位Pearson相关系数对比',
                    xaxis_title='直流名称',
                    yaxis_title='Pearson相关系数',
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 绘制散点对比图
                st.subheader("各直流散点分布对比")
                fig_scatter = go.Figure()
                
                for dc_name, df_power_dc in all_power_data.items():
                    df_power_filtered_dc = df_power_dc[
                        (df_power_dc['date'] >= str(start_date)) & 
                        (df_power_dc['date'] <= str(end_date))
                    ].copy()
                    
                    df_merged = pd.merge(df_power_filtered_dc, df_water_sum, on='date', how='inner')
                    df_merged = df_merged.dropna()
                    
                    if len(df_merged) > 0:
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=df_merged['water_level'],
                                y=df_merged['power_sum'],
                                mode='markers',
                                name=dc_name,
                                marker=dict(
                                    color=DC_CONFIG[dc_name]['color'],
                                    size=5,
                                    opacity=0.5
                                )
                            )
                        )
                
                fig_scatter.update_layout(
                    title=f'各直流与选定{len(compare_rivers)}条河流水位散点分布',
                    xaxis_title='河流水位总和(m)',
                    yaxis_title='日发电量(MWh)',
                    height=500
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("没有足够的数据进行对比分析")
        else:
            st.info("请选择至少一个直流和一条河流")

# ============================================================================
# 运行应用
# ============================================================================
if __name__ == "__main__":
    main()