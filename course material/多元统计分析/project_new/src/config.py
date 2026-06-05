from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT / "project_new"
RAW_DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
LOG_DIR = OUTPUT_DIR / "logs"

for path in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)


SELECTED_STRATEGIES = [
    {
        "display_name": "煤炭周期优选动态轮动策略",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录煤炭",
        "summary_name": "煤炭周期优选动态轮动策略",
        "theme": "周期轮动",
    },
    {
        "display_name": "沪深300增强策略",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录300",
        "summary_name": "沪深300增强策略",
        "theme": "大盘蓝筹",
    },
    {
        "display_name": "中证800增强",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录800",
        "summary_name": "中证800增强",
        "theme": "大盘均衡",
    },
    {
        "display_name": "中证2000增强",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录中证2000",
        "summary_name": "中证2000增强",
        "theme": "中小盘",
    },
    {
        "display_name": "中证1000增强",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录1000",
        "summary_name": "中证1000增强",
        "theme": "中小盘成长",
    },
    {
        "display_name": "半导体优选策略",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录半导体",
        "summary_name": "半导体优选策略",
        "theme": "科技行业",
    },
    {
        "display_name": "成长红利量化选股",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录红利",
        "summary_name": "成长红利量化选股",
        "theme": "成长红利",
    },
    {
        "display_name": "科创参数加强板",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录科创",
        "summary_name": "科创参数加强板",
        "theme": "科创成长",
    },
    {
        "display_name": "双创50增强",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录双创",
        "summary_name": "双创50增强",
        "theme": "双创成长",
    },
    {
        "display_name": "医疗ETF增强",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "医疗etf增强",
        "summary_name": "医疗etf增强",
        "theme": "医药行业",
    },
    {
        "display_name": "通信ETF增强",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录通信",
        "summary_name": "通信etf增强",
        "theme": "通信TMT",
    },
    {
        "display_name": "食品ETF增强",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录食品",
        "summary_name": "食品etf增强",
        "theme": "消费防御",
    },
    {
        "display_name": "军工ETF增强",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录军工",
        "summary_name": "军工etf增强",
        "theme": "军工行业",
    },
    {
        "display_name": "旅游ETF增强",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录旅游",
        "summary_name": "旅游etf增强",
        "theme": "消费主题",
    },
    {
        "display_name": "动量趋势策略",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录动量趋势",
        "summary_name": "动量趋势策略",
        "theme": "动量趋势",
    },
    {
        "display_name": "均衡持仓",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录均衡持仓",
        "summary_name": "均衡持仓",
        "theme": "均衡配置",
    },
    {
        "display_name": "杠铃",
        "workbook": "量化策略绩效-2.xlsx",
        "sheet_name": "交易记录杠铃",
        "summary_name": "杠铃",
        "theme": "杠铃配置",
    },
    {
        "display_name": "形态识别",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录形态",
        "summary_name": "形态识别",
        "theme": "技术择时",
    },
    {
        "display_name": "计算机ETF优选策略",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录计算机",
        "summary_name": "计算机ETF优选策略",
        "theme": "计算机行业",
    },
    {
        "display_name": "化工ETF优选策略",
        "workbook": "量化策略绩效-1.xlsx",
        "sheet_name": "交易记录化工",
        "summary_name": "化工ETF优选策略",
        "theme": "化工行业",
    },
]


CLIENT_FILES = [
    {"client_name": "客户A", "workbook": "模拟账户A的记录.xlsx", "initial_capital": 420000},
    {"client_name": "客户B", "workbook": "模拟账户B的记录.xlsx", "initial_capital": 1000000},
    {"client_name": "客户C", "workbook": "模拟账户C的记录.xlsx", "initial_capital": 400000},
]


STYLE_FEATURES = [
    "log_trade_freq",
    "avg_hold_days",
    "turnover_rate",
    "log_avg_trade_amount",
    "buy_sell_ratio",
    "trade_regularity",
    "num_stocks",
    "concentration_top3",
    "concentration_top1",
    "avg_position_pct",
    "position_peak_ratio",
    "stock_turnover",
]

BEHAVIOR_FEATURES = [
    "log_trade_freq",
    "avg_hold_days",
    "turnover_rate",
    "log_avg_trade_amount",
]

HOLDING_FEATURES = [
    "buy_sell_ratio",
    "trade_regularity",
    "num_stocks",
    "concentration_top3",
    "concentration_top1",
    "avg_position_pct",
    "position_peak_ratio",
    "stock_turnover",
]

PERFORMANCE_FEATURES = [
    "avg_return_pct",
    "win_rate",
    "profit_loss_ratio",
    "max_drawdown",
    "return_volatility",
    "sharpe_approx",
]

ALL_FEATURES = STYLE_FEATURES + PERFORMANCE_FEATURES
