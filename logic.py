import numpy as np
import pandas as pd
import math
from datetime import timedelta

# ==========================================================
# CONFIG
# ==========================================================
BOT_CONFIG = {
    "F": {"Min_Cover_Days": 7,  "Target_Cover_Days": 14, "Max_Cover_Days": 21, "Z": 1.65},
    "S": {"Min_Cover_Days": 14, "Target_Cover_Days": 30, "Max_Cover_Days": 45, "Z": 1.28},
    "N": {"Min_Cover_Days": 30, "Target_Cover_Days": 60, "Max_Cover_Days": 90, "Z": 1.04},
}

WAREHOUSE_MAX_CAPACITY = 30000


# ==========================================================
# 1️⃣ สร้างยอดขายย้อนหลัง 90 วัน
# ==========================================================
def build_sales_history_90(stock_out: pd.DataFrame, end_date=None) -> pd.DataFrame:
    df = stock_out.rename(columns={'Qty': 'Sales_Qty'})
    df = df[['Date', 'SKU', 'Sales_Qty']].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'SKU'])
    df['SKU'] = df['SKU'].astype(str)

    end_dt = pd.to_datetime(end_date) if end_date else df['Date'].max()
    start_dt = end_dt - timedelta(days=89)
    df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]

    all_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    skus = df['SKU'].unique()
    full = pd.MultiIndex.from_product([all_dates, skus], names=['Date', 'SKU'])

    out = (
        df.set_index(['Date', 'SKU'])
        .reindex(full, fill_value=0)
        .reset_index()
    )
    out.columns = ['Date', 'SKU', 'Sales_Qty']
    return out


# ==========================================================
# 2️⃣ คำนวณ FSN + Safety Stock Parameters
#    ✅ เพิ่ม Std_Daily_Demand สำหรับคำนวณ Safety Stock
# ==========================================================
def fsn_from_sales_90(sales_90: pd.DataFrame, lambda_decay: float = 0.08) -> pd.DataFrame:
    df = sales_90.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['SKU', 'Date'])

    last_date = df['Date'].max()
    df['Days_Ago'] = (last_date - df['Date']).dt.days
    df['Weight'] = np.exp(-lambda_decay * df['Days_Ago'])
    df['Weighted_Sales'] = df['Sales_Qty'] * df['Weight']

    agg = df.groupby('SKU', as_index=False).agg(
        Days_Sold_90=('Sales_Qty', lambda x: (x > 0).sum()),
        Total_Sales_90days=('Sales_Qty', 'sum'),
        Weighted_Sum=('Weighted_Sales', 'sum'),
        Weight_Total=('Weight', 'sum'),
        # ✅ เพิ่ม Std ของยอดขายรายวัน (รวม 0) สำหรับ Safety Stock
        Std_Daily_Demand=('Sales_Qty', 'std'),
    )

    agg['Avg_Daily_Demand'] = agg['Weighted_Sum'] / agg['Weight_Total']
    agg['Turnover_Rate'] = (agg['Days_Sold_90'] / 90) * 100
    # เติม NaN ที่อาจเกิดจาก std ของ series ค่าเดียว
    agg['Std_Daily_Demand'] = agg['Std_Daily_Demand'].fillna(0)

    def classify(avg):
        if avg >= 10:
            return "F"
        elif avg >= 3:
            return "S"
        else:
            return "N"

    agg['FSN_Class'] = agg['Avg_Daily_Demand'].apply(classify)

    return agg[['SKU', 'FSN_Class', 'Days_Sold_90', 'Turnover_Rate',
                'Avg_Daily_Demand', 'Std_Daily_Demand', 'Total_Sales_90days']]


# ==========================================================
# 3️⃣ คำนวณพื้นที่คลังคงเหลือ
# ==========================================================
def calc_warehouse_available(sku_df: pd.DataFrame, max_capacity: int = WAREHOUSE_MAX_CAPACITY) -> int:
    if "OnHand" not in sku_df.columns:
        return max_capacity
    used = sku_df["OnHand"].clip(lower=0).sum()
    available = int(max_capacity - used)
    return max(0, available)


# ==========================================================
# 4️⃣ คำนวณ Safety Stock รายตัว
#    ✅ ใหม่ทั้งหมด
# ==========================================================
def calc_safety_stock(std_demand: float, lead_time: int, z: float) -> int:
    """
    Safety Stock = Z × σ_demand × √Lead_Time
    
    Args:
        std_demand : ส่วนเบี่ยงเบนมาตรฐานของยอดขายรายวัน (จาก 90 วัน)
        lead_time  : Lead Time (วัน) จาก SKU_Master
        z          : Z-score ตาม Service Level ของ FSN Class
                     F=1.65 (95%), S=1.28 (90%), N=1.04 (85%)
    
    Returns:
        Safety Stock (หน่วย, ปัดขึ้น)
    """
    if std_demand <= 0 or lead_time <= 0:
        return 0
    ss = z * std_demand * math.sqrt(lead_time)
    return math.ceil(ss)


# ==========================================================
# 5️⃣ คำนวณคำแนะนำราย SKU (เพิ่ม Safety Stock)
# ==========================================================
def recommend_row(row: dict, bot_cfg: dict, warehouse_available: int) -> dict:
    on_hand   = int(row.get('OnHand', 0))
    avg       = float(row.get('Avg_Daily_Demand', 0) or 0)
    std       = float(row.get('Std_Daily_Demand', 0) or 0)
    fsn       = str(row.get('FSN_Class', 'N') or 'N')
    moq       = int(row.get('MOQ', 0) or 0)
    lead_time = int(row.get('Lead_Time', 7) or 7)

    cfg    = bot_cfg[fsn]
    target = cfg['Target_Cover_Days']
    z      = cfg['Z']

    # ✅ คำนวณ Safety Stock
    ss = calc_safety_stock(std, lead_time, z)

    backorder = abs(on_hand) if on_hand < 0 else 0

    # =============================
    # 1️⃣ Raw Calculation (+ Safety Stock)
    # =============================
    if backorder > 0:
        raw_calc = backorder + (target * avg) + ss
        decision, priority = "อนุมัติเร่งด่วน", "สูง"
        base_reason = f"Backorder {backorder} + Demand {int(target*avg)} + SS {ss}"
    else:
        cover = (on_hand / avg) if avg > 0 else 999

        if cover < cfg['Min_Cover_Days']:
            raw_calc = (target * avg) + ss - on_hand
            decision, priority = "อนุมัติ", "ปานกลาง"
            base_reason = f"Cover {cover:.1f} < {cfg['Min_Cover_Days']} + SS {ss}"
        elif cover > cfg['Max_Cover_Days']:
            raw_calc = 0
            decision, priority = "ปฏิเสธ", "ต่ำ"
            base_reason = f"Cover {cover:.1f} > {cfg['Max_Cover_Days']}"
        else:
            raw_calc = 0
            decision, priority = "ปฏิเสธ", "ต่ำ"
            base_reason = f"เพียงพอ (Cover {cover:.1f})"

    raw_calc = max(0, int(raw_calc))

    # =============================
    # 2️⃣ Capacity Adjustment
    # =============================
    after_capacity = min(raw_calc, int(warehouse_available))

    # =============================
    # 3️⃣ MOQ Check (แทน LOT)
    # =============================

    if moq > 0 and after_capacity < moq:
        final_qty = 0
    else:
        final_qty = after_capacity

    # =============================
    # 4️⃣ Reason Builder
    # =============================
    reason_parts = [base_reason]

    if raw_calc != after_capacity:
        reason_parts.append(f"จำกัดด้วยความจุ {warehouse_available}")

    if moq > 0 and after_capacity < moq:
        decision = "ปฏิเสธ"
        priority = "ต่ำ"
        reason_parts.append(f"ต่ำกว่า MOQ {moq}")

    if raw_calc > 0 and final_qty == 0 and warehouse_available <= 0:
        decision  = "รออนุมัติ"
        priority  = "ต่ำ"
        reason_parts.append("พื้นที่คลังไม่เพียงพอ")

    reason_parts.append(f"รับเข้า {final_qty}")
    reason = " | ".join(reason_parts)

    return {
        "SKU"              : row["SKU"],
        "FSN_Class"        : fsn,
        "OnHand"           : on_hand,
        "Avg_Daily_Demand" : round(avg, 4),
        "Std_Daily_Demand" : round(std, 4),
        "Safety_Stock"     : ss,           # ✅ ใหม่
        "Lead_Time"        : lead_time,    # ✅ ใหม่
        "ความต้องการจริง"  : raw_calc,
        "MOQ"              : moq,
        "Recommended_Qty"  : final_qty,
        "Decision"         : decision,
        "Priority"         : priority,
        "Reason"           : reason,
    }


# ==========================================================
# 6️⃣ Bulk Recommendation (FSN Priority F → S → N)
# ==========================================================
def bulk_recommend_dynamic_fsn_priority(fsn_df: pd.DataFrame,
                                         sku_df: pd.DataFrame,
                                         warehouse_available: int):
    # ✅ เพิ่ม Lead_Time จาก SKU_Master
    merge_cols = ["SKU", "OnHand", "MOQ", "Lead_Time"]
    merge_cols = [c for c in merge_cols if c in sku_df.columns]

    base = fsn_df.merge(sku_df[merge_cols], on="SKU", how="left")
    base["MOQ"]  = base["MOQ"].fillna(0).astype(int)
    base["Lead_Time"] = base["Lead_Time"].fillna(7).astype(int)

    order = {"F": 1, "S": 2, "N": 3}
    base["Priority_Sort"] = base["FSN_Class"].map(order)
    base = base.sort_values(["Priority_Sort", "Avg_Daily_Demand"],
                             ascending=[True, False])

    available = int(warehouse_available)
    results   = []

    for _, r in base.iterrows():
        res  = recommend_row(r.to_dict(), BOT_CONFIG, available)
        used = int(res["Recommended_Qty"])
        available -= used
        if available < 0:
            available = 0
        results.append(res)

    return pd.DataFrame(results), available

def end_to_end_dynamic_from_excel(
        xls_bytes_or_path,
        sheet_sku="SKU_Master",
        sheet_out="STOCK_OUT",
        end_date=None,
        warehouse_available=None,
        max_capacity=WAREHOUSE_MAX_CAPACITY,
        decay_rate=0.08):

    xls = pd.ExcelFile(xls_bytes_or_path)
    sku = pd.read_excel(xls, sheet_sku)
    stock_out = pd.read_excel(xls, sheet_out)

    stock_out['Date'] = pd.to_datetime(stock_out['Date'], errors='coerce')

    if not end_date:
        end_date = stock_out['Date'].max().strftime('%Y-%m-%d')

    sales90 = build_sales_history_90(stock_out, end_date)
    fsn = fsn_from_sales_90(sales90, lambda_decay=decay_rate)

    if "OnHand" not in sku.columns:
        sku["OnHand"] = 0
    if "MOQ" not in sku.columns:
        sku["MOQ"] = 0
    if "Lead_Time" not in sku.columns:
        sku["Lead_Time"] = 7

    initial_available = (
        calc_warehouse_available(sku, max_capacity)
        if warehouse_available is None else int(warehouse_available)
    )

    rec_df, remain = bulk_recommend_dynamic_fsn_priority(
        fsn, sku, initial_available
    )

    final = (
        rec_df
        .sort_values(
            ["Decision", "FSN_Class", "Avg_Daily_Demand"],
            ascending=[True, True, False]
        )
        .reset_index(drop=True)
    )

    return {
        "Sales_History_90days": sales90,
        "FSN_Classification": fsn,
        "Recommendation": final,
        "Warehouse_Initial_Available": int(initial_available),
        "Warehouse_Remaining": int(remain),
        "End_Date": end_date,
        "Decay_Rate": decay_rate
    }

# ==========================================================
# 7️⃣ Quick Test สำหรับ verify ว่า Safety Stock ทำงานถูกต้อง
# ==========================================================
if __name__ == "__main__":
    print("=" * 55)
    print("Safety Stock Verification")
    print("=" * 55)

    test_cases = [
        {"FSN": "F", "Z": 1.65, "std": 15.0, "lead_time": 7},
        {"FSN": "S", "Z": 1.28, "std": 8.0,  "lead_time": 11},
        {"FSN": "N", "Z": 1.04, "std": 3.0,  "lead_time": 14},
    ]

    for t in test_cases:
        ss = calc_safety_stock(t["std"], t["lead_time"], t["Z"])
        print(f"  Class {t['FSN']} | Z={t['Z']} | σ={t['std']} | LT={t['lead_time']}d "
              f"→ Safety Stock = {ss} units")

    print()
    print("Formula: SS = Z × σ_demand × √Lead_Time")
    print()
    print("Service Level by FSN Class:")
    for fsn, cfg in BOT_CONFIG.items():
        print(f"  {fsn}: Z={cfg['Z']} → ~{int(cfg['Z']*100/1.65*95 if fsn=='F' else (cfg['Z']/1.28*90 if fsn=='S' else cfg['Z']/1.04*85))}% service level")