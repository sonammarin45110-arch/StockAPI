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

    end_dt   = pd.to_datetime(end_date) if end_date else df['Date'].max()
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
# 2️⃣ คำนวณ FSN Classification
#
#  ✅ หลักการ: ใช้ Coefficient of Variation (CV) เป็นเกณฑ์หลัก
#              อ้างอิง: Tersine (1994), Gopalakrishnan & Sundaresan (1994)
#
#  CV = σ / μ  (ความผันผวนสัมพัทธ์ของยอดขาย)
#
#  ถ้าทุก SKU เคลื่อนไหว (ไม่มี Non-moving จริงๆ) → ใช้แค่ F/S:
#    F (Fast-moving)  : CV ต่ำ  → ขายสม่ำเสมอ คาดการณ์ง่าย
#    S (Slow-moving)  : CV สูง  → ขายไม่สม่ำเสมอ ความผันผวนสูง
#
#  threshold CV = median ของ dataset (data-driven, ไม่ใช่ hardcode)
#
#  ถ้ามี SKU ที่ไม่มีการเคลื่อนไหวเลย (mean=0) → จัด N โดยอัตโนมัติ
# ==========================================================
def fsn_from_sales_90(sales_90: pd.DataFrame, lambda_decay: float = 0.08) -> pd.DataFrame:
    df = sales_90.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['SKU', 'Date'])

    last_date = df['Date'].max()
    df['Days_Ago']       = (last_date - df['Date']).dt.days
    df['Weight']         = np.exp(-lambda_decay * df['Days_Ago'])
    df['Weighted_Sales'] = df['Sales_Qty'] * df['Weight']

    agg = df.groupby('SKU', as_index=False).agg(
        Days_Sold_90       = ('Sales_Qty', lambda x: (x > 0).sum()),
        Total_Sales_90days = ('Sales_Qty', 'sum'),
        Weighted_Sum       = ('Weighted_Sales', 'sum'),
        Weight_Total       = ('Weight', 'sum'),
        Std_Daily_Demand   = ('Sales_Qty', 'std'),
        Mean_Raw           = ('Sales_Qty', 'mean'),
    )

    agg['Avg_Daily_Demand'] = agg['Weighted_Sum'] / agg['Weight_Total']
    agg['Turnover_Rate']    = (agg['Days_Sold_90'] / 90) * 100
    agg['Std_Daily_Demand'] = agg['Std_Daily_Demand'].fillna(0)

    # ✅ CV = σ / μ (ใช้ mean raw สำหรับ CV ไม่ใช่ weighted)
    agg['CV'] = agg.apply(
        lambda r: r['Std_Daily_Demand'] / r['Mean_Raw'] if r['Mean_Raw'] > 0 else 999,
        axis=1
    )

    # threshold = median CV ของ dataset (data-driven)
    # ถ้าทุก SKU active → F/S เท่านั้น
    # ถ้ามี SKU ที่ Days_Sold_90 = 0 → N
    active_mask = agg['Days_Sold_90'] > 0
    cv_median   = agg.loc[active_mask, 'CV'].median()

    def classify(row):
        if row['Days_Sold_90'] == 0:
            return 'N'                    # Non-moving จริงๆ
        elif row['CV'] <= cv_median:
            return 'F'                    # CV ต่ำ = สม่ำเสมอ
        else:
            return 'S'                    # CV สูง = ผันผวน

    agg['FSN_Class'] = agg.apply(classify, axis=1)

    return agg[['SKU', 'FSN_Class', 'CV', 'Days_Sold_90', 'Turnover_Rate',
                'Avg_Daily_Demand', 'Std_Daily_Demand', 'Total_Sales_90days']]


# ==========================================================
# 3️⃣ คำนวณพื้นที่คลังคงเหลือ
# ==========================================================
def calc_warehouse_available(sku_df: pd.DataFrame,
                              max_capacity: int = WAREHOUSE_MAX_CAPACITY) -> int:
    if "OnHand" not in sku_df.columns:
        return max_capacity
    used      = sku_df["OnHand"].clip(lower=0).sum()
    available = int(max_capacity - used)
    return max(0, available)


# ==========================================================
# 4️⃣ คำนวณ Safety Stock รายตัว
# ==========================================================
def calc_safety_stock(std_demand: float, lead_time: int, z: float) -> int:
    """
    Safety Stock = Z × σ_demand × √Lead_Time
    Ref: Silver, Pyke & Peterson (1998) - Inventory Management and
         Production Planning and Scheduling, 3rd ed.
    """
    if std_demand <= 0 or lead_time <= 0:
        return 0
    return math.ceil(z * std_demand * math.sqrt(lead_time))


# ==========================================================
# 5️⃣ คำนวณคำแนะนำราย SKU
# ==========================================================
def recommend_row(row: dict, bot_cfg: dict, warehouse_available: int) -> dict:
    on_hand   = int(row.get('OnHand', 0))
    avg       = float(row.get('Avg_Daily_Demand', 0) or 0)
    std       = float(row.get('Std_Daily_Demand', 0) or 0)
    fsn       = str(row.get('FSN_Class', 'S') or 'S')
    moq       = int(row.get('MOQ', 0) or 0)
    lead_time = int(row.get('Lead_Time', 7) or 7)

    cfg    = bot_cfg.get(fsn, bot_cfg['S'])
    target = cfg['Target_Cover_Days']
    z      = cfg['Z']

    ss        = calc_safety_stock(std, lead_time, z)
    backorder = abs(on_hand) if on_hand < 0 else 0

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

    raw_calc       = max(0, int(raw_calc))
    after_capacity = min(raw_calc, int(warehouse_available))

    if moq > 0 and after_capacity < moq:
        final_qty = 0
    else:
        final_qty = after_capacity

    reason_parts = [base_reason]
    if raw_calc != after_capacity:
        reason_parts.append(f"จำกัดด้วยความจุ {warehouse_available}")
    if moq > 0 and after_capacity < moq:
        decision = "ปฏิเสธ"
        priority = "ต่ำ"
        reason_parts.append(f"ต่ำกว่า MOQ {moq}")
    if raw_calc > 0 and final_qty == 0 and warehouse_available <= 0:
        decision = "รออนุมัติ"
        priority = "ต่ำ"
        reason_parts.append("พื้นที่คลังไม่เพียงพอ")

    reason_parts.append(f"รับเข้า {final_qty}")

    return {
        "SKU"              : row["SKU"],
        "FSN_Class"        : fsn,
        "CV"               : round(row.get('CV', 0), 4),
        "OnHand"           : on_hand,
        "Avg_Daily_Demand" : round(avg, 4),
        "Std_Daily_Demand" : round(std, 4),
        "Safety_Stock"     : ss,
        "Lead_Time"        : lead_time,
        "ความต้องการจริง"  : raw_calc,
        "MOQ"              : moq,
        "Recommended_Qty"  : final_qty,
        "Decision"         : decision,
        "Priority"         : priority,
        "Reason"           : " | ".join(reason_parts),
    }


# ==========================================================
# 6️⃣ Bulk Recommendation (FSN Priority F → S → N)
# ==========================================================
def bulk_recommend_dynamic_fsn_priority(fsn_df: pd.DataFrame,
                                         sku_df: pd.DataFrame,
                                         warehouse_available: int):
    merge_cols = [c for c in ["SKU", "OnHand", "MOQ", "Lead_Time"] if c in sku_df.columns]
    base = fsn_df.merge(sku_df[merge_cols], on="SKU", how="left")
    base["MOQ"]       = base["MOQ"].fillna(0).astype(int)
    base["Lead_Time"] = base["Lead_Time"].fillna(7).astype(int)

    order = {"F": 1, "S": 2, "N": 3}
    base["Priority_Sort"] = base["FSN_Class"].map(order)
    base = base.sort_values(["Priority_Sort", "Avg_Daily_Demand"],
                             ascending=[True, False])

    available = int(warehouse_available)
    results   = []
    for _, r in base.iterrows():
        res       = recommend_row(r.to_dict(), BOT_CONFIG, available)
        available = max(0, available - int(res["Recommended_Qty"]))
        results.append(res)

    return pd.DataFrame(results), available


# ==========================================================
# 7️⃣ End-to-End Pipeline
# ==========================================================
def end_to_end_dynamic_from_excel(
        xls_bytes_or_path,
        sheet_sku           = "SKU_Master",
        sheet_out           = "STOCK_OUT",
        end_date            = None,
        warehouse_available = None,
        max_capacity        = WAREHOUSE_MAX_CAPACITY,
        decay_rate          = 0.08):

    xls       = pd.ExcelFile(xls_bytes_or_path)
    sku       = pd.read_excel(xls, sheet_sku)
    stock_out = pd.read_excel(xls, sheet_out)
    stock_out['Date'] = pd.to_datetime(stock_out['Date'], errors='coerce')

    if not end_date:
        end_date = stock_out['Date'].max().strftime('%Y-%m-%d')

    sales90 = build_sales_history_90(stock_out, end_date)
    fsn     = fsn_from_sales_90(sales90, lambda_decay=decay_rate)

    for col, default in [("OnHand", 0), ("MOQ", 0), ("Lead_Time", 7)]:
        if col not in sku.columns:
            sku[col] = default

    initial_available = (
        calc_warehouse_available(sku, max_capacity)
        if warehouse_available is None else int(warehouse_available)
    )

    rec_df, remain = bulk_recommend_dynamic_fsn_priority(fsn, sku, initial_available)

    final = (
        rec_df
        .sort_values(["Decision", "FSN_Class", "Avg_Daily_Demand"],
                     ascending=[True, True, False])
        .reset_index(drop=True)
    )

    return {
        "Sales_History_90days"        : sales90,
        "FSN_Classification"          : fsn,
        "Recommendation"              : final,
        "Warehouse_Initial_Available" : int(initial_available),
        "Warehouse_Remaining"         : int(remain),
        "End_Date"                    : end_date,
        "Decay_Rate"                  : decay_rate,
    }


# ==========================================================
# 8️⃣ Quick Test
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FSN Classification — CV-based (Tersine 1994)")
    print("=" * 60)
    print("""
  หลักการ: Coefficient of Variation (CV = σ/μ)
  ─────────────────────────────────────────────
  F (Fast-moving)  : CV ≤ median(CV)  → ขายสม่ำเสมอ ผันผวนต่ำ
  S (Slow-moving)  : CV > median(CV)  → ขายผันผวน คาดการณ์ยาก
  N (Non-moving)   : Days_Sold = 0    → ไม่มีการเคลื่อนไหวจริงๆ

  อ้างอิง:
  - Tersine, R.J. (1994). Principles of Inventory and
    Materials Management, 4th ed. Prentice Hall.
  - Gopalakrishnan, P. & Sundaresan, M. (1994).
    Materials Management: An Integrated Approach.
  - Silver, Pyke & Peterson (1998). Inventory Management
    and Production Planning and Scheduling, 3rd ed.
    (Safety Stock formula: SS = Z × σ × √LT)
    """)

    print("Safety Stock Verification:")
    print("-" * 60)
    for fsn_class, cfg in BOT_CONFIG.items():
        ss = calc_safety_stock(std_demand=5.0, lead_time=7, z=cfg['Z'])
        print(f"  {fsn_class} | Z={cfg['Z']} | σ=5.0/day | LT=7d → SS={ss} units")