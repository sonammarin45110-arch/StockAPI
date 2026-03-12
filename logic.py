import numpy as np
import pandas as pd
import math
from datetime import timedelta

# ==========================================================
# CONFIG
# ==========================================================
BOT_CONFIG = {
    "F": {"Min_Cover_Days": 7,  "Target_Cover_Days": 30, "Max_Cover_Days": 45, "Z": 1.65},
    "S": {"Min_Cover_Days": 14, "Target_Cover_Days": 30, "Max_Cover_Days": 45, "Z": 1.28},
    "N": {"Min_Cover_Days": 30, "Target_Cover_Days": 60, "Max_Cover_Days": 90, "Z": 1.04},
}

WAREHOUSE_MAX_CAPACITY = 30000


# ==========================================================
# 1️⃣ สร้างยอดขายย้อนหลัง 90 วัน
# ==========================================================
def build_sales_history_monthly(stock_out: pd.DataFrame,
                                 end_date: str = None,
                                 lookback_months: int = 6) -> pd.DataFrame:
    """
    รวบรวมยอดขายย้อนหลัง N เดือน (default=6) แบบรายเดือน
    เหมาะกับข้อมูลที่บันทึก event-based (ไม่ใช่รายวันสมบูรณ์)

    Ref: Syntetos & Boylan (2005) - "The accuracy of intermittent
         demand estimates", International Journal of Forecasting
         → แนะนำ lookback ≥ 6 periods สำหรับ demand estimation
    """
    df = stock_out.rename(columns={"Qty": "Sales_Qty"}).copy()
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["SKU"]   = df["SKU"].astype(str)
    df = df.dropna(subset=["Date", "SKU"])

    # แปลง Date → Month key (YYYYMM)
    df["MonthKey"] = df["Date"].dt.to_period("M")

    if end_date:
        end_period = pd.Period(end_date, "M")
    else:
        end_period = df["MonthKey"].max()

    start_period = end_period - (lookback_months - 1)
    df = df[(df["MonthKey"] >= start_period) & (df["MonthKey"] <= end_period)]

    # aggregate รายเดือน
    monthly = (
        df.groupby(["MonthKey", "SKU"], as_index=False)["Sales_Qty"]
        .sum()
    )

    # reindex ให้ครบทุก SKU ทุกเดือน (fill 0 ถ้าไม่มียอด)
    all_periods = pd.period_range(start=start_period, end=end_period, freq="M")
    skus        = df["SKU"].unique()
    full_idx    = pd.MultiIndex.from_product([all_periods, skus],
                                              names=["MonthKey", "SKU"])
    monthly = (
        monthly.set_index(["MonthKey", "SKU"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    # เพิ่ม Months_Ago สำหรับ exponential decay
    monthly["Months_Ago"] = (end_period - monthly["MonthKey"]).apply(lambda x: x.n)
    return monthly


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
def fsn_from_sales_monthly(sales_monthly: pd.DataFrame,
                            lambda_decay: float = 0.15) -> pd.DataFrame:
    """
    คำนวณ FSN + Demand Parameters จากข้อมูลรายเดือน

    Weighted Avg: decay รายเดือน (lambda=0.15 → เดือนก่อนมีน้ำหนัก ~86%)
    FSN: CV-based (Tersine 1994)
         F = CV ≤ median → ขายสม่ำเสมอ
         S = CV > median → ขายผันผวน
         N = ไม่มีการเคลื่อนไหวเลย (Months_Active = 0)

    lambda_decay=0.15 สำหรับรายเดือน ≈ lambda=0.05 รายวัน
    Ref: Gardner (1985) Exponential Smoothing, J. of Forecasting
    """
    df = sales_monthly.copy()
    df["Weight"]         = np.exp(-lambda_decay * df["Months_Ago"])
    df["Weighted_Sales"] = df["Sales_Qty"] * df["Weight"]

    agg = df.groupby("SKU", as_index=False).agg(
        Months_Active      = ("Sales_Qty", lambda x: (x > 0).sum()),
        Total_Sales        = ("Sales_Qty", "sum"),
        Weighted_Sum       = ("Weighted_Sales", "sum"),
        Weight_Total       = ("Weight", "sum"),
        Std_Monthly_Demand = ("Sales_Qty", "std"),
        Mean_Monthly_Raw   = ("Sales_Qty", "mean"),
        Lookback_Months    = ("Sales_Qty", "count"),
    )

    # Weighted avg monthly demand
    agg["Avg_Monthly_Demand"] = agg["Weighted_Sum"] / agg["Weight_Total"]
    # แปลงเป็น daily (หาร 30)
    agg["Avg_Daily_Demand"]   = agg["Avg_Monthly_Demand"] / 30
    agg["Std_Daily_Demand"] = (agg["Std_Monthly_Demand"].fillna(0)) / math.sqrt(30)

    # CV = σ_monthly / μ_monthly
    agg["CV"] = agg.apply(
        lambda r: r["Std_Monthly_Demand"] / r["Mean_Monthly_Raw"]
                  if r["Mean_Monthly_Raw"] > 0 else 999,
        axis=1
    )

    # Turnover Rate = % เดือนที่มีการเคลื่อนไหว
    agg["Turnover_Rate"] = (agg["Months_Active"] / agg["Lookback_Months"]) * 100

    # FSN Classification (CV-based, data-driven threshold)
    active_mask = agg["Months_Active"] > 0
    cv_median   = agg.loc[active_mask, "CV"].median() if active_mask.any() else 1.0

    def classify(row):
        if row["Months_Active"] == 0:
            return "N"
        elif row["CV"] <= cv_median:
            return "F"
        else:
            return "S"

    agg["FSN_Class"] = agg.apply(classify, axis=1)

    return agg[["SKU", "FSN_Class", "CV", "Months_Active", "Turnover_Rate",
                "Avg_Daily_Demand", "Std_Daily_Demand",
                "Avg_Monthly_Demand", "Total_Sales", "Lookback_Months"]]


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
    if std_demand <= 0 or lead_time <= 0:
        return 0
    return math.ceil(z * std_demand * math.sqrt(lead_time))


# ==========================================================
# 5️⃣ คำนวณคำแนะนำราย SKU
# ==========================================================
def recommend_row(row: dict, bot_cfg: dict, warehouse_available: int) -> dict:
    on_hand   = int(row.get('OnHand', 0))
    avg       = float(row.get('Avg_Daily_Demand', 0) or 0)
    std = float(row.get('Std_Daily_Demand') or 0)
    std = 0 if math.isnan(std) else std
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
        decay_rate          = 0.15,      # ✅ ปรับเป็น monthly decay
        lookback_months     = 6):        # ✅ ใหม่: lookback 6 เดือน

    xls       = pd.ExcelFile(xls_bytes_or_path)
    sku       = pd.read_excel(xls, sheet_sku)
    stock_out = pd.read_excel(xls, sheet_out)
    stock_out['Date'] = pd.to_datetime(stock_out['Date'], errors='coerce')

    if not end_date:
        end_date = stock_out['Date'].max().strftime('%Y-%m-%d')

    sales90 = build_sales_history_monthly(stock_out, end_date,
                                          lookback_months=lookback_months)
    fsn     = fsn_from_sales_monthly(sales90, lambda_decay=decay_rate)

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
        "Sales_History_Monthly"       : sales90,
        "FSN_Classification"          : fsn,
        "Recommendation"              : final,
        "Warehouse_Initial_Available" : int(initial_available),
        "Warehouse_Remaining"         : int(remain),
        "End_Date"                    : end_date,
        "Decay_Rate"                  : decay_rate,
        "Lookback_Months"             : lookback_months,
    }


# ==========================================================
# 8️⃣ Quick Test
# ==========================================================
if __name__ == "__main__":
    print("=" * 65)
    print("BOT Engine — Monthly Lookback + CV-based FSN")
    print("=" * 65)
    print("""
  Lookback Window: 6 เดือนล่าสุด (เดิม 90 วัน)
  ─────────────────────────────────────────────────────────────
  เหตุผล:
  - ข้อมูลเป็น event-based (10 transactions/เดือน ไม่ใช่รายวัน)
  - 90 วัน → ข้อมูลจริงแค่ ~30 rows/SKU มี 0 เยอะมาก
  - 6 เดือน = สมดุล trend ล่าสุด + stability เพียงพอ
  Ref: Syntetos & Boylan (2005), Int'l Journal of Forecasting

  Weighted Decay: λ = 0.15/เดือน (เดิม 0.08/วัน)
  → เดือนก่อน weight = e^(-0.15) ≈ 86%
  → 6 เดือนก่อน weight = e^(-0.90) ≈ 41%
  Ref: Gardner (1985), Journal of Forecasting

  FSN Classification: CV-based (ไม่ใช้ absolute threshold)
  - F : CV ≤ median(CV) → ขายสม่ำเสมอ
  - S : CV > median(CV) → ขายผันผวน
  - N : Months_Active = 0 → Non-moving จริงๆ
  Ref: Tersine (1994), Gopalakrishnan & Sundaresan (1994)

  Safety Stock: SS = Z × σ_daily × √Lead_Time
  Ref: Silver, Pyke & Peterson (1998)
    """)

    print("Safety Stock Verification (σ=5.0/day, LT=7d):")
    print("-" * 65)
    for fsn_class, cfg in BOT_CONFIG.items():
        ss = calc_safety_stock(std_demand=5.0, lead_time=7, z=cfg["Z"])
        print(f"  {fsn_class} | Z={cfg['Z']} | → SS={ss} units")