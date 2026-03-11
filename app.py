# ==========================================
# Inventory Bot API (Power Platform Ready)
# ==========================================

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
import time
import base64

from logic import end_to_end_dynamic_from_excel

app = FastAPI(title="Inventory Bot API")

# ==========================================
# CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Health Check
# ==========================================
@app.get("/")
def root():
    return {"status": "API running on Azure"}

# ==========================================================
# Request Model (Base64 JSON)
# ✅ เพิ่ม lookback_months / ปรับ decay_rate default → 0.15
# ==========================================================
class ExcelBase64Request(BaseModel):
    file_base64: str
    max_capacity: int = 30000
    decay_rate: float = 0.15        # ✅ เปลี่ยนจาก 0.08 → 0.15 (monthly)
    lookback_months: int = 6        # ✅ ใหม่


# ==========================================================
# Helper: Calculate warehouse_available
# ==========================================================
def calculate_warehouse_available(file_buffer, max_capacity):
    df = pd.read_excel(file_buffer, sheet_name="SKU_Master")

    if "OnHand" not in df.columns:
        raise Exception("Column 'OnHand' not found in SKU_Master")

    total_onhand_positive = df[df["OnHand"] > 0]["OnHand"].sum()
    warehouse_available   = max_capacity - total_onhand_positive

    return warehouse_available


# ==========================================================
# 1) JSON → Return JSON
# ==========================================================
@app.post("/analyze-excel-json")
async def analyze_excel_json(data: ExcelBase64Request):
    try:
        decoded = base64.b64decode(data.file_base64)
        tmp = io.BytesIO(decoded)

        warehouse_available = calculate_warehouse_available(tmp, data.max_capacity)
        tmp.seek(0)

        out = end_to_end_dynamic_from_excel(
            tmp,
            sheet_sku           = "SKU_Master",
            sheet_out           = "STOCK_OUT",
            warehouse_available = warehouse_available,
            decay_rate          = data.decay_rate,
            max_capacity        = data.max_capacity,
            lookback_months     = data.lookback_months,   # ✅ ใหม่
        )

        return {
            "success"                    : True,
            "warehouse_initial_available": warehouse_available,
            "warehouse_remaining"        : out["Warehouse_Remaining"],
            "end_date"                   : out["End_Date"],
            "lookback_months"            : out["Lookback_Months"],   # ✅ ใหม่
            "fsn"                        : out["FSN_Classification"].to_dict(orient="records"),
            "recommendation"             : out["Recommendation"].to_dict(orient="records"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ==========================================================
# 2) JSON → Return Excel (Download)
# ==========================================================
@app.post("/analyze-excel-xlsx")
async def analyze_excel_xlsx(data: ExcelBase64Request):
    try:
        decoded = base64.b64decode(data.file_base64)
        tmp = io.BytesIO(decoded)

        warehouse_available = calculate_warehouse_available(tmp, data.max_capacity)
        tmp.seek(0)

        out = end_to_end_dynamic_from_excel(
            tmp,
            sheet_sku           = "SKU_Master",
            sheet_out           = "STOCK_OUT",
            warehouse_available = warehouse_available,
            decay_rate          = data.decay_rate,
            max_capacity        = data.max_capacity,
            lookback_months     = data.lookback_months,   # ✅ ใหม่
        )

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            # ✅ เปลี่ยนชื่อ key: Sales_History_90days → Sales_History_Monthly
            out["Sales_History_Monthly"].to_excel(
                w, index=False, sheet_name="Sales_History_Monthly")
            out["FSN_Classification"].to_excel(
                w, index=False, sheet_name="FSN_Classification")
            out["Recommendation"].to_excel(
                w, index=False, sheet_name="Recommendation")

            pd.DataFrame([{
                "Warehouse_Max_Capacity"     : data.max_capacity,
                "Warehouse_Initial_Available": warehouse_available,
                "Warehouse_Remaining"        : out["Warehouse_Remaining"],
                "SKU_Count"                  : len(out["Recommendation"]),
                "Lambda_Decay"               : out["Decay_Rate"],
                "Lookback_Months"            : out["Lookback_Months"],  # ✅ ใหม่
                "End_Date"                   : out["End_Date"],
            }]).to_excel(w, index=False, sheet_name="Summary")

        buf.seek(0)
        filename = f"Bot_Recommendation_{int(time.time())}.xlsx"

        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )


# ==========================================================
# 3) Swagger Upload File → Return Excel
# ✅ เพิ่ม lookback_months Form param / ปรับ decay_rate default
# ==========================================================
@app.post("/analyze-excel-upload")
async def analyze_excel_upload(
    file: UploadFile = File(...),
    max_capacity:    int   = Form(30000),
    decay_rate:      float = Form(0.15),   # ✅ เปลี่ยนจาก 0.08 → 0.15
    lookback_months: int   = Form(6),      # ✅ ใหม่
):
    try:
        data = await file.read()
        tmp  = io.BytesIO(data)

        warehouse_available = calculate_warehouse_available(tmp, max_capacity)
        tmp.seek(0)

        out = end_to_end_dynamic_from_excel(
            tmp,
            sheet_sku           = "SKU_Master",
            sheet_out           = "STOCK_OUT",
            warehouse_available = warehouse_available,
            decay_rate          = decay_rate,
            max_capacity        = max_capacity,
            lookback_months     = lookback_months,   # ✅ ใหม่
        )

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            # ✅ เปลี่ยนชื่อ key: Sales_History_90days → Sales_History_Monthly
            out["Sales_History_Monthly"].to_excel(
                w, index=False, sheet_name="Sales_History_Monthly")
            out["FSN_Classification"].to_excel(
                w, index=False, sheet_name="FSN_Classification")
            out["Recommendation"].to_excel(
                w, index=False, sheet_name="Recommendation")

        buf.seek(0)
        filename = f"Bot_Recommendation_{int(time.time())}.xlsx"

        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)},
        )