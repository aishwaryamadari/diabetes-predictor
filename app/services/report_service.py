"""
app/services/report_service.py — Generate downloadable PDF prediction reports.
Uses reportlab for PDF generation.
"""
import io
from datetime import datetime


def generate_pdf_report(prediction_data: dict, inputs: dict) -> bytes:
    """
    Generate a PDF report for a single prediction result.
    Returns raw PDF bytes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise ImportError("reportlab is required. pip install reportlab")

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm, bottomMargin=2*cm)

    # ── Colours ────────────────────────────────────────────────────────────
    BLUE    = colors.HexColor("#4A90E2")
    GREEN   = colors.HexColor("#27AE60")
    RED     = colors.HexColor("#E05252")
    AMBER   = colors.HexColor("#E8900C")
    BG      = colors.HexColor("#F7F9FB")
    DARK    = colors.HexColor("#2C2C2C")
    MUTED   = colors.HexColor("#7A8899")

    risk    = prediction_data.get("risk_level", "Low")
    risk_color = {"Low": GREEN, "Moderate": AMBER, "High": RED}.get(risk, BLUE)

    styles  = getSampleStyleSheet()
    h1_style = ParagraphStyle("h1", fontSize=20, textColor=DARK,
                               spaceAfter=4, fontName="Helvetica-Bold")
    h2_style = ParagraphStyle("h2", fontSize=12, textColor=BLUE,
                               spaceAfter=6, spaceBefore=14, fontName="Helvetica-Bold")
    body     = ParagraphStyle("body", fontSize=9.5, textColor=DARK,
                               spaceAfter=4, leading=14)
    small    = ParagraphStyle("small", fontSize=8, textColor=MUTED, leading=11)
    center   = ParagraphStyle("center", fontSize=9.5, textColor=DARK, alignment=TA_CENTER)

    story = []

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Diabetes Risk Assessment Report", h1_style))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%B %d, %Y  %H:%M UTC')}",
        small
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=14))

    # ── Result Banner ───────────────────────────────────────────────────────
    pct = round(prediction_data.get("probability", 0) * 100, 1)
    label = prediction_data.get("label", "—")

    banner_data = [[
        Paragraph(f'<font color="{risk_color.hexval()}" size="16"><b>{label}</b></font>', center),
        Paragraph(f'<font size="22"><b>{pct}%</b></font><br/>'
                  '<font color="#7A8899" size="8">Probability</font>', center),
        Paragraph(f'<font color="{risk_color.hexval()}" size="13"><b>{risk} Risk</b></font>', center),
    ]]
    banner_tbl = Table(banner_data, colWidths=["33%", "34%", "33%"])
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BG),
        ("ROUNDEDCORNERS", [8]),
        ("INNERGRID", (0,0), (-1,-1), 0, colors.white),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#E4EAF2")),
        ("TOPPADDING", (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 16))

    # ── Input Values ────────────────────────────────────────────────────────
    story.append(Paragraph("Patient Input Values", h2_style))

    label_map = {
        "Pregnancies":              ("Pregnancies",               "—"),
        "Glucose":                  ("Glucose",                   "mg/dL"),
        "BloodPressure":            ("Blood Pressure",            "mmHg"),
        "SkinThickness":            ("Skin Thickness",            "mm"),
        "Insulin":                  ("Insulin",                   "μU/mL"),
        "BMI":                      ("BMI",                       "kg/m²"),
        "DiabetesPedigreeFunction": ("Diabetes Pedigree Function","—"),
        "Age":                      ("Age",                       "years"),
    }

    rows = [["Metric", "Value", "Unit"]]
    for k, (display, unit) in label_map.items():
        val = inputs.get(k, "—")
        rows.append([display, str(val), unit])

    t = Table(rows, colWidths=[200, 100, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#E4EAF2")),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # ── Feature Importance ──────────────────────────────────────────────────
    factors = prediction_data.get("top_factors", [])
    if factors:
        story.append(Paragraph("Key Contributing Factors", h2_style))
        f_rows = [["Feature", "Importance"]]
        for f in factors[:5]:
            name = f["feature"].replace("_", " ")
            imp  = f"{f['importance']*100:.2f}%"
            f_rows.append([name, imp])

        ft = Table(f_rows, colWidths=[260, 140])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2C2C2C")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#E4EAF2")),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        story.append(ft)
        story.append(Spacer(1, 14))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E4EAF2"), spaceBefore=8))
    story.append(Paragraph(
        "⚠  This report is generated by an AI model for educational purposes only. "
        "It does not constitute medical advice and should not replace consultation "
        "with a qualified healthcare professional.",
        small
    ))

    doc.build(story)
    return buf.getvalue()
