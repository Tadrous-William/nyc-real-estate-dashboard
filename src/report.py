"""Generate the monthly NYC market report as a PDF (reports/nyc_market_report_YYYY-MM.pdf)."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from common import PROCESSED, REPORTS, mix_adjusted_change  # noqa: E402

BLUE = colors.HexColor("#2563EB")
INK = colors.HexColor("#0F172A")
GREY = colors.HexColor("#64748B")
BORO = {"Manhattan": "#2563EB", "Brooklyn": "#DC2626", "Queens": "#059669", "Bronx": "#D97706",
        "Staten Island": "#7C3AED"}
MAIN = ["One-family home", "Two-family home", "Three-family home", "Condo apartment", "Co-op apartment"]


def money(x):
    return f"${x / 1e6:,.2f}M" if x >= 1e6 else f"${x:,.0f}"


def chart_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def main():
    s = pd.read_parquet(PROCESSED / "sales.parquet")
    s = s[s["property_type"].isin(MAIN)]
    zy = pd.read_parquet(PROCESSED / "zip_yield.parquet")
    month = s["month"].max()
    q_now = s[s["month"] > month - pd.DateOffset(months=3)]
    q_ago = s[(s["month"] > month - pd.DateOffset(months=15)) & (s["month"] <= month - pd.DateOffset(months=12))]
    m_now = s[s["month"] == month]

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=INK, fontSize=22, leading=26, alignment=0)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=INK, fontSize=14, spaceBefore=10)
    body = ParagraphStyle("b", parent=styles["BodyText"], textColor=GREY, fontSize=9.5, leading=13)
    story = [Paragraph(f"NYC Housing Market Report: {month:%B %Y}", h1),
             Paragraph("Homes, condos and co-ops across the five boroughs. Public NYC Department of Finance sales "
                       "and Zillow rent data. Prepared by Tadrous William, data scientist.", body), Spacer(1, 12)]

    mp, mp0 = q_now["sale_price"].median(), q_ago["sale_price"].median()
    pp, pp0 = q_now["price_per_sqft"].median(), q_ago["price_per_sqft"].median()
    kpis = [["Sales this month", "Median price (3 mo.)", "Price per sq ft (3 mo.)", "Sales over $1M (3 mo.)"],
            [f"{len(m_now):,}", money(mp), money(pp), f"{(q_now['sale_price'] >= 1e6).mean():.0%}"],
            ["", f"{mp / mp0 - 1:+.1%} vs a year ago", f"{pp / pp0 - 1:+.1%} vs a year ago", ""]]
    t = Table(kpis, colWidths=[4.2 * cm] * 4)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica", 8), ("TEXTCOLOR", (0, 0), (-1, 0), GREY),
                           ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 16), ("TEXTCOLOR", (0, 1), (-1, 1), BLUE),
                           ("FONT", (0, 2), (-1, 2), "Helvetica", 8), ("TEXTCOLOR", (0, 2), (-1, 2), GREY),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [t, Spacer(1, 14)]

    recent = s[s["month"] > month - pd.DateOffset(months=36)]
    mon = recent.groupby(["month", "borough_name"])["sale_price"].median().unstack()
    mon = mon.rolling(3, min_periods=1).median()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for b in mon.columns:
        ax.plot(mon.index, mon[b] / 1e6, label=b, color=BORO[b], lw=2)
    ax.set_title("Median sale price by borough, last 3 years (3-month rolling)", loc="left", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:.1f}M"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper left")
    story += [Image(chart_png(fig), width=17 * cm, height=7.6 * cm), Spacer(1, 8)]

    story.append(Paragraph("By borough (last 3 months vs same period a year ago)", h2))
    rows = [["Borough", "Sales", "Median price", "1-year change*", "Price per sq ft"]]
    bchg = mix_adjusted_change(q_now, q_ago, ["borough_name"])["price_change_1y"]
    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        a = q_now[q_now["borough_name"] == b]
        rows.append([b, f"{len(a):,}", money(a["sale_price"].median()),
                     f"{bchg.get(b, float('nan')):+.1%}",
                     money(a["price_per_sqft"].median()) if a["price_per_sqft"].notna().any() else "-"])
    bt = Table(rows, colWidths=[4 * cm, 2.4 * cm, 3.4 * cm, 3 * cm, 3.4 * cm])
    bt.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9), ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                            ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                            ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
    story += [bt, Spacer(1, 4),
              Paragraph("* Like-for-like: compared within the same property type, weighted by this year's sales.", body),
              PageBreak()]

    y_now = s[s["month"] > month - pd.DateOffset(months=12)]
    y_ago = s[(s["month"] > month - pd.DateOffset(months=24)) & (s["month"] <= month - pd.DateOffset(months=12))]
    info = y_now.groupby("neighborhood").agg(n=("sale_price", "size"), p=("sale_price", "median"),
                                             b=("borough_name", "first"))
    g = info.join(mix_adjusted_change(y_now, y_ago, ["neighborhood"], min_type_sales=15, min_total=50), how="inner")
    top = g.nlargest(10, "price_change_1y")
    story.append(Paragraph("Fastest-rising neighborhoods (last 12 months vs the year before)", h2))
    story.append(Paragraph("Like-for-like change: prices are compared within the same property type (houses with "
                           "houses, condos with condos), so a shift in what was sold does not look like growth.", body))
    rows = [["Neighborhood", "Borough", "Sales", "Median price", "1-year change"]]
    rows += [[i, r.b, f"{r.n:,}", money(r.p), f"{r.price_change_1y:+.1%}"] for i, r in top.iterrows()]
    nt = Table(rows, colWidths=[5 * cm, 3 * cm, 2 * cm, 3.4 * cm, 3 * cm])
    nt.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9), ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                            ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                            ("ALIGN", (2, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
    story += [nt, Spacer(1, 10)]

    story.append(Paragraph("Highest gross rental yields by zip code", h2))
    story.append(Paragraph("Gross yield = 12 x typical monthly rent (Zillow ZORI) / median sale price of condos and "
                           "one-family homes in the last 12 months, before taxes, fees and vacancy. Co-ops are excluded "
                           "because their price does not include the monthly maintenance fee.", body))
    zt = zy.head(10)
    rows = [["Zip", "Main neighborhood", "Borough", "Median price", "Monthly rent", "Gross yield"]]
    rows += [[r.zip_code, r.neighborhood, r.borough_name, money(r.median_price), money(r.rent), f"{r.gross_yield:.1%}"]
             for r in zt.itertuples()]
    yt = Table(rows, colWidths=[1.6 * cm, 4.6 * cm, 2.8 * cm, 2.8 * cm, 2.6 * cm, 2.2 * cm])
    yt.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9), ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#059669")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDF4")]),
                            ("ALIGN", (3, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
    story += [Spacer(1, 4), yt, Spacer(1, 14),
              Paragraph("Sources: NYC Department of Finance Rolling and Annualized Sales (NYC Open Data), Zillow "
                        "Observed Rent Index, US Census ZCTA gazetteer. Market sales only: transfers under $50,000 "
                        "and multi-property portfolio deeds are excluded.", body)]

    out = REPORTS / f"nyc_market_report_{month:%Y-%m}.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                      bottomMargin=1.6 * cm, title=f"NYC Housing Market Report {month:%B %Y}",
                      author="Tadrous William").build(story)
    print("saved", out)


if __name__ == "__main__":
    main()
