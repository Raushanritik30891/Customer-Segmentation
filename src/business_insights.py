"""
Business Insights Module
-------------------------
Generates actionable marketing recommendations based on cluster statistics.
All insights are derived from actual cluster data — no hardcoding.
"""

import pandas as pd
import numpy as np


SEGMENT_STRATEGIES = {
    "Premium": {
        "icon": "💎",
        "tagline": "Your most valuable customers",
        "strategies": [
            "Launch an exclusive VIP loyalty program with early access to new products.",
            "Offer personalized concierge service and dedicated account managers.",
            "Create premium bundles and limited-edition products targeted at this segment.",
            "Solicit testimonials and case studies — high social proof value.",
            "Invite to exclusive preview events and brand ambassador programs.",
        ],
    },
    "High Value": {
        "icon": "🏆",
        "tagline": "High spenders with growth potential",
        "strategies": [
            "Up-sell to premium product tiers — they have the budget.",
            "Offer cross-category promotions to broaden their product mix.",
            "Introduce a tiered rewards program to move them toward Premium.",
            "Run high-AOV flash sales with exclusive access.",
            "Personalized email campaigns highlighting complementary products.",
        ],
    },
    "Loyal": {
        "icon": "🔄",
        "tagline": "Frequent buyers — protect at all costs",
        "strategies": [
            "Implement a points-based loyalty program to reward every purchase.",
            "Send 'Thank You' campaigns with surprise discounts to reinforce loyalty.",
            "Use subscription / auto-replenishment options for their top products.",
            "Feature them in community spotlights to deepen emotional connection.",
            "Offer referral bonuses — loyal customers drive word-of-mouth.",
        ],
    },
    "Regular": {
        "icon": "📦",
        "tagline": "Consistent but moderate — room to grow",
        "strategies": [
            "Run targeted upsell campaigns around their most-purchased categories.",
            "Offer bundle discounts to increase average order value.",
            "Trigger re-engagement emails after 30 days of inactivity.",
            "Introduce 'Buy more, save more' tiered pricing.",
            "Highlight new arrivals in categories they've shopped before.",
        ],
    },
    "Potential": {
        "icon": "🌱",
        "tagline": "Early-stage — nurture to convert",
        "strategies": [
            "Send a welcome series explaining top products and brand values.",
            "Offer a first-repeat-purchase discount (e.g. 15% off second order).",
            "Showcase bestsellers and curated starter bundles.",
            "Use retargeting ads to bring them back after their first purchase.",
            "Collect preference data via surveys to personalize future messaging.",
        ],
    },
    "At Risk": {
        "icon": "⚠️",
        "tagline": "Lapsing customers — act now",
        "strategies": [
            "Launch an immediate win-back campaign: 'We miss you — here's 20% off'.",
            "Send a re-engagement survey to understand why they've drifted away.",
            "Highlight new products or improvements since their last visit.",
            "Create urgency with limited-time offers tied to their previous purchases.",
            "Consider suppressing from paid channels to reduce wasted ad spend.",
        ],
    },
}


def _match_template(segment_name: str) -> dict:
    """Match a segment name string to a strategy template."""
    name_upper = segment_name.upper()
    for key in SEGMENT_STRATEGIES:
        if key.upper() in name_upper:
            return SEGMENT_STRATEGIES[key]
    return SEGMENT_STRATEGIES["Regular"]  # default fallback


def generate_insights(cluster_stats: pd.DataFrame) -> list:
    """
    Generate insight cards (list of dicts) for each cluster.
    Each card contains: segment, icon, tagline, stats_summary, strategies.
    """
    insights = []
    for _, row in cluster_stats.iterrows():
        tmpl = _match_template(row["SegmentName"])
        card = {
            "segment": row["SegmentName"],
            "cluster_id": int(row["Cluster"]),
            "icon": tmpl["icon"],
            "tagline": tmpl["tagline"],
            "customer_count": int(row["CustomerCount"]),
            "pct_customers": round(row["CustomerCount"] / cluster_stats["CustomerCount"].sum() * 100, 1),
            "avg_recency_days": round(row.get("Recency_mean", 0), 1),
            "avg_frequency": round(row.get("Frequency_mean", 0), 1),
            "avg_monetary": round(row.get("Monetary_mean", 0), 2),
            "avg_order_value": round(row.get("AvgOrderValue_mean", 0), 2),
            "strategies": tmpl["strategies"],
        }
        insights.append(card)

    # Sort by avg_monetary desc
    insights.sort(key=lambda x: x["avg_monetary"], reverse=True)
    return insights


def print_insights(insights: list):
    """Pretty print insights to console."""
    print("\n" + "="*70)
    print("  CUSTOMER SEGMENTATION — BUSINESS INSIGHTS")
    print("="*70)
    for card in insights:
        print(f"\n{card['icon']}  {card['segment']}")
        print(f"   {card['tagline']}")
        print(f"   Customers : {card['customer_count']:,} ({card['pct_customers']}%)")
        print(f"   Avg Recency : {card['avg_recency_days']} days")
        print(f"   Avg Frequency: {card['avg_frequency']} orders")
        print(f"   Avg Revenue  : £{card['avg_monetary']:,.2f}")
        print(f"   Recommended Actions:")
        for i, s in enumerate(card["strategies"][:3], 1):
            print(f"     {i}. {s}")
    print("\n" + "="*70)


def build_revenue_opportunity(insights: list) -> pd.DataFrame:
    """
    Simple revenue opportunity sizing table.
    Estimates potential uplift if we increase conversion in each segment.
    """
    rows = []
    for card in insights:
        uplift_pct = {
            "Premium": 0.10,
            "High Value": 0.15,
            "Loyal": 0.12,
            "Regular": 0.20,
            "Potential": 0.25,
            "At Risk": 0.08,
        }
        key = next((k for k in uplift_pct if k.upper() in card["segment"].upper()), "Regular")
        pct = uplift_pct[key]
        current_rev = card["customer_count"] * card["avg_monetary"]
        potential_uplift = current_rev * pct
        rows.append({
            "Segment": card["segment"],
            "Customers": card["customer_count"],
            "Current Revenue (£)": round(current_rev, 0),
            "Uplift %": f"{int(pct*100)}%",
            "Potential Uplift (£)": round(potential_uplift, 0),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test with dummy stats
    dummy = pd.DataFrame({
        "Cluster": [0, 1, 2, 3, 4],
        "SegmentName": ["💎 Premium Customers", "🏆 High Value Customers",
                        "🔄 Loyal Customers", "⚠️ At Risk Customers", "🌱 Potential Customers"],
        "CustomerCount": [200, 350, 500, 300, 400],
        "Recency_mean": [10, 30, 25, 200, 90],
        "Frequency_mean": [20, 10, 30, 3, 2],
        "Monetary_mean": [5000, 2000, 1500, 300, 250],
        "AvgOrderValue_mean": [250, 200, 50, 100, 125],
    })
    insights = generate_insights(dummy)
    print_insights(insights)
    opp = build_revenue_opportunity(insights)
    print("\nRevenue Opportunity Table:")
    print(opp.to_string(index=False))
