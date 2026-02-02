"""
Streamlit Web Application for Shopper Segment Response Predictor

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from likelihood_predictor import LikelihoodPredictor, AVAILABLE_ATTRIBUTES

# Page configuration
st.set_page_config(
    page_title="Shopper Response Predictor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .factor-positive {
        color: #28a745;
    }
    .factor-negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key")

    st.markdown("---")

    st.markdown("### 📊 Segments Overview")
    st.markdown("""
    | Segment | Size |
    |---------|------|
    | Premium Purity Seekers | 37.1% |
    | Price-Conscious Pragmatists | 21.6% |
    | Values-Driven Advocates | 17.5% |
    | Brand Loyalists | 15.5% |
    | Low-Engagement Shoppers | 8.2% |
    """)

    st.markdown("---")

    st.markdown("### 📝 Example Questions")
    example_questions = [
        "Would you try a new organic snack priced 20% higher?",
        "Would you switch from conventional to organic milk?",
        "How likely are you to read ingredient labels?",
        "Would you pay extra for sustainably sourced products?",
        "Would you try a new brand if it had a clean label?"
    ]
    for i, q in enumerate(example_questions):
        if st.button(q, key=f"example_{i}"):
            st.session_state["question_input"] = q

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This tool predicts how different shopper segments
    would respond to questions/scenarios based on their
    behavioral attribute profiles.

    **How it works:**
    1. Enter a question about shopping behavior
    2. AI extracts relevant attributes with weights
    3. Calculates likelihood for each segment
    4. Shows detailed factors driving predictions
    """)


# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""


# Main content
st.title("🛒 Shopper Segment Response Predictor")
st.markdown("Enter a question to predict how each segment would respond based on their attribute profiles.")

# Question input
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_area(
        "Enter a question or scenario:",
        value=st.session_state.get("question_input", ""),
        placeholder="e.g., Would you try a new organic snack priced 20% higher than your usual?",
        height=100,
        key="question_text"
    )

with col2:
    st.write("")  # Spacing
    st.write("")
    predict_clicked = st.button("🔮 Predict Response", type="primary", use_container_width=True)
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state["question_input"] = ""
    st.rerun()

# Prediction logic
if predict_clicked:
    if not api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar")
    elif not question.strip():
        st.error("⚠️ Please enter a question")
    else:
        with st.spinner("🔄 Analyzing question and calculating likelihoods..."):
            try:
                predictor = LikelihoodPredictor(api_key)
                result = predictor.predict(question.strip())

                # Store in history
                st.session_state["history"].append({
                    "question": question.strip(),
                    "result": result
                })

                # Display results
                st.success("✅ Analysis complete!")

                # Extracted Attributes Section
                st.subheader("📊 Extracted Attributes")

                attr_data = []
                for attr, info in result["attributes"].items():
                    attr_data.append({
                        "Attribute": attr,
                        "Weight": info["weight"],
                        "Polarity": "➕ Positive" if info["polarity"] == 1 else "➖ Negative",
                        "Reason": info["reason"]
                    })

                attr_df = pd.DataFrame(attr_data)
                attr_df = attr_df.sort_values("Weight", ascending=False)

                # Display as styled dataframe
                st.dataframe(
                    attr_df.style.background_gradient(subset=["Weight"], cmap="YlGn"),
                    use_container_width=True,
                    hide_index=True
                )

                # Likelihood Chart Section
                st.subheader("📈 Segment Likelihood")

                pred_df = pd.DataFrame(result["predictions"])
                pred_df = pred_df.sort_values("likelihood", ascending=True)

                # Create horizontal bar chart
                fig = go.Figure()

                # Color scale based on likelihood
                colors = pred_df["likelihood"].apply(
                    lambda x: f"rgb({int(255*(1-x))}, {int(200*x + 55)}, {int(100*x)})"
                ).tolist()

                fig.add_trace(go.Bar(
                    x=pred_df["likelihood"],
                    y=pred_df["segment"],
                    orientation="h",
                    marker=dict(
                        color=pred_df["likelihood"],
                        colorscale="RdYlGn",
                        cmin=0,
                        cmax=1
                    ),
                    text=pred_df["likelihood"].apply(lambda x: f"{x:.0%}"),
                    textposition="auto",
                    hovertemplate="<b>%{y}</b><br>Likelihood: %{x:.1%}<extra></extra>"
                ))

                fig.update_layout(
                    xaxis_title="Likelihood",
                    yaxis_title="",
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                summary_cols = st.columns(5)
                for i, pred in enumerate(sorted(result["predictions"], key=lambda x: -x["likelihood"])):
                    with summary_cols[i % 5]:
                        likelihood_pct = pred["likelihood"] * 100
                        color = "green" if likelihood_pct >= 60 else ("orange" if likelihood_pct >= 40 else "red")
                        st.metric(
                            label=pred["segment"].split()[0],  # First word
                            value=f"{likelihood_pct:.0f}%",
                            delta=pred["interpretation"].split(" - ")[0]
                        )

                # Segment Details Section
                st.subheader("🔍 Segment Details")

                for pred in sorted(result["predictions"], key=lambda x: -x["likelihood"]):
                    likelihood_pct = pred["likelihood"] * 100
                    emoji = "🟢" if likelihood_pct >= 60 else ("🟡" if likelihood_pct >= 40 else "🔴")

                    with st.expander(f"{emoji} {pred['segment']} ({likelihood_pct:.0f}% likelihood) - Size: {pred['size_pct']:.1f}%"):
                        st.markdown(f"**Interpretation:** {pred['interpretation']}")

                        # Representative Customer Example
                        st.markdown("---")
                        rep_customer = pred["representative_customer"]
                        customer_likelihood_pct = rep_customer["likelihood"] * 100

                        # Determine response text based on likelihood
                        if customer_likelihood_pct >= 60:
                            response_text = "This customer would likely say **YES** because..."
                        elif customer_likelihood_pct >= 40:
                            response_text = "This customer might say **MAYBE** because..."
                        else:
                            response_text = "This customer would likely say **NO** because..."

                        st.markdown(f"### 👤 Example Customer: `{rep_customer['customer_id']}`")
                        st.markdown(response_text)
                        st.markdown("")

                        # Display relevant scores
                        st.markdown("**Their relevant scores:**")
                        for attr, score in rep_customer["relevant_scores"].items():
                            # Get polarity from extracted attributes
                            polarity = result["attributes"][attr].get("polarity", 1)

                            # Determine if this score helps or hurts
                            if polarity == 1:
                                is_helpful = score >= 0.5
                            else:
                                is_helpful = score < 0.5

                            icon = "✅" if is_helpful else "❌"

                            # Score interpretation
                            if score >= 0.8:
                                level = "very high"
                            elif score >= 0.6:
                                level = "high"
                            elif score >= 0.4:
                                level = "moderate"
                            elif score >= 0.2:
                                level = "low"
                            else:
                                level = "very low"

                            st.markdown(f"{icon} **{attr}**: `{score:.2f}` ({level})")

                        st.markdown(f"\n**Individual Likelihood:** `{customer_likelihood_pct:.1f}%`")

                        st.markdown("---")
                        st.markdown("**Key Factors:**")

                        # Separate positive and negative factors
                        positive_factors = [f for f in pred["key_factors"] if f["impact"] > 0]
                        negative_factors = [f for f in pred["key_factors"] if f["impact"] <= 0]

                        col_pos, col_neg = st.columns(2)

                        with col_pos:
                            st.markdown("**✅ Supporting Factors**")
                            if positive_factors:
                                for f in positive_factors[:5]:
                                    polarity_str = "+" if f["polarity"] == 1 else "-"
                                    st.markdown(f"- **{f['attribute']}**: {f['score']:.2f} (weight={f['weight']:.2f}, polarity={polarity_str})")
                            else:
                                st.markdown("*None*")

                        with col_neg:
                            st.markdown("**❌ Opposing Factors**")
                            if negative_factors:
                                for f in negative_factors[:5]:
                                    polarity_str = "+" if f["polarity"] == 1 else "-"
                                    st.markdown(f"- **{f['attribute']}**: {f['score']:.2f} (weight={f['weight']:.2f}, polarity={polarity_str})")
                            else:
                                st.markdown("*None*")

                        # Decision Tree Breakdown
                        st.markdown("---")
                        st.markdown("**🌳 Decision Tree Breakdown:**")

                        decision_tree = pred["decision_tree"]

                        for step in decision_tree["steps"]:
                            # Determine color based on verdict
                            if step["verdict"] in ["STRONG YES", "LEAN YES"]:
                                color = "green"
                                icon = "✅"
                            else:
                                color = "red"
                                icon = "❌"

                            with st.container():
                                st.markdown(f"""
**Step {step['step']}: {step['attribute'].upper()}** {step['symbol']}

| | |
|---|---|
| Question | {step['question']} |
| Segment Score | `{step['raw_score']:.2f}` |
| Polarity | `{step['polarity']:+d}` {'(inverted)' if step['polarity'] == -1 else '(direct)'} |
| Adjusted Score | `{step['adjusted_score']:.2f}` |
| Weight | `{step['weight']:.2f}` |
| Contribution | `{step['adjusted_score']:.2f} × {step['weight']:.2f} = {step['contribution']:.3f}` |
| Verdict | {icon} **{step['verdict']}** |
| Running Likelihood | **{step['running_likelihood']*100:.1f}%** |
""")
                                st.markdown("---")

                        # Final result box
                        st.success(f"**FINAL: {decision_tree['final_likelihood']*100:.1f}%** - {decision_tree['interpretation']}")

                        # Show segment profile chart
                        st.markdown("**Relevant Attribute Profile:**")

                        # Get only relevant attributes for this segment
                        relevant_attrs = list(result["attributes"].keys())
                        profile_data = predictor.profiles[pred["segment_id"]]

                        radar_df = pd.DataFrame({
                            "Attribute": relevant_attrs,
                            "Score": [profile_data[attr] for attr in relevant_attrs]
                        })

                        fig_radar = px.bar(
                            radar_df,
                            x="Attribute",
                            y="Score",
                            color="Score",
                            color_continuous_scale="RdYlGn",
                            range_color=[0, 1]
                        )
                        fig_radar.update_layout(
                            height=250,
                            xaxis_tickangle=-45,
                            margin=dict(l=20, r=20, t=20, b=80)
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# History section
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("📜 Question History")

    for i, item in enumerate(reversed(st.session_state["history"][-5:])):
        with st.expander(f"Q{len(st.session_state['history']) - i}: {item['question'][:50]}..."):
            st.markdown(f"**Full Question:** {item['question']}")

            # Quick summary
            preds = item["result"]["predictions"]
            top_seg = max(preds, key=lambda x: x["likelihood"])
            bottom_seg = min(preds, key=lambda x: x["likelihood"])

            st.markdown(f"""
            **Summary:**
            - Highest: {top_seg['segment']} ({top_seg['likelihood']:.0%})
            - Lowest: {bottom_seg['segment']} ({bottom_seg['likelihood']:.0%})
            - Attributes used: {', '.join(item['result']['attributes'].keys())}
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>Built with Streamlit | Powered by Claude AI | Based on shopper segmentation data</small>
</div>
""", unsafe_allow_html=True)
