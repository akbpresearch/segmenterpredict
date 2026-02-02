"""
Attribute-Based Likelihood Predictor

A data-driven prediction system that:
1. Identifies which attributes a question relates to (using LLM)
2. Uses actual segment profile scores to calculate response likelihood
3. Returns numeric predictions grounded in the clustered data
"""

import json
import pandas as pd
from anthropic import Anthropic


# Attribute extraction prompt template
ATTRIBUTE_EXTRACTION_PROMPT = """You are analyzing a question about food shopping behavior to identify which behavioral attributes are relevant.

QUESTION: "{question}"

AVAILABLE ATTRIBUTES (20 total):

CAPABILITY:
- label_literacy: Ability to read and understand food labels, ingredient lists, certifications
- nutrition_knowledge: Understanding of nutrition concepts (macros, nutrients, health impacts)
- planning_ability: Level of meal planning, routine structure, organized shopping
- cooking_skill: Ability to cook from scratch with fresh ingredients

VALUES:
- health_priority: How much health considerations drive food decisions
- purity_priority: Importance of clean ingredients, avoiding additives, "natural" products
- sustainability_priority: Environmental concern, ethical sourcing, eco-practices
- authority_trust: Trust in certifications (FDA/USDA), expert guidance
- greenwashing_skepticism: Critical questioning of marketing claims

DECISION STYLE:
- brand_loyalty: Tendency to stick with familiar brands vs switching
- novelty_seeking: Interest in trying new products, discovery behavior
- satisficing: Acceptance of "good enough" vs optimization
- risk_aversion: Preference for safe choices vs experimentation

TRADEOFF:
- price_pain: Sensitivity to price as a barrier
- premium_wtp: Willingness to pay premium for values-aligned products

MECHANISMS:
- purity_avoidance: Disgust-based avoidance of "contaminating" ingredients
- outcome_optimization: Focus on specific health outcomes/performance
- identity_signaling: Food choices as expression of values/social belonging
- convenience_rationalization: Tendency to justify convenience over ideals
- price_sensitivity: Impact of price on actual purchasing decisions

TASK:
1. Identify which attributes are relevant to this question
2. Assign a weight (0.0-1.0) for how strongly each attribute influences the answer
3. Assign polarity: +1 if HIGH score means more likely YES, -1 if HIGH score means more likely NO

Example polarities:
- "Would you try a new brand?" → brand_loyalty has polarity -1 (high loyalty = LESS likely to try new)
- "Would you pay extra for organic?" → premium_wtp has polarity +1 (high WTP = MORE likely)
- "Would you switch from your usual brand?" → brand_loyalty polarity -1, risk_aversion polarity -1

Return ONLY valid JSON:
{{
  "attributes": {{
    "attribute_name": {{
      "weight": 0.0-1.0,
      "polarity": 1 or -1,
      "reason": "brief explanation"
    }},
    ...
  }}
}}

Only include attributes with weight >= 0.2"""


# All available attributes
AVAILABLE_ATTRIBUTES = [
    "label_literacy", "nutrition_knowledge", "planning_ability", "cooking_skill",
    "health_priority", "purity_priority", "sustainability_priority", "authority_trust",
    "greenwashing_skepticism", "brand_loyalty", "novelty_seeking", "satisficing",
    "risk_aversion", "price_pain", "premium_wtp", "purity_avoidance",
    "outcome_optimization", "identity_signaling", "convenience_rationalization",
    "price_sensitivity"
]


class LikelihoodPredictor:
    """
    Main prediction engine that extracts relevant attributes from questions
    and calculates likelihood scores for each segment.
    """

    def __init__(self, api_key: str, profiles_path: str = "segment_profiles.csv",
                 customers_path: str = "clustered_respondents.csv"):
        """
        Initialize the predictor.

        Args:
            api_key: Anthropic API key
            profiles_path: Path to segment_profiles.csv
            customers_path: Path to clustered_respondents.csv
        """
        self.client = Anthropic(api_key=api_key)
        self.profiles = self._load_profiles(profiles_path)
        self.segment_names = self._load_segment_names(profiles_path)
        self.segment_sizes = self._load_segment_sizes(profiles_path)
        self.customers = self._load_customers(customers_path)

    def _load_profiles(self, path: str) -> dict:
        """Load segment profiles from CSV into dict format."""
        df = pd.read_csv(path)
        profiles = {}
        for _, row in df.iterrows():
            seg_id = int(row["Segment"])
            profiles[seg_id] = {
                attr: row[attr] for attr in AVAILABLE_ATTRIBUTES
            }
        return profiles

    def _load_segment_names(self, path: str) -> dict:
        """Load segment names from CSV."""
        df = pd.read_csv(path)
        return {int(row["Segment"]): row["Segment_Name"] for _, row in df.iterrows()}

    def _load_segment_sizes(self, path: str) -> dict:
        """Load segment sizes (percentages) from CSV."""
        df = pd.read_csv(path)
        return {int(row["Segment"]): row["Percentage"] for _, row in df.iterrows()}

    def _load_customers(self, path: str) -> pd.DataFrame:
        """Load individual customer data from clustered_respondents.csv."""
        return pd.read_csv(path)

    def get_representative_customer(self, segment_id: int, extracted_attrs: dict,
                                    segment_likelihood: float) -> dict:
        """
        Find the most representative customer - one whose individual likelihood
        is closest to the segment's mean likelihood.

        This shows a "typical" person from that segment.

        Args:
            segment_id: The segment ID to find a representative for
            extracted_attrs: Dict of attribute -> {weight, polarity, reason}
            segment_likelihood: The calculated segment mean likelihood

        Returns:
            Dict with customer_id, likelihood, and relevant_scores
        """
        segment_customers = self.customers[self.customers["Segment"] == segment_id]

        best_customer = None
        best_diff = float('inf')
        best_likelihood = 0

        for _, row in segment_customers.iterrows():
            customer_profile = {attr: row[attr] for attr in AVAILABLE_ATTRIBUTES}
            customer_likelihood = self.calculate_likelihood(customer_profile, extracted_attrs)

            # Find customer closest to segment mean likelihood
            diff = abs(customer_likelihood - segment_likelihood)
            if diff < best_diff:
                best_diff = diff
                best_customer = row
                best_likelihood = customer_likelihood

        return {
            "customer_id": best_customer["Customer_ID"],
            "likelihood": best_likelihood,
            "relevant_scores": {attr: best_customer[attr] for attr in extracted_attrs.keys()}
        }

    def extract_attributes(self, question: str) -> dict:
        """
        Use Claude to identify relevant attributes, weights, and polarities.

        Args:
            question: The question or scenario to analyze

        Returns:
            Dict of attribute -> {weight, polarity, reason}
        """
        prompt = ATTRIBUTE_EXTRACTION_PROMPT.format(question=question)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parse JSON response
        response_text = response.content[0].text

        # Extract JSON from response (handle potential markdown code blocks)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        parsed = json.loads(json_str.strip())
        return parsed.get("attributes", parsed)

    def calculate_likelihood(self, segment_profile: dict, extracted_attrs: dict) -> float:
        """
        Calculate weighted likelihood from segment scores.

        Args:
            segment_profile: Dict of attribute -> score (0-1)
            extracted_attrs: Dict of attribute -> {weight, polarity, reason}

        Returns:
            Likelihood score (0-1)
        """
        weighted_sum = 0
        total_weight = 0

        for attr, info in extracted_attrs.items():
            if attr not in segment_profile:
                continue

            score = segment_profile[attr]
            weight = info.get("weight", 0.5)
            polarity = info.get("polarity", 1)

            # If polarity is -1, invert the score (high loyalty = low switch likelihood)
            adjusted_score = score if polarity == 1 else (1 - score)

            weighted_sum += adjusted_score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _get_key_factors(self, segment_profile: dict, extracted_attrs: dict) -> list:
        """
        Get the key factors influencing the likelihood for a segment.

        Returns list of dicts with attribute, score, impact, and explanation.
        """
        factors = []

        for attr, info in extracted_attrs.items():
            if attr not in segment_profile:
                continue

            score = segment_profile[attr]
            weight = info.get("weight", 0.5)
            polarity = info.get("polarity", 1)

            # Calculate impact (how much this attribute contributes)
            adjusted_score = score if polarity == 1 else (1 - score)
            impact = adjusted_score * weight

            # Determine if this is helping or hurting likelihood
            if polarity == 1:
                if score >= 0.6:
                    direction = "supports"
                    sign = 1
                else:
                    direction = "weakens"
                    sign = -1
            else:
                if score >= 0.6:
                    direction = "reduces"
                    sign = -1
                else:
                    direction = "supports"
                    sign = 1

            factors.append({
                "attribute": attr,
                "score": score,
                "weight": weight,
                "polarity": polarity,
                "impact": impact * (1 if sign > 0 else -1),
                "explanation": f"{attr}={score:.2f} {direction} likelihood (polarity={polarity:+d})"
            })

        # Sort by absolute impact
        factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return factors

    def _interpret_likelihood(self, likelihood: float) -> str:
        """Convert likelihood score to human-readable interpretation."""
        if likelihood >= 0.75:
            return "HIGH - very likely"
        elif likelihood >= 0.60:
            return "MODERATE-HIGH - likely"
        elif likelihood >= 0.45:
            return "MODERATE - may or may not"
        elif likelihood >= 0.30:
            return "MODERATE-LOW - unlikely"
        else:
            return "LOW - very unlikely"

    def get_decision_tree_breakdown(self, segment_profile: dict, extracted_attrs: dict, segment_name: str) -> dict:
        """
        Generate a decision tree style breakdown showing how each attribute
        contributes to the final likelihood score.

        Returns dict with steps, running totals, and final score.
        """
        steps = []
        running_weighted_sum = 0
        running_total_weight = 0

        # Sort attributes by weight (most important first)
        sorted_attrs = sorted(
            extracted_attrs.items(),
            key=lambda x: x[1].get("weight", 0),
            reverse=True
        )

        for attr, info in sorted_attrs:
            if attr not in segment_profile:
                continue

            score = segment_profile[attr]
            weight = info.get("weight", 0.5)
            polarity = info.get("polarity", 1)
            reason = info.get("reason", "")

            # Calculate adjusted score
            adjusted_score = score if polarity == 1 else (1 - score)

            # Calculate contribution
            contribution = adjusted_score * weight

            # Update running totals
            running_weighted_sum += contribution
            running_total_weight += weight

            # Calculate running likelihood
            running_likelihood = running_weighted_sum / running_total_weight if running_total_weight > 0 else 0.5

            # Determine verdict for this attribute
            if adjusted_score >= 0.7:
                verdict = "STRONG YES"
                symbol = "+++"
            elif adjusted_score >= 0.5:
                verdict = "LEAN YES"
                symbol = "+"
            elif adjusted_score >= 0.3:
                verdict = "LEAN NO"
                symbol = "-"
            else:
                verdict = "STRONG NO"
                symbol = "---"

            steps.append({
                "step": len(steps) + 1,
                "attribute": attr,
                "question": self._get_attribute_question(attr, polarity),
                "raw_score": score,
                "polarity": polarity,
                "adjusted_score": adjusted_score,
                "weight": weight,
                "contribution": contribution,
                "verdict": verdict,
                "symbol": symbol,
                "reason": reason,
                "running_likelihood": running_likelihood,
                "running_weighted_sum": running_weighted_sum,
                "running_total_weight": running_total_weight
            })

        final_likelihood = running_weighted_sum / running_total_weight if running_total_weight > 0 else 0.5

        return {
            "segment_name": segment_name,
            "steps": steps,
            "final_likelihood": final_likelihood,
            "interpretation": self._interpret_likelihood(final_likelihood)
        }

    def _get_attribute_question(self, attr: str, polarity: int) -> str:
        """Generate a yes/no question for each attribute based on polarity."""
        questions = {
            "label_literacy": ("Can they read labels well?", "Do they struggle with labels?"),
            "nutrition_knowledge": ("Do they understand nutrition?", "Do they lack nutrition knowledge?"),
            "planning_ability": ("Do they plan meals?", "Do they shop impulsively?"),
            "cooking_skill": ("Can they cook from scratch?", "Do they avoid cooking?"),
            "health_priority": ("Is health a priority?", "Do they ignore health?"),
            "purity_priority": ("Do they want pure/clean food?", "Do they ignore purity?"),
            "sustainability_priority": ("Do they care about sustainability?", "Do they ignore eco-concerns?"),
            "authority_trust": ("Do they trust certifications?", "Do they distrust authorities?"),
            "greenwashing_skepticism": ("Are they skeptical of claims?", "Do they trust marketing?"),
            "brand_loyalty": ("Will they stick with brands?", "Are they open to switching?"),
            "novelty_seeking": ("Do they try new things?", "Do they avoid new things?"),
            "satisficing": ("Do they accept 'good enough'?", "Do they optimize choices?"),
            "risk_aversion": ("Do they avoid risks?", "Are they risk-tolerant?"),
            "price_pain": ("Is price a barrier?", "Is price not a concern?"),
            "premium_wtp": ("Will they pay premium?", "Do they avoid premium prices?"),
            "purity_avoidance": ("Do they avoid 'bad' ingredients?", "Are they tolerant of additives?"),
            "outcome_optimization": ("Do they focus on health outcomes?", "Do they ignore outcomes?"),
            "identity_signaling": ("Do food choices signal identity?", "Is food just functional?"),
            "convenience_rationalization": ("Do they justify convenience?", "Do they resist shortcuts?"),
            "price_sensitivity": ("Does price affect decisions?", "Is price ignored?")
        }

        if attr in questions:
            return questions[attr][0] if polarity == 1 else questions[attr][1]
        return f"Is {attr} high?" if polarity == 1 else f"Is {attr} low?"

    def print_decision_tree(self, breakdown: dict) -> str:
        """Generate a formatted decision tree string for display."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"  DECISION TREE: {breakdown['segment_name']}")
        lines.append(f"{'='*70}\n")

        for step in breakdown["steps"]:
            # Step header
            lines.append(f"STEP {step['step']}: {step['attribute'].upper()}")
            lines.append(f"├── Question: {step['question']}")
            lines.append(f"├── Segment Score: {step['raw_score']:.2f}")

            if step["polarity"] == -1:
                lines.append(f"├── Polarity: -1 (inverted: 1 - {step['raw_score']:.2f} = {step['adjusted_score']:.2f})")
            else:
                lines.append(f"├── Polarity: +1 (direct: {step['adjusted_score']:.2f})")

            lines.append(f"├── Weight: {step['weight']:.2f}")
            lines.append(f"├── Contribution: {step['adjusted_score']:.2f} × {step['weight']:.2f} = {step['contribution']:.3f}")
            lines.append(f"├── Verdict: {step['symbol']} {step['verdict']}")
            lines.append(f"└── Running Likelihood: {step['running_likelihood']*100:.1f}%")
            lines.append(f"    (sum={step['running_weighted_sum']:.3f} / weights={step['running_total_weight']:.2f})")
            lines.append("")

        # Final result
        lines.append(f"{'='*70}")
        lines.append(f"  FINAL LIKELIHOOD: {breakdown['final_likelihood']*100:.1f}%")
        lines.append(f"  INTERPRETATION: {breakdown['interpretation']}")
        lines.append(f"{'='*70}")

        return "\n".join(lines)

    def predict(self, question: str) -> dict:
        """
        Main prediction method: extract attributes and calculate per-segment likelihood.

        Args:
            question: The question or scenario to analyze

        Returns:
            Dict with question, attributes, and predictions for each segment
        """
        # Extract relevant attributes using LLM
        attrs = self.extract_attributes(question)

        # Calculate likelihood for each segment
        predictions = []
        for seg_id in sorted(self.profiles.keys()):
            profile = self.profiles[seg_id]
            likelihood = self.calculate_likelihood(profile, attrs)
            key_factors = self._get_key_factors(profile, attrs)
            decision_tree = self.get_decision_tree_breakdown(profile, attrs, self.segment_names[seg_id])
            representative_customer = self.get_representative_customer(seg_id, attrs, likelihood)

            predictions.append({
                "segment_id": seg_id,
                "segment": self.segment_names[seg_id],
                "size_pct": self.segment_sizes[seg_id],
                "likelihood": likelihood,
                "interpretation": self._interpret_likelihood(likelihood),
                "key_factors": key_factors,
                "decision_tree": decision_tree,
                "representative_customer": representative_customer
            })

        # Sort by likelihood descending
        predictions.sort(key=lambda x: x["likelihood"], reverse=True)

        return {
            "question": question,
            "attributes": attrs,
            "predictions": predictions
        }

    def predict_batch(self, questions: list) -> pd.DataFrame:
        """
        Process multiple questions and return results as DataFrame.

        Args:
            questions: List of questions to analyze

        Returns:
            DataFrame with one row per question-segment combination
        """
        rows = []

        for question in questions:
            result = self.predict(question)

            for pred in result["predictions"]:
                # Get top positive and negative factors
                positive = [f for f in pred["key_factors"] if f["impact"] > 0]
                negative = [f for f in pred["key_factors"] if f["impact"] < 0]

                top_positive = ", ".join([
                    f"{f['attribute']} ({f['score']:.2f})"
                    for f in positive[:3]
                ])
                top_negative = ", ".join([
                    f"{f['attribute']} ({f['score']:.2f})"
                    for f in negative[:3]
                ])

                rows.append({
                    "question": question,
                    "segment": pred["segment"],
                    "segment_size_pct": pred["size_pct"],
                    "likelihood": pred["likelihood"],
                    "interpretation": pred["interpretation"],
                    "top_positive_factors": top_positive,
                    "top_negative_factors": top_negative
                })

        return pd.DataFrame(rows)


def main():
    """Interactive CLI mode."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python likelihood_predictor.py <API_KEY>")
        print("\nInteractive mode: Enter questions to predict segment responses.")
        sys.exit(1)

    api_key = sys.argv[1]
    predictor = LikelihoodPredictor(api_key)

    print("\n" + "="*60)
    print("  Shopper Segment Response Predictor")
    print("="*60)
    print("\nType 'quit' to exit.\n")

    while True:
        question = input("> Enter question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nExtracting relevant attributes...")

        try:
            result = predictor.predict(question)

            # Display extracted attributes
            print("\nRELEVANT ATTRIBUTES:")
            for attr, info in result["attributes"].items():
                polarity_str = "+1" if info["polarity"] == 1 else "-1"
                print(f"  {attr:25s} weight={info['weight']:.2f}, polarity={polarity_str:>2s} ({info['reason']})")

            # Display predictions
            print("\nSEGMENT PREDICTIONS:")
            print("-" * 70)
            print(f"{'#':<3} {'Segment':<30} {'Likelihood':>12} {'Interpretation':<20}")
            print("-" * 70)

            for i, pred in enumerate(result["predictions"]):
                likelihood_pct = f"{pred['likelihood']*100:.1f}%"
                print(f"{i+1:<3} {pred['segment']:<30} {likelihood_pct:>12} {pred['interpretation']:<20}")

            print("-" * 70)

            # Ask if user wants to see decision tree
            print("\nEnter segment # to see decision tree (or press Enter to skip): ", end="")
            choice = input().strip()

            if choice.isdigit() and 1 <= int(choice) <= len(result["predictions"]):
                selected = result["predictions"][int(choice) - 1]
                tree_output = predictor.print_decision_tree(selected["decision_tree"])
                print(tree_output)

            print()

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
