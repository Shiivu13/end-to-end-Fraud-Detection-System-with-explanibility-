# pipelines/genai_explainer.py
# GenAI explanation with graceful fallback

def genai_explain(technical_explanation: str) -> str:
    """
    Fallback GenAI explainer.
    If API is not available, returns a safe business explanation.
    """

    return (
        "This transaction was flagged as potentially fraudulent because "
        "it shows patterns and behaviors that are commonly associated with "
        "known fraud cases."
    )


if __name__ == "__main__":
    test_explanation = (
        "The model flagged this transaction as fraud mainly because "
        "V14 increased fraud risk, V17 increased fraud risk, "
        "and log_amount increased fraud risk."
    )

    print("\nGenAI Business Explanation (Fallback Mode):\n")
    print(genai_explain(test_explanation))