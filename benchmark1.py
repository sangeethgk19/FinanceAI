import requests
import time
import psutil
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ─── MODELS TO BENCHMARK ───
MODELS = ["mistral", "llama3.2", "phi3:mini"]

# ─── 20 FINANCE BENCHMARK QUESTIONS ───
QUESTIONS = [
    # Category 1 — Factual Financial Knowledge
    "What is the difference between Tier 1 and Tier 2 capital under the Basel III framework?",
    "What is a credit default swap and how does it function as a risk management tool?",
    "What is the difference between systematic risk and unsystematic risk in investment portfolios?",
    "What is the repo rate and how does it influence commercial bank lending rates?",
    "What is the yield curve and what does an inverted yield curve typically signal about the economy?",

    # Category 2 — Regulatory and Compliance
    "What are the FCA's core consumer duty outcomes that firms must deliver under the Consumer Duty rules?",
    "Under UK GDPR, what obligations does a financial firm have when using a third-party processor to handle client data?",
    "What is the Market Abuse Regulation and what types of behaviour does it prohibit in UK financial markets?",
    "What are the key capital adequacy requirements that UK banks must maintain under the Basel III framework?",
    "What is the FCA's definition of a vulnerable customer and what obligations does it place on financial firms?",

    # Category 3 — Financial Reasoning
    "A bank reports a Liquidity Coverage Ratio of 85%. What does this indicate and what steps should the bank take?",
    "A portfolio has a beta of 1.4. What does this mean for the portfolio's expected behaviour during a market decline of 10%?",
    "A company has a debt-to-equity ratio of 3.5. What are the implications for investors and creditors assessing financial risk?",
    "An investment fund has a Sharpe ratio of 0.4 compared to a benchmark fund with 1.2. What does this tell you about risk-adjusted performance?",
    "A central bank raises interest rates by 0.5%. Walk through the chain of effects on mortgage holders, bond prices, and currency value.",

    # Category 4 — Plain Language Explanation
    "Explain what quantitative easing is to a client who has no financial background.",
    "Explain what diversification means in investment and why it reduces risk, using a simple everyday analogy.",
    "A client asks what the difference is between a stocks and shares ISA and a cash ISA. Explain clearly which might suit them better.",
    "Explain to a small business owner what AML checks are and why their bank requires them.",
    "A first-time investor asks what the difference is between a bond and a share. Explain clearly in simple terms."
]

# ─── CATEGORY LABELS ───
CATEGORIES = (
    ["Factual Financial Knowledge"] * 5 +
    ["Regulatory and Compliance"] * 5 +
    ["Financial Reasoning"] * 5 +
    ["Plain Language Explanation"] * 5
)

# ─── GEMINI JUDGE FUNCTION ───
def score_answer(question, answer):
    """Send question + answer to Gemini Flash for automatic scoring"""
    
    judge_model = genai.GenerativeModel("gemini-1.5-flash")
    
    scoring_prompt = f"""You are evaluating the quality of a financial AI assistant's answer.

Question: {question}

Answer: {answer}

Score this answer from 1 to 5 using this rubric:
5 = Complete, accurate, clearly explained — a finance professional would find it genuinely useful
4 = Correct but missing some detail or clarity
3 = Partially correct or incomplete
2 = Vague, unhelpful, or mostly wrong
1 = Incorrect, irrelevant, or hallucinated

Reply with ONLY a single number between 1 and 5. Nothing else. No explanation."""

    try:
        response = judge_model.generate_content(scoring_prompt)
        score_text = response.text.strip()
        score = int(score_text[0])
        if score < 1 or score > 5:
            score = 3
        return score
    except Exception as e:
        print(f"Scoring error: {e}")
        return 3  # Default to middle if something goes wrong

# ─── MAIN BENCHMARK LOOP ───
def run_benchmark():
    results = []
    total = len(MODELS) * len(QUESTIONS)
    count = 0

    print("=" * 50)
    print("FinanceAI Benchmark Starting")
    print(f"Models: {MODELS}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Total answers to generate: {total}")
    print("=" * 50)

    for model in MODELS:
        print(f"\nStarting model: {model.upper()}")
        print("-" * 30)

        for i, question in enumerate(QUESTIONS):
            count += 1
            category = CATEGORIES[i]

            print(f"[{count}/{total}] {model} — Q{i+1}: {question[:50]}...")

            # Measure RAM before
            ram_before = psutil.virtual_memory().used
            start = time.time()

            try:
                # Send question to FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"text": question, "model": model},
                    timeout=300
                ).json()

                duration = round(time.time() - start, 2)
                ram_used = round(
                    (psutil.virtual_memory().used - ram_before) / (1024 * 1024), 2
                )
                answer = response["answer"]

                print(f"  Time: {duration}s | RAM: {ram_used}MB")
                print(f"  Scoring with Gemini...")

                # Score with Gemini
                quality_score = score_answer(question, answer)
                print(f"  Quality score: {quality_score}/5")

                results.append({
                    "model": model,
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "seconds": duration,
                    "ram_mb": ram_used,
                    "quality_score": quality_score
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "model": model,
                    "category": category,
                    "question": question,
                    "answer": f"ERROR: {e}",
                    "seconds": 0,
                    "ram_mb": 0,
                    "quality_score": 0
                })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results_scored.csv", index=False)

    print("\n" + "=" * 50)
    print("Benchmark Complete!")
    print(f"Results saved to results_scored.csv")
    print(f"Total answers: {len(results)}")
    print("\nAverage scores per model:")
    summary = df.groupby("model")[["seconds", "quality_score"]].mean().round(2)
    print(summary)
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()