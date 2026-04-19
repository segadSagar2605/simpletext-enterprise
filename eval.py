
from dotenv import load_dotenv
load_dotenv()
import httpx
import csv
import re
import os
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def call_ask_endpoint(question: str) -> dict:
    try:
        response=httpx.post(
            "http://127.0.0.1:8000/ask",
            json={"q": question},
            timeout=30.0
        )
        return response.json()
    except Exception as e:
        return {"error" : str(e)}

def gemini_as_judge(question: str, ground_truth: str, retrieved_context: str) -> int:
    try:
        import re
        context_snippet = retrieved_context[:4000]
        
        print(f"Context length being sent: {len(context_snippet)}")  # add this
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=f"You are a strict RAG evaluation expert.\n\n"
                     f"Question: {question}\n\n"
                     f"Expected answer: {ground_truth}\n\n"
                     f"Retrieved context: {context_snippet}\n\n"
                     f"Your task: Score whether the retrieved context contains enough information to correctly answer the question.\n\n"
                     f"Scoring rules:\n"
                     f"5 = Context fully answers the question with all required details\n"
                     f"4 = Context mostly answers the question, minor gaps\n"
                     f"3 = Context partially answers, significant information missing\n"
                     f"2 = Context has weak relevance, mostly off-topic\n"
                     f"1 = Context is almost entirely irrelevant\n"
                     f"0 = The answer does not exist anywhere in the retrieved context OR the question asks for information the documents clearly do not contain\n\n"
                     f"CRITICAL RULE: If the expected answer indicates the information does not exist in the system, "
                     f"or if the retrieved context contains documents that are related to the topic but do not actually "
                     f"contain the specific information being asked for, you MUST score 0. "
                     f"Do not reward retrieval of tangentially related content as a substitute for the actual answer.\n\n"
                     f"Reply with a SINGLE digit only: 0, 1, 2, 3, 4, or 5.",
            config={"temperature": 0}
        )
        
        raw = response.text.strip()
        print(f"Gemini raw: '{raw}'")
        
        match = re.search(r'[0-5]', raw)
        return int(match.group()) if match else 0
        
    except Exception as e:
        print(f"Judge error: {e}")
        return 0
         
def run_eval(questions: list) -> list:
    results = []
    for q in questions:
        # call call_ask_endpoint with q["question"]
        answer=call_ask_endpoint(q["question"])
        print(f"Q: {q['id']} | retrieved: {answer}") 
        # store q["id"], q["type"], q["question"], 
        # q["ground_truth"], and the retrieved_context
        score=gemini_as_judge(q["question"],q["ground_truth"],answer.get("retrieved_context", ""))
        result={
            "id":q["id"],
            "type":q["type"],
            "question":q["question"],
            "ground_truth":q["ground_truth"],
            "retrieved_context": answer.get("retrieved_context", ""),
            "rerank_scores": str(answer.get("rerank_scores", [])),
            "sources_found": answer.get("sources_found", 0),
            "score":score
        }
        results.append(result)
    return results

def save_to_csv(results: list):
    #import csv
    fieldnames = ["id", "type", "question", "ground_truth", "retrieved_context", "rerank_scores", "sources_found", "score"]
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to eval_results.csv")


QUESTIONS = [
    {"id": "Q1", "type": "simple", "question": "What is the total budget for Project Zicon?", "ground_truth": "INR 42.7 Cr baseline, INR 45.36 Cr including 15% contingency"},
    {"id": "Q2", "type": "simple", "question": "What species were recorded in the biodiversity census?", "ground_truth": "Silver-Tailed Fox (Vulnerable), Crested Newt (Endangered), Emerald Dragonfly (Stable)"},
    {"id": "Q3", "type": "simple", "question": "What are the maintenance requirements for the MRH-99 rotor hub?", "ground_truth": "Torque exactly 450 Nm, max 475 Nm, only X-Lube Synthetic Grade 5, petroleum greases prohibited"},
    {"id": "Q4", "type": "multi-hop", "question": "Which Project Zicon milestones are at risk given current status?", "ground_truth": "MS-03 Vendor Contract is AT RISK due to R-02, vendor flagged 3-week delay on migration tooling POC"},
    {"id": "Q5", "type": "multi-hop", "question": "Who owns the highest scoring risks in Project Zicon?", "ground_truth": "R-02 score 16 owned by Procurement, R-01 score 15 owned by CISO/Legal"},
    {"id": "Q6", "type": "cross-doc", "question": "What is the overall health of Project Zicon?", "ground_truth": "Overall AMBER downgraded from GREEN, Vendor Management RED, Architecture 65% done, budget fine, hard launch 15 Dec 2026 locked"},
    {"id": "Q7", "type": "negative", "question": "What is the quarterly revenue for Project Zicon?", "ground_truth": "Not in the system - no revenue data exists, only cost and budget data"},
    {"id": "Q8", "type": "negative", "question": "What is the fuel capacity of the aircraft?", "ground_truth": "Not in the system - aviation manual only covers MRH-99 rotor hub maintenance"},
    {"id": "Q9", "type": "negative", "question": "What is the headcount plan for Project Zicon?", "ground_truth": "Not in the system - no headcount data exists in any document"},
]

if __name__ == "__main__":
    print("Running SimpleText Eval Framework...")
    results = run_eval(QUESTIONS)
    save_to_csv(results)
    
    # quick summary
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.1f} / 5")
    for t in ["simple", "multi-hop", "cross-doc", "negative"]:
        subset = [r for r in results if r["type"] == t]
        avg = sum(r["score"] for r in subset) / len(subset)
        print(f"  {t}: {avg:.1f} / 5")



