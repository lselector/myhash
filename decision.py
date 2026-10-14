from openai import OpenAI

client = OpenAI()  # Reads OPENAI_API_KEY

decision = client.decisions.create(
    model="gpt-6-luna",
    input="I was charged twice for my order. Please refund the duplicate.",
    questions=[
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this request?",
            "choices": [
                {
                    "value": "billing",
                    "description": "Payments, invoices, and refunds.",
                },
                {
                    "value": "technical",
                    "description": "Problems using the product.",
                },
                {
                    "value": "other",
                    "description": "Requests outside the listed categories.",
                },
            ],
        }
    ],
)

answer = decision.answers[0]

if answer.type == "refusal":
    print("Decision refused:", answer.name)
elif answer.type == "choice":
    print("Route:", answer.choice)
    print("Confidence:", answer.confidence)
    print("Probabilities:", answer.probabilities)

# Inspect the complete response, including token usage:
print(decision.model_dump_json(indent=2))

