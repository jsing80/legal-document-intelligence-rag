from guardrails.output_guardrail import (
    validate_output,
    OutputGuardrailError
)


test_responses = [
    "The agreement can be terminated with 30 days' notice.",

    "   The agreement has a one-year initial term.   ",

    "",

    "     ",

    None,

    123,

    "Legal response. " * 1000
]


for response in test_responses:

    print("\nInput:", str(response)[:100])

    try:
        validated_response = validate_output(response)

        print("Status: ALLOWED")
        print("Validated response:", validated_response[:150])

    except OutputGuardrailError as error:

        print("Status: REJECTED")
        print("Category:", error.category)
        print("Rule:", error.rule_id)
        print("Reason:", error)