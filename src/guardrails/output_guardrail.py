from pathlib import Path

import yaml


# -------------------------------------------------
# Configuration path
# -------------------------------------------------

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "output_guardrail.yaml"
)


# -------------------------------------------------
# Custom exception
# -------------------------------------------------

class OutputGuardrailError(ValueError):
    """
    Raised when a generated response fails
    output guardrail validation.
    """

    def __init__(self, message, category=None, rule_id=None):
        super().__init__(message)

        self.category = category
        self.rule_id = rule_id


# -------------------------------------------------
# Output Guardrail
# -------------------------------------------------

class OutputGuardrail:

    def __init__(self, config_path=CONFIG_PATH):

        # Load output guardrail configuration.
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

        self.settings = self.config.get("settings", {})
        self.rules = self.config.get("rules", [])

        # Validate and load enabled rules.
        self.enabled_rules = []

        supported_rules = {
            "empty_response",
            "response_too_long",
            "trim_whitespace"
        }

        for rule in self.rules:

            if not rule.get("enabled", True):
                continue

            rule_id = rule.get("id")
            action = rule.get("action")

            # Ensure the configured rule is supported.
            if rule_id not in supported_rules:
                raise ValueError(
                    f"Unsupported output guardrail rule: {rule_id}"
                )

            # Validate action for each rule.
            expected_actions = {
                "empty_response": "reject",
                "response_too_long": "reject",
                "trim_whitespace": "cleanup"
            }

            if action != expected_actions[rule_id]:
                raise ValueError(
                    f"Invalid action '{action}' for rule '{rule_id}'."
                )

            self.enabled_rules.append(rule)

    # -------------------------------------------------
    # Validate generated response
    # -------------------------------------------------

    def validate(self, response):
        """
        Validate and clean the generated response.

        Returns:
            str: Validated response.

        Raises:
            OutputGuardrailError: If response validation fails.
        """

        # 1. Mandatory response type validation.
        # This check always runs, even if configurable
        # output guardrail rules are disabled.

        if not isinstance(response, str):

            raise OutputGuardrailError(
                "Generated response must be a string.",
                category="response_validation",
                rule_id="invalid_response_type"
            )

        # 2. If configurable rules are disabled,
        # return the response after mandatory type validation.

        if not self.settings.get("enabled", True):
            return response

        # 3. Execute configured output guardrail rules.

        for rule in self.enabled_rules:

            rule_id = rule["id"]
            category = rule.get(
                "category",
                "response_validation"
            )

            action = rule["action"]

            # -----------------------------------------
            # Rule 1: Empty response validation
            # -----------------------------------------

            if rule_id == "empty_response":

                if not response.strip():

                    raise OutputGuardrailError(
                        "The generated response is empty.",
                        category=category,
                        rule_id=rule_id
                    )

            # -----------------------------------------
            # Rule 2: Maximum response length
            # -----------------------------------------

            elif rule_id == "response_too_long":

                max_characters = self.settings.get(
                    "max_characters",
                    12000
                )

                if len(response) > max_characters:

                    raise OutputGuardrailError(
                        f"Generated response exceeds the maximum "
                        f"allowed length of {max_characters} characters.",
                        category=category,
                        rule_id=rule_id
                    )

            # -----------------------------------------
            # Rule 3: Whitespace cleanup
            # -----------------------------------------

            elif rule_id == "trim_whitespace":

                response = response.strip()

        # 4. Final validation after cleanup.

        if not response.strip():

            raise OutputGuardrailError(
                "No usable response remains after output validation.",
                category="response_validation",
                rule_id="empty_after_cleanup"
            )

        return response


# -------------------------------------------------
# Reusable instance and public function
# -------------------------------------------------

# Load configuration once when this module is imported.
output_guardrail = OutputGuardrail()


def validate_output(response):
    """
    Public function invoked by the RAG orchestrator.
    """
    return output_guardrail.validate(response)