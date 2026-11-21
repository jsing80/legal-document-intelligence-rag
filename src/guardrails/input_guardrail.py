import re
import unicodedata
from pathlib import Path

import yaml


# -------------------------------------------------
# Configuration path
# -------------------------------------------------

# input_guardrail.py is under src/guardrails/
# YAML configuration is under src/config/
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "input_guardrail.yaml"
)


# -------------------------------------------------
# Custom exception
# -------------------------------------------------

class InputGuardrailError(ValueError):
    """
    Raised when user input fails a guardrail check.
    """

    def __init__(self, message, category=None, rule_id=None):
        super().__init__(message)

        self.category = category
        self.rule_id = rule_id


# -------------------------------------------------
# Input Guardrail
# -------------------------------------------------

class InputGuardrail:

    def __init__(self, config_path=CONFIG_PATH):

        # Load guardrail configuration from YAML.
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

        self.settings = self.config.get("settings", {})
        self.rules = self.config.get("rules", [])

        # Compile enabled rules once during initialization.
        # This avoids recompiling regex patterns for every query.
        self.compiled_rules = []

        for rule in self.rules:

            if not rule.get("enabled", True):
                continue

            rule_id = rule.get("id", "unknown_rule")
            action = rule.get("action")

            # Only explicitly supported actions are allowed.
            if action not in {"reject", "redact", "cleanup"}:
                raise ValueError(
                    f"Unsupported action '{action}' "
                    f"in guardrail rule '{rule_id}'."
                )

            patterns = []

            for pattern in rule.get("patterns", []):

                try:
                    patterns.append(re.compile(pattern))

                except re.error as error:
                    raise ValueError(
                        f"Invalid regex in guardrail rule "
                        f"'{rule_id}': {error}"
                    ) from error

            self.compiled_rules.append(
                {
                    "id": rule_id,
                    "category": rule.get("category", "uncategorized"),
                    "description": rule.get("description", ""),
                    "action": action,
                    "replacement": rule.get("replacement", ""),
                    "patterns": patterns
                }
            )

    # -------------------------------------------------
    # 1. Input normalization
    # -------------------------------------------------

    def _normalize_input(self, text):
        """
        Normalize the input before applying guardrail rules.

        This handles formatting and selected invisible
        characters. It does not attempt to decode arbitrary
        encoded instructions.
        """

        normalization = self.settings.get("normalization", {})

        # Normalize equivalent Unicode representations.
        if normalization.get("unicode_nfkc", True):
            text = unicodedata.normalize("NFKC", text)

        # Remove null bytes.
        if normalization.get("remove_null_bytes", True):
            text = text.replace("\x00", "")

        # Remove selected zero-width Unicode characters.
        if normalization.get("remove_zero_width_characters", True):
            text = re.sub(
                r"[\u200B-\u200D\uFEFF]",
                "",
                text
            )

        # Collapse repeated spaces, tabs and newlines.
        if normalization.get("collapse_whitespace", True):
            text = re.sub(r"\s+", " ", text).strip()

        return text

    # -------------------------------------------------
    # 2. Input validation and rule execution
    # -------------------------------------------------

    def validate(self, user_input):
        """
        Validate and sanitize the user's question.

        Returns:
            str: Normalized and sanitized user question.

        Raises:
            InputGuardrailError: If a rejection rule is triggered.
        """

        # Check input type regardless of whether configurable
        # security rules are enabled.
        if not isinstance(user_input, str):

            raise InputGuardrailError(
                "Input must be a text string.",
                category="input_validation",
                rule_id="invalid_input_type"
            )

        # Allow configurable rules to be disabled.
        # Basic type validation above is still enforced.
        if not self.settings.get("enabled", True):
            return user_input

        # Normalize input before applying security checks.
        user_input = self._normalize_input(user_input)

        # Check for empty input.
        if not user_input:

            raise InputGuardrailError(
                "Please enter a question.",
                category="input_validation",
                rule_id="empty_input"
            )

        # Check maximum input length.
        max_characters = self.settings.get("max_characters", 4000)

        if len(user_input) > max_characters:

            raise InputGuardrailError(
                f"Input exceeds the maximum allowed length "
                f"of {max_characters} characters.",
                category="input_validation",
                rule_id="input_too_long"
            )

        # -------------------------------------------------
        # Execute configurable security rules.
        #
        # Rules are evaluated in YAML order.
        # Reject stops processing immediately.
        # Redact and cleanup modify the input.
        # -------------------------------------------------

        for rule in self.compiled_rules:

            for pattern in rule["patterns"]:

                match = pattern.search(user_input)

                if not match:
                    continue

                action = rule["action"]

                # Action 1: Reject the request.
                if action == "reject":

                    raise InputGuardrailError(
                        f"Input rejected: {rule['description']}",
                        category=rule["category"],
                        rule_id=rule["id"]
                    )

                # Action 2: Redact matching sensitive information.
                elif action == "redact":

                    user_input = pattern.sub(
                        rule["replacement"],
                        user_input
                    )

                # Action 3: Remove or clean matching content.
                elif action == "cleanup":

                    user_input = pattern.sub(
                        rule["replacement"],
                        user_input
                    )

        # Ensure sanitization has not left an empty question.
        if not user_input.strip():

            raise InputGuardrailError(
                "No usable question remains after input sanitization.",
                category="input_validation",
                rule_id="empty_after_sanitization"
            )

        return user_input


# -------------------------------------------------
# Reusable instance and public function
# -------------------------------------------------

# Load configuration and compile patterns once when
# this module is first imported.
input_guardrail = InputGuardrail()


def validate_input(user_input):
    """
    Public function called by the RAG orchestrator.
    """
    return input_guardrail.validate(user_input)