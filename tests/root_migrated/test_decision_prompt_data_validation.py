#!/usr/bin/env python3
"""
Tests for data validation integration in decision_prompt.txt

Verifies that the decision prompt correctly documents data validation tools
and includes them in pipeline planning patterns.
"""

import re
from pathlib import Path


class TestDecisionPromptDataValidation:
    """Test suite for data validation in decision prompt."""

    def __init__(self):
        self.prompt_path = Path("prompts/decision_prompt.txt")
        self.prompt_content = self.prompt_path.read_text()
        self.results = {"passed": 0, "failed": 0}

    def record_result(self, passed: bool, test_name: str, details: str = ""):
        """Record test result."""
        if passed:
            self.results["passed"] += 1
            print(f"  \033[92m\u2713 PASS\033[0m {test_name}")
        else:
            self.results["failed"] += 1
            print(f"  \033[91m\u2717 FAIL\033[0m {test_name}")
            if details:
                print(f"       \033[93m{details}\033[0m")

    def test_data_quality_section_exists(self):
        """Test that Data Quality & Validation section exists."""
        test_name = "Data Quality & Validation section exists"
        passed = "### Data Quality & Validation" in self.prompt_content
        self.record_result(passed, test_name, "Section header not found")

    def test_tool_count_updated(self):
        """Test that tool count is updated to include data validation tools."""
        test_name = "Tool count includes data validation tools (46 total)"
        passed = "## Available Tools (46 total)" in self.prompt_content
        self.record_result(passed, test_name, "Expected '46 total' in tools count")

    def test_validate_dataset_tool_documented(self):
        """Test that validate_dataset tool is documented."""
        test_name = "validate_dataset tool documented"
        passed = "| `validate_dataset`" in self.prompt_content
        self.record_result(passed, test_name, "validate_dataset not in tool table")

    def test_create_expectation_suite_tool_documented(self):
        """Test that create_expectation_suite tool is documented."""
        test_name = "create_expectation_suite tool documented"
        passed = "| `create_expectation_suite`" in self.prompt_content
        self.record_result(passed, test_name, "create_expectation_suite not in tool table")

    def test_check_data_quality_tool_documented(self):
        """Test that check_data_quality tool is documented."""
        test_name = "check_data_quality tool documented"
        passed = "| `check_data_quality`" in self.prompt_content
        self.record_result(passed, test_name, "check_data_quality not in tool table")

    def test_profile_dataset_tool_documented(self):
        """Test that profile_dataset tool is documented."""
        test_name = "profile_dataset tool documented"
        passed = "| `profile_dataset`" in self.prompt_content
        self.record_result(passed, test_name, "profile_dataset not in tool table")

    def test_detect_anomalies_tool_documented(self):
        """Test that detect_anomalies tool is documented."""
        test_name = "detect_anomalies tool documented"
        passed = "| `detect_anomalies`" in self.prompt_content
        self.record_result(passed, test_name, "detect_anomalies not in tool table")

    def test_validate_schema_tool_documented(self):
        """Test that validate_schema tool is documented."""
        test_name = "validate_schema tool documented"
        passed = "| `validate_schema`" in self.prompt_content
        self.record_result(passed, test_name, "validate_schema not in tool table")

    def test_compare_distributions_tool_documented(self):
        """Test that compare_distributions tool is documented."""
        test_name = "compare_distributions tool documented"
        passed = "| `compare_distributions`" in self.prompt_content
        self.record_result(passed, test_name, "compare_distributions not in tool table")

    def test_dependency_order_includes_data_validation(self):
        """Test that dependency order mentions data validation."""
        test_name = "Dependency order includes data validation step"
        passed = "**Data Validation**" in self.prompt_content
        self.record_result(passed, test_name, "Data Validation not in dependency order")

    def test_full_pipeline_includes_validate_dataset(self):
        """Test that full pipeline pattern includes validate_dataset."""
        test_name = "Full pipeline pattern includes validate_dataset"
        # Check that validate_dataset appears in the Full Pipeline Setup section
        full_pipeline_match = re.search(
            r"\*\*Full Pipeline Setup:\*\*.*?```(.*?)```", self.prompt_content, re.DOTALL
        )
        passed = full_pipeline_match and "validate_dataset" in full_pipeline_match.group(1)
        self.record_result(passed, test_name, "validate_dataset not in Full Pipeline Setup pattern")

    def test_data_validation_patterns_section_exists(self):
        """Test that Data Validation Patterns section exists."""
        test_name = "Data Validation Patterns section exists"
        passed = "**Data Validation Patterns:**" in self.prompt_content
        self.record_result(passed, test_name, "Data Validation Patterns section not found")

    def test_basic_data_quality_check_pattern(self):
        """Test that basic data quality check pattern exists."""
        test_name = "Basic Data Quality Check pattern exists"
        passed = "*Basic Data Quality Check:*" in self.prompt_content
        self.record_result(passed, test_name, "Basic Data Quality Check pattern not found")

    def test_full_data_validation_pattern(self):
        """Test that full data validation with expectations pattern exists."""
        test_name = "Full Data Validation with Expectations pattern exists"
        passed = "*Full Data Validation with Expectations:*" in self.prompt_content
        self.record_result(passed, test_name, "Full Data Validation pattern not found")

    def test_data_drift_detection_pattern(self):
        """Test that data drift detection pattern exists."""
        test_name = "Data Drift Detection pattern exists"
        passed = "*Data Drift Detection:*" in self.prompt_content
        self.record_result(passed, test_name, "Data Drift Detection pattern not found")

    def test_schema_validation_pattern(self):
        """Test that schema validation pattern exists."""
        test_name = "Schema Validation pattern exists"
        passed = "*Schema Validation:*" in self.prompt_content
        self.record_result(passed, test_name, "Schema Validation pattern not found")

    def test_pre_training_validation_pattern(self):
        """Test that pre-training data validation pattern exists."""
        test_name = "Pre-Training Data Validation pattern exists"
        passed = "*Pre-Training Data Validation:*" in self.prompt_content
        self.record_result(passed, test_name, "Pre-Training Data Validation pattern not found")

    def test_important_rule_data_validation(self):
        """Test that important rules include data validation requirement."""
        test_name = "Important rules include data validation requirement"
        passed = "Validate data before training" in self.prompt_content
        self.record_result(passed, test_name, "Data validation rule not in Important Rules")

    def test_tool_args_documented(self):
        """Test that data validation tools have key args documented."""
        test_name = "Data validation tools have key args documented"
        # Check for key args in tool descriptions
        checks = [
            "dataset_path" in self.prompt_content,
            "expectations" in self.prompt_content,
            "fail_on_error" in self.prompt_content,
            "outlier_threshold" in self.prompt_content,
        ]
        passed = all(checks)
        self.record_result(passed, test_name, "Some key args not documented")

    def test_seven_data_quality_tools(self):
        """Test that exactly 7 data quality tools are documented."""
        test_name = "Exactly 7 data quality tools documented"
        # Check the header says 7 tools
        passed = "### Data Quality & Validation (7 tools)" in self.prompt_content
        self.record_result(passed, test_name, "Section header should say '7 tools'")

    def run_all_tests(self):
        """Run all tests."""
        print("\n\033[1m\033[96m" + "=" * 60 + "\033[0m")
        print(
            "\033[1m\033[96m" + "Testing Data Validation in Decision Prompt".center(60) + "\033[0m"
        )
        print("\033[1m\033[96m" + "=" * 60 + "\033[0m\n")

        # Run all test methods
        test_methods = [
            self.test_data_quality_section_exists,
            self.test_tool_count_updated,
            self.test_validate_dataset_tool_documented,
            self.test_create_expectation_suite_tool_documented,
            self.test_check_data_quality_tool_documented,
            self.test_profile_dataset_tool_documented,
            self.test_detect_anomalies_tool_documented,
            self.test_validate_schema_tool_documented,
            self.test_compare_distributions_tool_documented,
            self.test_dependency_order_includes_data_validation,
            self.test_full_pipeline_includes_validate_dataset,
            self.test_data_validation_patterns_section_exists,
            self.test_basic_data_quality_check_pattern,
            self.test_full_data_validation_pattern,
            self.test_data_drift_detection_pattern,
            self.test_schema_validation_pattern,
            self.test_pre_training_validation_pattern,
            self.test_important_rule_data_validation,
            self.test_tool_args_documented,
            self.test_seven_data_quality_tools,
        ]

        for test in test_methods:
            try:
                test()
            except Exception as e:
                self.record_result(False, test.__name__, str(e))

        # Print summary
        print("\n" + "=" * 60)
        total = self.results["passed"] + self.results["failed"]
        print(f"\033[1mResults: {self.results['passed']}/{total} tests passed\033[0m")

        if self.results["failed"] > 0:
            print(f"\033[91m{self.results['failed']} tests failed\033[0m")
            return 1
        else:
            print("\033[92mAll tests passed!\033[0m")
            return 0


def main():
    """Run the test suite."""
    test_suite = TestDecisionPromptDataValidation()
    return test_suite.run_all_tests()


if __name__ == "__main__":
    exit(main())
