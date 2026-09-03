from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# These evaluations can block deployment.
QUALITY_EVALUATIONS = [
    (
        "Retrieval Evaluation",
        "evaluation.run_eval",
    ),
    (
        "RBAC Evaluation",
        "evaluation.rbac_eval",
    ),
    (
        "No-Answer Evaluation",
        "evaluation.no_answer_eval",
    ),
    (
        "Generation Evaluation",
        "evaluation.generation_eval",
    ),
]

# Performance is measured, but does not block deployment yet.
INFORMATIONAL_EVALUATIONS = [
    (
        "Performance Evaluation",
        "evaluation.performance_eval",
    ),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class EvaluationRun:
    name: str
    module: str
    passed: bool
    duration_seconds: float
    blocking: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(
    title: str,
) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def run_evaluation(
    *,
    name: str,
    module: str,
    blocking: bool,
) -> EvaluationRun:
    """
    Run one evaluation module as a separate Python process.

    Using subprocess keeps each evaluator independent and
    preserves its existing SystemExit-based quality gate.
    """

    print_header(
        f"STARTING: {name}"
    )

    start = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
        ],
        check=False,
    )

    duration_seconds = (
        time.perf_counter()
        - start
    )

    passed = (
        result.returncode == 0
    )

    print()

    if passed:
        print(
            f"{name}: PASSED "
            f"({duration_seconds:.1f}s)"
        )
    else:
        print(
            f"{name}: FAILED "
            f"({duration_seconds:.1f}s)"
        )

    return EvaluationRun(
        name=name,
        module=module,
        passed=passed,
        duration_seconds=duration_seconds,
        blocking=blocking,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    results: list[EvaluationRun],
) -> None:
    print()
    print("#" * 80)
    print("ENTERPRISE RAG EVALUATION SUMMARY")
    print("#" * 80)

    for result in results:

        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        gate_type = (
            "QUALITY GATE"
            if result.blocking
            else "INFORMATIONAL"
        )

        print(
            f"{result.name:<28} "
            f"{status:<6} "
            f"{result.duration_seconds:>7.1f}s "
            f"[{gate_type}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    overall_start = (
        time.perf_counter()
    )

    results: list[
        EvaluationRun
    ] = []

    print_header(
        "ENTERPRISE RAG EVALUATION SUITE"
    )

    print(
        "Running deployment quality gates..."
    )

    # -----------------------------------------------------------------------
    # Blocking evaluations
    # -----------------------------------------------------------------------

    for (
        name,
        module,
    ) in QUALITY_EVALUATIONS:

        result = run_evaluation(
            name=name,
            module=module,
            blocking=True,
        )

        results.append(
            result
        )

        # Fail fast.
        #
        # Example:
        # if RBAC fails, there is no reason to continue
        # toward deployment.
        if not result.passed:

            print_summary(
                results
            )

            print()
            print("#" * 80)
            print(
                "DEPLOYMENT QUALITY GATE: FAILED"
            )
            print("#" * 80)

            print(
                f"{name} failed."
            )

            print(
                "Deployment must not continue."
            )

            raise SystemExit(1)

    # -----------------------------------------------------------------------
    # Informational evaluations
    # -----------------------------------------------------------------------

    print()
    print(
        "All quality gates passed."
    )

    print(
        "Running informational evaluations..."
    )

    for (
        name,
        module,
    ) in INFORMATIONAL_EVALUATIONS:

        result = run_evaluation(
            name=name,
            module=module,
            blocking=False,
        )

        results.append(
            result
        )

        # Performance failure is reported but
        # does not currently block deployment.

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    total_duration = (
        time.perf_counter()
        - overall_start
    )

    print_summary(
        results
    )

    informational_failures = [
        result
        for result in results
        if (
            not result.blocking
            and not result.passed
        )
    ]

    print()
    print("#" * 80)
    print(
        "DEPLOYMENT QUALITY GATE: PASSED"
    )
    print("#" * 80)

    print(
        "Retrieval          ✓"
    )

    print(
        "RBAC               ✓"
    )

    print(
        "No-answer          ✓"
    )

    print(
        "Generation quality ✓"
    )

    if informational_failures:
        print()
        print(
            "Warning: some informational "
            "evaluations failed:"
        )

        for result in (
            informational_failures
        ):
            print(
                f"- {result.name}"
            )

    print()
    print(
        f"Total evaluation time: "
        f"{total_duration:.1f}s"
    )


if __name__ == "__main__":
    main()