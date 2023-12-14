import pytest
import pytest_html
from pathlib import Path
import os

def pytest_html_report_title(report):
    report.title = "Test report of test_dynamic_models"

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    extras = getattr(report, "extras", [])
    if report.when == "call":

        for i in range(5):
            extras.append(pytest_html.extras.png(f"C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/tests/test_dynamic_models/figure{i}.png"))
        xfail = hasattr(report, "wasxfail")
        if (report.skipped and xfail) or (report.failed and not xfail):
            # only add additional html on failure
            pass
        report.extras = extras