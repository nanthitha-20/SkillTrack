import os

with open('app/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update imports to include deliverable_service
old_imp = "from src.ml.e2e_demo_view import render_e2e_demo_page"
new_imp = """from src.ml.e2e_demo_view import render_e2e_demo_page
    from src.data.deliverable_service import (
        get_all_milestones,
        get_milestone_by_id,
        get_intern_deliverable_history,
        save_pdf_deliverable,
        record_mentor_deliverable_review,
        get_all_submissions_for_mentor,
        get_submission_by_id,
        get_cohort_deliverable_metrics,
        DeliverableValidationError
    )"""

assert old_imp in text, "old_imp not found"
text = text.replace(old_imp, new_imp, 1)

# 2. Add fallback declarations in except block
old_fallback = "render_e2e_demo_page = None"
new_fallback = """render_e2e_demo_page = None
    get_all_milestones = None
    get_milestone_by_id = None
    get_intern_deliverable_history = None
    save_pdf_deliverable = None
    record_mentor_deliverable_review = None
    get_all_submissions_for_mentor = None
    get_submission_by_id = None
    get_cohort_deliverable_metrics = None
    DeliverableValidationError = Exception"""

assert old_fallback in text, "old_fallback not found"
text = text.replace(old_fallback, new_fallback, 1)

# Write back
with open('app/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Step 1 (deliverable imports) applied successfully!")
