
import os
import io
import json
import re

from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv
from openai import OpenAI

import pandas as pd
from docx import Document
from PyPDF2 import PdfReader

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# Shared teacher access password
ACCESS_PASSWORD = "teacheraccess"


def read_file_to_text(file_storage):
    filename = (file_storage.filename or "").lower()
    raw_bytes = file_storage.read()

    def as_io():
        return io.BytesIO(raw_bytes)

    if filename.endswith((
        ".txt", ".py", ".java", ".cpp", ".c", ".md", ".html", ".json"
    )):
        try:
            return raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return raw_bytes.decode("latin-1", errors="ignore")

    elif filename.endswith(".csv"):
        df = pd.read_csv(as_io())
        return df.to_string(index=False)

    elif filename.endswith(".xlsx"):
        df = pd.read_excel(as_io())
        return df.to_string(index=False)

    elif filename.endswith(".docx"):
        doc = Document(as_io())
        return "\n".join(p.text for p in doc.paragraphs)

    elif filename.endswith(".pdf"):
        reader = PdfReader(as_io())
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)

    else:
        try:
            return raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return raw_bytes.decode("latin-1", errors="ignore")


def parse_rubric_to_json(rubric_text, model="gpt-4.1-mini"):
    system_prompt = (
        "You are an expert at transforming teacher rubrics into structured JSON. "
        "You must ONLY return valid JSON with NO extra commentary. "
        "Follow this schema strictly: a list of objects with keys: "
        "'criterion', 'description', 'points', 'requirements'. "
        "If the rubric does not have explicit points, infer a reasonable point "
        "value for each."
    )

    user_prompt = f"""RUBRIC TEXT:
{rubric_text}

TASK:
1. Read the rubric.
2. Break it into criteria.
3. For each criterion, create:
   - "criterion": short title
   - "description": longer explanation (if available)
   - "points": integer points for this criterion
   - "requirements": list of expectations
4. Return ONLY a JSON array (no markdown, no backticks, no explanation)."""  # noqa: E501

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    try:
        raw_text = response.output[0].content[0].text
        rubric_json = json.loads(raw_text)
        if not isinstance(rubric_json, list):
            rubric_json = [rubric_json]
        return rubric_json
    except Exception:
        fallback = [
            {
                "criterion": "Overall Rubric",
                "description": "Entire rubric treated as one criterion.",
                "points": 100,
                "requirements": [rubric_text.strip()[:1000]],
            }
        ]
        return fallback


def grade_with_rubric_json(rubric_json, student_text, model="gpt-4.1-mini"):
    rubric_json_str = json.dumps(rubric_json, indent=2)

    system_prompt = (
        "You are an encouraging, supportive computer science teacher who grades fairly "
        "using a structured rubric. Your tone is positive, balanced, and growth-oriented. "
        "You award partial credit with medium generosity when there is clear evidence of "
        "partial understanding or effort. You still respect the rubric and do not give "
        "full credit when requirements are missing, but you avoid harsh language. "
        "You always highlight strengths first, then gently suggest improvements. "
        "Write in clear paragraphs only. Do NOT use bullet points, dashes, asterisks, "
        "or numbered lists. Do NOT use markdown formatting. "
        "You must keep the student-facing summary separate from the teacher report."
    )

    user_prompt = f"""RUBRIC (JSON)
----------------
{rubric_json_str}

STUDENT SUBMISSION
------------------
{student_text}

YOUR TASK
---------
1. For EACH rubric item:
   Write a block in this exact structure:

   Criterion: <criterion name> (X points)
   Score: Y/X
   Explanation:
   <one or more sentences that first describe what the student did well, then gently describe what can be improved, in a supportive tone>

2. After you have graded all rubric items, write an overall teacher-only evaluation paragraph in this format:

Overall Teacher Comment:
<four to six sentences written for the teacher, with an encouraging but honest evaluation of the work. Mention strengths and specific areas to develop.>

3. Finally, at the very end, write a student-facing summary in this format:

Student Summary:
<three to five sentences written directly to the student in a neutral, encouraging tone. Focus on effort, progress, and one or two clear next steps.>

FORMAT RULES
------------
- Use only plain text.
- Do not use bullet points, hyphens, or numbered lists.
- Do not include the Student Summary text anywhere except after the 'Student Summary:' label at the very end.
- Do not restate the score line inside the Student Summary.
"""  # noqa: E501

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    try:
        output_block = response.output[0].content[0].text
    except Exception:
        output_block = str(response)

    return output_block


def extract_scores(feedback_text):
    pattern = r"Score:\s*(\d+)\s*/\s*(\d+)"
    matches = re.findall(pattern, feedback_text)
    total_earned = 0
    total_possible = 0
    for earned, possible in matches:
        total_earned += int(earned)
        total_possible += int(possible)
    return total_earned, total_possible


def extract_student_summary(feedback_text):
    marker = "Student Summary:"
    idx = feedback_text.find(marker)
    if idx == -1:
        return (
            "You completed meaningful work and demonstrated growing skill. "
            "Keep practicing and refining your logic, and use this feedback as a guide "
            "for your next draft."
        )
    student_part = feedback_text[idx + len(marker):].strip()
    return student_part


def strip_student_summary(feedback_text):
    marker = "Student Summary:"
    idx = feedback_text.find(marker)
    if idx == -1:
        return feedback_text.strip()
    return feedback_text[:idx].strip()


@app.route("/", methods=["GET", "POST"])
def access_gate():
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if password == ACCESS_PASSWORD:
            session["is_authenticated"] = True
            return redirect(url_for("grade"))
    return render_template("login.html")


@app.route("/grade", methods=["GET", "POST"])
def grade():
    if not session.get("is_authenticated"):
        return redirect(url_for("access_gate"))

    feedback = None
    summary_text = None
    error = None
    rubric_loaded = bool(session.get("rubric_text"))

    if request.method == "POST":
        rubric_text = (request.form.get("rubric_text") or "").strip()
        student_text = (request.form.get("student_text") or "").strip()

        rubric_file = request.files.get("rubric_file")
        student_file = request.files.get("student_file")

        try:
            if rubric_file and rubric_file.filename:
                rubric_file.stream.seek(0)
                rubric_text = read_file_to_text(rubric_file)

            if student_file and student_file.filename:
                student_file.stream.seek(0)
                student_text = read_file_to_text(student_file)
        except Exception as e:
            error = f"Error reading uploaded files: {e}"

        if not error:
            # Rubric caching: only update when teacher explicitly supplies a rubric
            if rubric_text:
                session["rubric_text"] = rubric_text

            rubric_text_cached = session.get("rubric_text", "")

            if not rubric_text_cached:
                error = "Please provide a rubric (either text or file)."
            elif not student_text:
                error = "Please provide student work (either text or file)."
            else:
                try:
                    rubric_json = parse_rubric_to_json(rubric_text_cached)
                    full_feedback = grade_with_rubric_json(rubric_json, student_text)

                    # Separate teacher report and student summary
                    earned, possible = extract_scores(full_feedback)
                    if possible > 0:
                        percent = round((earned / possible) * 100)
                        score_line = f"Your Score: {earned} / {possible} ({percent}%)"
                    else:
                        score_line = "Your Score: N/A"

                    student_summary = extract_student_summary(full_feedback)
                    summary_text = f"{score_line}\n\n{student_summary}".strip()

                    feedback = strip_student_summary(full_feedback)
                    rubric_loaded = True
                except Exception as e:
                    error = f"Error during AI grading: {e}"

    return render_template(
        "index.html",
        feedback=feedback,
        summary_text=summary_text,
        error=error,
        rubric_loaded=rubric_loaded,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
