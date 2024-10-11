import json
from docx import Document


def extract_mcq_from_docx(docx_path):
    # Open the docx file
    doc = Document(docx_path)

    mcq_list = []
    current_question = None
    current_answers = {}
    correct_answer = None
    answer_index = "A"

    for paragraph in doc.paragraphs:
        # Check if the paragraph is a question (usually starts without bullet or numbering)
        if paragraph.style.name == "Normal" and "?" in paragraph.text:
            if current_question:
                # If we encounter a new question, save the previous one
                mcq_list.append(
                    {
                        "question": current_question.strip(),
                        "answers": current_answers,
                        "correct_answer": correct_answer,
                    }
                )
            # Start a new question
            current_question = paragraph.text
            current_answers = {}
            correct_answer = None
            answer_index = "A"


        elif paragraph.text.strip().startswith(("A", "B", "C", "D", "E")):
            answer_text = paragraph.text[
                2:
            ].strip()

            # Check if the answer is bold (meaning it's the correct answer)
            is_bold = any(run.bold for run in paragraph.runs)
            if is_bold:
                correct_answer = answer_index

            # Store the answer in the dictionary
            current_answers[answer_index] = answer_text
            answer_index = chr(
                ord(answer_index) + 1
            )

    # Append the last question after finishing the loop
    if current_question:
        mcq_list.append(
            {
                "question": current_question.strip(),
                "answers": current_answers,
                "correct_answer": correct_answer,
            }
        )

    return mcq_list


def save_as_json(data, output_path):
    # Save the extracted MCQ data into a JSON file
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


# Path to your docx file
docx_file_path = "data/test-bank.docx"
output_json_path = "data/json/test-bank.json"

# Extract MCQs and save to JSON
mcq_data = extract_mcq_from_docx(docx_file_path)
save_as_json(mcq_data, output_json_path)

print(f"MCQs extracted and saved to {output_json_path}")
