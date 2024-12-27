import cv2
import numpy as np

# Correct answers for 10 questions
correct_answers = ['A', 'B', 'C', 'B', 'D', 'C', 'C', 'B', 'C', 'C']

def preprocess_image(image_path):
    """
    Load and preprocess the OMR image by applying thresholding.
    :param image_path: Path to the input OMR image.
    :return: Binary thresholded image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_answers(thresh, num_questions=10, options_per_question=4):
    """
    Extract marked answers from the OMR sheet.
    :param thresh: Binary thresholded image.
    :param num_questions: Total number of questions.
    :param options_per_question: Number of options per question (A, B, C, D).
    :return: List of detected answers (e.g., ['A', 'B', 'C']).
    """
    detected_answers = []

    # Get dimensions of the thresholded image
    height, width = thresh.shape
    row_height = height // num_questions
    option_width = width // options_per_question

    for q in range(num_questions):
        # Extract the row region for each question
        row_start = q * row_height
        row_end = (q + 1) * row_height
        row_region = thresh[row_start:row_end, :]

        # Initialize variables for detecting the marked option
        max_fill = 0
        detected_option = None

        for o in range(options_per_question):
            option_start = o * option_width
            option_end = (o + 1) * option_width
            option_region = row_region[:, option_start:option_end]

            # Calculate the fill intensity (number of white pixels)
            fill_intensity = cv2.countNonZero(option_region)

            # Detect the most filled option
            if fill_intensity > max_fill:
                max_fill = fill_intensity
                detected_option = o

        # Map the detected option index to 'A', 'B', 'C', 'D'
        if detected_option is not None:
            detected_answers.append(chr(65 + detected_option))
        else:
            detected_answers.append('-')  # '-' indicates no answer marked

    return detected_answers

def check_answers(detected_answers, correct_answers):
    """
    Compare detected answers with the correct answers.
    :param detected_answers: List of detected answers.
    :param correct_answers: List of correct answers.
    """
    score = 0
    for i, (detected, correct) in enumerate(zip(detected_answers, correct_answers)):
        if detected == correct:
            score += 1
        print(f"Q{i + 1}: Your Answer: {detected} | Correct Answer: {correct} | {'Correct' if detected == correct else 'Wrong'}")

    print(f"\nFinal Score: {score}/{len(correct_answers)}")

def main(image_path):
    """
    Main function to process the OMR sheet and check answers.
    :param image_path: Path to the input OMR image.
    """
    # Preprocess the image
    thresh = preprocess_image(image_path)

    # Extract marked answers
    detected_answers = extract_answers(thresh)

    # Check answers and calculate the score
    print("\nChecking Answers:")
    check_answers(detected_answers, correct_answers)

    # Display the processed image
    cv2.imshow("Processed OMR Sheet", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your uploaded OMR image
image_path = 'omrim.jpeg'  # Replace with the path to your OMR sheet image
main(image_path)
