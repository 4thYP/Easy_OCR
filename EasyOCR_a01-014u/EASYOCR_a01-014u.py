import re
import easyocr
from PIL import Image
from pdf2image import convert_from_path
from difflib import SequenceMatcher
import docx
import os
import numpy as np
import math
from collections import Counter
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download NLTK data if needed
nltk.download('punkt', quiet=True)

# ---------- OCR + IO ----------

def ocr_pdf(pdf_path, lang="en", dpi=350, first_page=None, last_page=None):
    """
    Perform OCR on PDF using EasyOCR
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader([lang])

    pages = convert_from_path(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
    page_texts = []

    for i, page in enumerate(pages, 1):
        # Convert PIL image to numpy array
        img_array = np.array(page)

        # Perform OCR
        results = reader.readtext(img_array, detail=0, paragraph=True)

        # Combine all text blocks
        page_text = "\n".join(results)
        page_texts.append(page_text)

        print(f"[debug] OCR'd Page {i}: {len(page_text)} chars, {len(results)} text blocks")

    return page_texts

def read_docx(file_path):
    """Read text from a Word document"""
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

# ---------- Preprocessing ----------

def preprocess_text(text):
    """
    Preprocess text by removing extra whitespace, normalizing case,
    and removing special characters for better comparison
    """
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()\-]', '', text)
    return text.strip()

# ---------- Character-level Metrics ----------

def calculate_character_metrics(ocr_text, ground_truth):
    """
    Calculate character-level error metrics between OCR text and ground truth
    """
    # Preprocess both texts
    ocr_clean = preprocess_text(ocr_text)
    gt_clean = preprocess_text(ground_truth)

    # Calculate similarity ratio
    matcher = SequenceMatcher(None, ocr_clean, gt_clean)
    similarity_ratio = matcher.ratio()

    # Calculate error rate
    error_rate = (1 - similarity_ratio) * 100

    # Find matching blocks and differences
    matching_blocks = matcher.get_matching_blocks()
    total_chars = max(len(ocr_clean), len(gt_clean))
    matched_chars = sum(block.size for block in matching_blocks)

    return {
        'error_rate': error_rate,
        'accuracy': similarity_ratio * 100,
        'matched_chars': matched_chars,
        'total_chars': total_chars,
        'matcher': matcher,
        'ocr_clean': ocr_clean,
        'gt_clean': gt_clean
    }

# ---------- Word-level Metrics (WER) ----------

def calculate_word_error_rate(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis texts
    WER = (S + D + I) / N
    """
    # Tokenize into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Create a matrix for dynamic programming (Levenshtein distance at word level)
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    # Initialize matrix
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]  # No cost for matching words
            else:
                substitution = d[i-1, j-1] + 1
                insertion = d[i, j-1] + 1
                deletion = d[i-1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    # Backtrack to find the operations
    i, j = len(ref_words), len(hyp_words)
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i, j] == d[i, j-1] + 1:
                insertions += 1
                j -= 1
            elif i > 0 and d[i, j] == d[i-1, j] + 1:
                deletions += 1
                i -= 1

    # Calculate WER
    total_words = len(ref_words)
    if total_words == 0:
        return {
            'wer': 0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'total_words': 0,
            'error_count': 0
        }

    wer = (substitutions + deletions + insertions) / total_words * 100

    return {
        'wer': wer,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'total_words': total_words,
        'error_count': substitutions + deletions + insertions
    }

# ---------- BLEU Score ----------

def calculate_bleu_score(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis texts
    BLEU is a precision-oriented metric that measures n-gram overlap
    """
    # Tokenize into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Handle empty edge cases
    if not ref_words or not hyp_words:
        return 0.0

    # Use smoothing function to handle short sentences
    smoothing = SmoothingFunction().method1

    # Calculate BLEU score (using 4-gram with smoothing)
    bleu_score = sentence_bleu([ref_words], hyp_words,
                              weights=(0.25, 0.25, 0.25, 0.25),  # 4-gram weights
                              smoothing_function=smoothing)

    return bleu_score * 100  # Convert to percentage

# ---------- ROUGE Scores ----------

def calculate_rouge_scores(reference, hypothesis):
    """
    Calculate ROUGE scores between reference and hypothesis texts
    ROUGE is a recall-oriented metric that measures n-gram overlap
    """
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate scores
    scores = scorer.score(reference, hypothesis)

    # Extract F1 scores (harmonic mean of precision and recall)
    rouge1 = scores['rouge1'].fmeasure * 100
    rouge2 = scores['rouge2'].fmeasure * 100
    rougeL = scores['rougeL'].fmeasure * 100

    return {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL
    }

# ---------- BERTScore ----------

def calculate_bertscore(reference, hypothesis):
    """
    Calculate BERTScore between reference and hypothesis texts
    BERTScore measures semantic similarity using contextual embeddings
    """
    # Calculate BERTScore
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)

    # Return F1 score (harmonic mean of precision and recall)
    return F1.item() * 100

# ---------- Word-level Differences ----------

def highlight_word_differences(reference, hypothesis):
    """
    Generate a text with highlights showing word-level differences
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Use SequenceMatcher for word-level comparison
    matcher = SequenceMatcher(None, ref_words, hyp_words)

    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result.extend(ref_words[i1:i2])
        elif tag == 'replace':
            # Substitution
            result.append(f"[SUB: {' '.join(ref_words[i1:i2])} -> {' '.join(hyp_words[j1:j2])}]")
        elif tag == 'delete':
            # Deletion
            result.append(f"[DEL: {' '.join(ref_words[i1:i2])}]")
        elif tag == 'insert':
            # Insertion
            result.append(f"[INS: {' '.join(hyp_words[j1:j2])}]")

    return ' '.join(result)

# ---------- Main Function ----------

def main():
    # File paths
    pdf_file = "a01-014u.pdf"  # Change to your PDF file
    gt_file = "a01-014u.docx"  # Change to your ground truth file
    output_ocr_file = "23138_easyocr_full.txt"
    output_report_file = "easyocr_complete_report.txt"

    # Perform OCR on the PDF using EasyOCR
    print(f"Performing OCR on {pdf_file} using EasyOCR...")
    ocr_pages = ocr_pdf(pdf_file)

    # Read ground truth from Word document
    print(f"Reading ground truth from {gt_file}...")
    ground_truth = read_docx(gt_file)

    # Save OCR results
    full_ocr_text = "".join(f"\n\n--- Page {i} ---\n{t}" for i, t in enumerate(ocr_pages, 1))
    with open(output_ocr_file, "w", encoding="utf-8") as f:
        f.write(full_ocr_text)
    print(f"\nSaved full OCR to {output_ocr_file}")

    print("\n" + "="*80)
    print("EASYOCR COMPLETE ACCURACY ANALYSIS REPORT")
    print("="*80)

    # Preprocess texts for metric calculations
    gt_clean = preprocess_text(ground_truth)

    # Calculate metrics for each page
    page_char_metrics = []
    page_wer_metrics = []
    page_bleu_scores = []
    page_rouge_scores = []
    page_bertscores = []

    for i, page_text in enumerate(ocr_pages, 1):
        # Character-level metrics
        char_metrics = calculate_character_metrics(page_text, ground_truth)
        page_char_metrics.append(char_metrics)

        # Word-level metrics (WER)
        ocr_clean = preprocess_text(page_text)
        wer_metrics = calculate_word_error_rate(gt_clean, ocr_clean)
        page_wer_metrics.append(wer_metrics)

        # BLEU score
        bleu_score = calculate_bleu_score(gt_clean, ocr_clean)
        page_bleu_scores.append(bleu_score)

        # ROUGE scores
        rouge_scores = calculate_rouge_scores(gt_clean, ocr_clean)
        page_rouge_scores.append(rouge_scores)

        # BERTScore
        bertscore = calculate_bertscore(gt_clean, ocr_clean)
        page_bertscores.append(bertscore)

        print(f"Page {i}:")
        print(f"  Character Error Rate = {char_metrics['error_rate']:.2f}%")
        print(f"  Word Error Rate (WER) = {wer_metrics['wer']:.2f}%")
        print(f"  BLEU Score = {bleu_score:.2f}%")
        print(f"  ROUGE-1 = {rouge_scores['rouge1']:.2f}%")
        print(f"  ROUGE-2 = {rouge_scores['rouge2']:.2f}%")
        print(f"  ROUGE-L = {rouge_scores['rougeL']:.2f}%")
        print(f"  BERTScore = {bertscore:.2f}%")
        print(f"  Errors: {wer_metrics['error_count']} words (S:{wer_metrics['substitutions']}, D:{wer_metrics['deletions']}, I:{wer_metrics['insertions']})")
        print(f"  Total words in reference: {wer_metrics['total_words']}")
        print()

    # Calculate overall metrics
    combined_ocr = " ".join(ocr_pages)
    combined_ocr_clean = preprocess_text(combined_ocr)

    overall_char_metrics = calculate_character_metrics(combined_ocr, ground_truth)
    overall_wer_metrics = calculate_word_error_rate(gt_clean, combined_ocr_clean)
    overall_bleu = calculate_bleu_score(gt_clean, combined_ocr_clean)
    overall_rouge = calculate_rouge_scores(gt_clean, combined_ocr_clean)
    overall_bertscore = calculate_bertscore(gt_clean, combined_ocr_clean)

    print(f"\nOVERALL RESULTS:")
    print(f"="*50)
    print(f"Character Error Rate = {overall_char_metrics['error_rate']:.2f}%")
    print(f"Word Error Rate (WER) = {overall_wer_metrics['wer']:.2f}%")
    print(f"BLEU Score = {overall_bleu:.2f}%")
    print(f"ROUGE-1 = {overall_rouge['rouge1']:.2f}%")
    print(f"ROUGE-2 = {overall_rouge['rouge2']:.2f}%")
    print(f"ROUGE-L = {overall_rouge['rougeL']:.2f}%")
    print(f"BERTScore = {overall_bertscore:.2f}%")
    print(f"Total Errors: {overall_wer_metrics['error_count']} words (S:{overall_wer_metrics['substitutions']}, D:{overall_wer_metrics['deletions']}, I:{overall_wer_metrics['insertions']})")
    print(f"Total words in reference: {overall_wer_metrics['total_words']}")
    print(f"Matched Characters: {overall_char_metrics['matched_chars']} / {overall_char_metrics['total_chars']}")

    # Show detailed differences for the first page
    print("\n" + "="*80)
    print("WORD-LEVEL DIFFERENCES (Page 1)")
    print("="*80)
    diff_text = highlight_word_differences(
        page_char_metrics[0]['gt_clean'],
        page_char_metrics[0]['ocr_clean']
    )
    print(diff_text[:1000] + "..." if len(diff_text) > 1000 else diff_text)

    # Save detailed comparison to file
    with open(output_report_file, "w", encoding="utf-8") as f:
        f.write("EASYOCR COMPLETE ACCURACY ANALYSIS REPORT\n")
        f.write("="*80 + "\n")

        f.write("\nCHARACTER-LEVEL METRICS:\n")
        f.write("-" * 50 + "\n")
        for i, metrics in enumerate(page_char_metrics, 1):
            f.write(f"Page {i}: Error Rate = {metrics['error_rate']:.2f}%, Accuracy = {metrics['accuracy']:.2f}%\n")

        f.write("\nWORD-LEVEL METRICS (WER):\n")
        f.write("-" * 50 + "\n")
        for i, metrics in enumerate(page_wer_metrics, 1):
            f.write(f"Page {i}: WER = {metrics['wer']:.2f}%, Errors: {metrics['error_count']} (S:{metrics['substitutions']}, D:{metrics['deletions']}, I:{metrics['insertions']})\n")

        f.write("\nBLEU SCORES:\n")
        f.write("-" * 50 + "\n")
        for i, score in enumerate(page_bleu_scores, 1):
            f.write(f"Page {i}: BLEU = {score:.2f}%\n")

        f.write("\nROUGE SCORES:\n")
        f.write("-" * 50 + "\n")
        for i, scores in enumerate(page_rouge_scores, 1):
            f.write(f"Page {i}: ROUGE-1 = {scores['rouge1']:.2f}%, ROUGE-2 = {scores['rouge2']:.2f}%, ROUGE-L = {scores['rougeL']:.2f}%\n")

        f.write("\nBERTSCORES:\n")
        f.write("-" * 50 + "\n")
        for i, score in enumerate(page_bertscores, 1):
            f.write(f"Page {i}: BERTScore = {score:.2f}%\n")

        f.write(f"\nOVERALL RESULTS:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Character Error Rate = {overall_char_metrics['error_rate']:.2f}%")
        f.write(f"Word Error Rate (WER) = {overall_wer_metrics['wer']:.2f}%\n")
        f.write(f"BLEU Score = {overall_bleu:.2f}%\n")
        f.write(f"ROUGE-1 = {overall_rouge['rouge1']:.2f}%\n")
        f.write(f"ROUGE-2 = {overall_rouge['rouge2']:.2f}%\n")
        f.write(f"ROUGE-L = {overall_rouge['rougeL']:.2f}%\n")
        f.write(f"BERTScore = {overall_bertscore:.2f}%\n")
        f.write(f"Total Errors: {overall_wer_metrics['error_count']} words (S:{overall_wer_metrics['substitutions']}, D:{overall_wer_metrics['deletions']}, I:{overall_wer_metrics['insertions']})")
        f.write(f"Total words in reference: {overall_wer_metrics['total_words']}\n")
        f.write(f"Matched Characters: {overall_char_metrics['matched_chars']} / {overall_char_metrics['total_chars']}\n\n")

        f.write("WORD-LEVEL DIFFERENCES\n")
        f.write("="*80 + "\n")
        for i, (char_metrics, wer_metrics) in enumerate(zip(page_char_metrics, page_wer_metrics), 1):
            f.write(f"\n--- Page {i} Differences ---\n")
            f.write(f"WER: {wer_metrics['wer']:.2f}%, Errors: {wer_metrics['error_count']}\n")
            diff = highlight_word_differences(char_metrics['gt_clean'], char_metrics['ocr_clean'])
            f.write(diff + "\n")

    print(f"\nSaved detailed comparison report to {output_report_file}")

if __name__ == "__main__":
    main()