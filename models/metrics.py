from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def score(reference: str, generated: str):
    """
    Compute BLEU score between a reference and a generated sentence.
    """
    refs = [reference.split()]
    gen = generated.split()
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(refs, gen, smoothing_function=smoothing)
    return bleu_score