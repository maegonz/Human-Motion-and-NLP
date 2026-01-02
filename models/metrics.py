from nltk.translate.bleu_score import sentence_bleu

def score(reference: str, generated: str):
    """
    Compute BLEU score between a reference and a generated sentence.
    """
    refs = [ref.split(' ') for ref in reference]
    gen = generated.split(' ')
    bleu_score = sentence_bleu(refs, gen)
    return bleu_score