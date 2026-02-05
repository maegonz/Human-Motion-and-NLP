import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

nltk.download('omw-1.4')
nltk.download('wordnet')

def bleu(reference: str, generated: str):
    """
    Compute BLEU score between a reference and a generated sentence.
    """
    refs = [reference.split()]
    gen = generated.split()
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(refs, gen, smoothing_function=smoothing)
    return bleu_score


rouge_scorer_ = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)
def rouge(reference: str, generated: str):
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between a reference and a generated sentence.
    """
    scores = rouge_scorer_.score(reference, generated)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def meteor(reference: str, generated: str):
    """
    Compute METEOR score between a reference and a generated sentence.
    """
    return meteor_score([reference.split()], generated.split())


cider_scorer = Cider()
def cider(reference: str, generated: str):
    """
    Compute CIDEr score between a reference and a generated sentence.
    """
    refs = {0: [reference]}
    gens = {0: [generated]}
    score, _ = cider_scorer.compute_score(refs, gens)
    return score


def scores(reference: str, generated: str):
    """
    Compute BLEU, ROUGE, METEOR, and CIDEr scores between a reference and a generated sentence.
    """
    scores = {}
    scores['bleu'] = bleu(reference, generated)
    scores['meteor'] = meteor(reference, generated)
    scores.update(rouge(reference, generated))
    scores['cider'] = cider(reference, generated)
    return scores
