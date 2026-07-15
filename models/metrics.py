import torch
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)

# def bleu(reference: str, generated: str):
#     """
#     Compute BLEU score between a reference and a generated sentence.
#     """
#     refs = [reference.split()]
#     gen = generated.split()
#     smoothing = SmoothingFunction().method1
#     bleu_score = sentence_bleu(refs, gen, smoothing_function=smoothing)
#     return bleu_score


# rouge_scorer_ = rouge_scorer.RougeScorer(
#     ['rouge1', 'rouge2', 'rougeL'],
#     use_stemmer=True
# )
# def rouge(reference: str, generated: str):
#     """
#     Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between a reference and a generated sentence.
#     """
#     scores = rouge_scorer_.score(reference, generated)
#     return {
#         'rouge1': scores['rouge1'].fmeasure,
#         'rouge2': scores['rouge2'].fmeasure,
#         'rougeL': scores['rougeL'].fmeasure,
#     }


# def meteor(reference: str, generated: str):
#     """
#     Compute METEOR score between a reference and a generated sentence.
#     """
#     return meteor_score([reference.split()], generated.split())


# cider_scorer = Cider()
# def cider(reference: str, generated: str):
#     """
#     Compute CIDEr score between a reference and a generated sentence.
#     """
#     refs = {0: [reference]}
#     gens = {0: [generated]}
#     score, _ = cider_scorer.compute_score(refs, gens)
#     return score


# def scores(reference: str, generated: str):
#     """
#     Compute BLEU, ROUGE, METEOR, and CIDEr scores between a reference and a generated sentence.
#     """
#     scores = {}
#     scores['bleu'] = bleu(reference, generated)
#     scores['meteor'] = meteor(reference, generated)
#     scores.update(rouge(reference, generated))
#     scores['cider'] = cider(reference, generated)
#     return scores

def contrastive_loss(motion_features, text_features, temperature=0.07):
    """
    motion_features: [batch, dim] (Mean-pooled output of MotionAdapter)
    text_features: [batch, dim] (Mean-pooled hidden states of T5 embeddings)
    """
    # Normalize to get cosine similarity
    motion_features = F.normalize(motion_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Calculate logits (batch_size, batch_size)
    logits = torch.matmul(motion_features, text_features.T) / temperature
    
    # Labels are diagonal (each motion matches its own text)
    labels = torch.arange(motion_features.size(0)).to(motion_features.device)
    
    loss_m = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_m + loss_t) / 2


class Evaluator:
    def __init__(self):
        self.rouge_scorer_ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.cider_scorer = Cider()
        self.smoothing = SmoothingFunction().method1

    def compute_metrics(self, references: list[str], generated: list[str]):
        """
        Compute corpus-level metrics for lists of references and generated texts.
        references: list of ground-truth strings
        generated: list of generated strings
        """
        assert len(references) == len(generated), "Must have equal number of refs and gens"
        
        # 1. BLEU (Corpus Level)
        # Corpus bleu expects: [[['ref1_word1', 'ref1_word2']], [['ref2_word1']]]
        refs_bleu = [[ref.split()] for ref in references]
        gens_bleu = [gen.split() for gen in generated]
        bleu_score = corpus_bleu(refs_bleu, gens_bleu, smoothing_function=self.smoothing)

        # 2. CIDEr (Corpus Level)
        # CIDEr expects dicts mapping ID to list of strings: {0: ["ref1"], 1: ["ref2"]}
        refs_cider = {i: [ref] for i, ref in enumerate(references)}
        gens_cider = {i: [gen] for i, gen in enumerate(generated)}
        cider_score, _ = self.cider_scorer.compute_score(refs_cider, gens_cider)

        # 3. METEOR (Average of sentence scores)
        # NLTK meteor_score expects tokenized lists
        meteor_scores = [
            meteor_score([ref.split()], gen.split()) 
            for ref, gen in zip(references, generated)
        ]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        # 4. ROUGE (Average of sentence scores)
        rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
        for ref, gen in zip(references, generated):
            scores = self.rouge_scorer_.score(ref, gen)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure
            
        n = len(references)

        return {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
            'ROUGE-1': rouge1 / n,
            'ROUGE-2': rouge2 / n,
            'ROUGE-L': rougeL / n,
            'CIDEr': cider_score
        }