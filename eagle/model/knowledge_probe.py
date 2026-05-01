# eagle/model/knowledge_probe.py
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class KnowledgeProbe(nn.Module):
    """
    Probe that performs:
      - NER (BIO tagging) on each token
      - Relation classification between entity spans
    """
    def __init__(self, hidden_dim: int, num_entity_types: int, num_relations: int):
        super().__init__()
        self.ner_head = nn.Linear(hidden_dim, num_entity_types * 3 + 1)  # BIO tags
        self.rel_head = nn.Linear(hidden_dim * 2, num_relations)         # subject+object embedding

    def forward_ner(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (seq_len, hidden_dim)
        return self.ner_head(hidden_states)

    def forward_relation(self, subj_rep: torch.Tensor, obj_rep: torch.Tensor) -> torch.Tensor:
        # subj_rep, obj_rep: (batch, hidden_dim)
        combined = torch.cat([subj_rep, obj_rep], dim=-1)
        return self.rel_head(combined)


class KnowledgeExtractor:
    """
    Wraps the trained probes and provides a method to decode token sequences + hidden states into triples.
    """
    def __init__(self, probe_model: KnowledgeProbe, tokenizer, id2label: dict, id2relation: dict):
        self.probe = probe_model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.id2relation = id2relation

    def extract_triples(
        self,
        token_ids: List[int],
        hidden_states: torch.Tensor   # shape (seq_len, hidden_dim)
    ) -> List[Tuple[str, str, str]]:
        """
        Decode NER tags from hidden_states, then for every pair of entity spans,
        predict a relation and return (subject, relation, object) triples.
        """
        if hidden_states.shape[0] != len(token_ids):
            # align if mismatch (e.g., from padding)
            hidden_states = hidden_states[:len(token_ids)]

        logits = self.probe.forward_ner(hidden_states)  # (seq_len, num_labels)
        preds = logits.argmax(dim=-1).tolist()

        # Simple span aggregation (BIO)
        entities = []
        current_ent = []
        current_type = None
        for idx, (tid, tag) in enumerate(zip(token_ids, preds)):
            label = self.id2label[tag]
            if label.startswith("B-"):
                if current_ent:
                    entities.append((current_ent, current_type))
                current_ent = [idx]
                current_type = label[2:]
            elif label.startswith("I-") and current_ent:
                current_ent.append(idx)
            else:
                if current_ent:
                    entities.append((current_ent, current_type))
                current_ent = []
                current_type = None
        if current_ent:
            entities.append((current_ent, current_type))

        triples = []
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                subj_indices, subj_type = entities[i]
                obj_indices, obj_type = entities[j]
                # use mean pooling of token hidden states as entity representation
                subj_repr = hidden_states[subj_indices].mean(dim=0, keepdim=True)
                obj_repr = hidden_states[obj_indices].mean(dim=0, keepdim=True)
                rel_logits = self.probe.forward_relation(subj_repr, obj_repr)
                rel_id = rel_logits.argmax(dim=-1).item()
                rel_label = self.id2relation[rel_id]

                # only keep non-null relations
                if rel_label != "no_relation":
                    subj_text = self.tokenizer.decode([token_ids[i] for i in subj_indices])
                    obj_text = self.tokenizer.decode([token_ids[i] for i in obj_indices])
                    triples.append((subj_text, rel_label, obj_text))
        return triples
