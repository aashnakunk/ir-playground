# Beam Search

A width-limited tree search. At each step, expand every candidate, then keep
only the top-`beam_width` by some score. The default decoding strategy for
many LLM-generation tasks.

## Intuition

Writing a sentence. Brainstorm 3 candidate next words. Continue each as its
own draft. After a few words, look at all the drafts, pick the 3 best, drop
the rest. Repeat. You'd never get the best sentence by always picking the
most obvious next word at each step.

## How it works

```
beams = [(start_sequence, log_prob=0)]
for step in range(max_steps):
    candidates = []
    for seq, lp in beams:
        for next_token, p in next_token_distribution(seq):
            candidates.append((seq + [next_token], lp + log(p)))
    beams = top_k(candidates, key=lp, k=beam_width)
return best(beams)
```

- **beam_width = 1** → greedy. Just pick the most likely next token each step.
- **beam_width = 5** → explore 5 parallel drafts.
- Higher width = better text quality but linearly more compute per step.

## Why log-probabilities?

You want to multiply probabilities along a sequence (joint probability), but
products of small numbers underflow. Adding logs of probabilities is
equivalent and numerically stable.

## When to use

- LLM decoding for tasks where quality matters more than diversity
  (translation, summarization, code completion).
- Sequence-to-sequence tasks generally.

## When not to use

- Creative / diverse generation. Beam search makes text repetitive and bland;
  use temperature sampling or top-p (nucleus) sampling instead.
- Real-time chat. Greedy + sampling is what production LLMs (ChatGPT, Claude,
  Gemini) actually use, because beam search is slower and users prefer
  variation.

## Where it shows up

- Classic NMT (Google Translate before Transformers).
- Constrained generation tasks where you want the most probable output.
- The conceptual root of many decoding strategies: nucleus sampling,
  contrastive decoding, speculative decoding.

## Trade-offs

- **Greedy (width 1)** can get stuck on a locally good first word that leads
  to a bad sentence.
- **Beam search** recovers from that by keeping alternatives alive — but
  collapses to repetitive text if the width is wide and the model has a
  high-probability boring continuation.

## Related

- [[BFS-DFS]] (beam search is essentially width-limited BFS)
- [[MCTS]] (alternative when you have a reward signal but no good next-token
  probability)
