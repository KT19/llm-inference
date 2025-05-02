import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from .MCTSNode import MCTSNode


class MCTSLM:
    def __init__(
        self,
        model_name: str,
        select_k: int = 5,
        num_simulations: int = 10,
        max_depth: int = 20,
        temperature: float = 0.8,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.select_k = select_k
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.temperature = temperature

    def generate(
        self, prompt: str, n_tokens: int = 1, ignore_input_prompt: bool = False
    ) -> str:
        cur_tokens = 0
        res = """"""

        while cur_tokens < n_tokens:
            (gen_tokens, generated) = self.search(
                prompt=prompt + res, n_tokens=n_tokens
            )
            if gen_tokens == 0:
                break
            cur_tokens += gen_tokens
            res += generated

        res = " ".join(res.split()[:n_tokens])
        if ignore_input_prompt:
            return res

        return prompt + " " + res

    def search(self, prompt: str, n_tokens: int = 1) -> tuple[int, str]:
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Create root node
        root = MCTSNode()

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            current_sequence = input_ids.clone()
            depth = 0

            # Selection phase
            while not node.is_leaf() and not node.is_terminal(self.max_depth, depth):
                next_node = node.select_child()
                if next_node is None:
                    break
                node = next_node
                current_sequence = torch.cat(
                    [
                        current_sequence,
                        torch.tensor([[node.action]], dtype=current_sequence.dtype),
                    ],
                    dim=1,
                )
                depth += 1

            # Expansion phase
            if not node.is_terminal(self.max_depth, depth):
                # Get model predictions for next token
                with torch.no_grad():
                    outputs = self.model(current_sequence)
                    logits = outputs.logits[:, -1, :]
                    probs = (
                        torch.softmax(logits / self.temperature, dim=-1)
                        .to("cpu")
                        .numpy()
                    )

                # Sample select k
                select_k = min(
                    self.select_k, probs.shape[-1]
                )  # Limit expanstion to top-k tokens

                candidate_indices = random.choices(
                    range(probs.shape[-1]), weights=probs[0], k=select_k
                )

                # Create child nodes for each potential token
                for token_id in candidate_indices:
                    if token_id not in node.children:
                        prior = probs[0][token_id].item()
                        node.children[token_id] = MCTSNode(
                            parent=node, action=token_id, prior=prior
                        )

            # If we can expand further, select a child for simulation
            if not node.is_terminal(self.max_depth, depth) and node.children:
                # Choose child with highest prior for simulation
                action = max(node.children.items(), key=lambda x: x[1].prior)[0]
                node = node.children[action]
                current_sequence = torch.cat(
                    [
                        current_sequence,
                        torch.tensor([[node.action]], dtype=current_sequence.dtype),
                    ],
                    dim=1,
                )
                depth += 1

            # rollout
            value = self._evaluate_sequence(current_sequence)

            # backpropagation
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent

        # Select best action from root based on visit count
        best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]

        # Generate the final output by following the most visited path
        # pick the root
        node = root.children[best_action]
        generated_ids = torch.tensor([[best_action]], dtype=input_ids.dtype)

        # generate remaining tokens
        for _ in range(n_tokens - 1):
            if not node.children:
                break

            next_node = max(node.children.items(), key=lambda x: x[1].visits)[1]
            generated_ids = torch.cat(
                [
                    generated_ids,
                    torch.tensor([[next_node.action]], dtype=generated_ids.dtype),
                ],
                dim=1,
            )
            node = next_node

        return (
            len(generated_ids[0]),
            self.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
        )

    def _evaluate_sequence(self, sequence: torch.Tensor) -> float:
        """
        Run a quick forward pass and evaluate the quality of a sequence using a value function
        """
        # A simple measure
        with torch.no_grad():
            outputs = self.model(sequence, labels=sequence)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return -perplexity.item()
