import click
from modules.MCTSLM import MCTSLM
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


@click.command()
@click.option("--n_tokens", type=int, default=10)
@click.option("--select_k", type=int, default=5)
@click.option("--num_simulations", type=int, default=10)
def main(n_tokens: int, select_k: int, num_simulations: int):
    model_name = "gpt2"
    my_lm = MCTSLM(
        model_name=model_name, select_k=select_k, num_simulations=num_simulations
    )

    # prompt
    prompt = (
        "The future of artificial intelligence"  # input("Please enter a prompt ->")
    )

    print(f"\nThe prompt: {prompt}")
    print("Generating with MCTS...")

    generated_text = my_lm.generate(prompt=prompt, n_tokens=n_tokens)

    print("Generated text:")
    print(generated_text)

    # Compare with standard generation (without MCTS)
    print("\nComparing with standard generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt")
    standard_output = model.generate(
        input_ids["input_ids"],
        attention_mask=input_ids["attention_mask"],
        max_new_tokens=n_tokens,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    standard_text = tokenizer.decode(standard_output[0], skip_special_tokens=True)
    print("Standard generation:")
    print(standard_text)


if __name__ == "__main__":
    main()
