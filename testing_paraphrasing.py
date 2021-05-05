import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message):
        """
        Prints error message and help.
        :param message: error message to print
        """
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Main method
    """
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage()
    parser.description = "Paraphrase with T5"
    parser.add_argument("--input", help="Input file", required=True, type=Path)
    parser.add_argument("--output", help="Output file", required=True, type=Path)

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    logging.info("STARTED")

    statements = []
    with input_file.open(newline='') as csvfile:
        for line in csvfile:
            row = json.loads(line)
            statements.append(row)
    print(statements)
    exit()
    
    set_seed(42)

    model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')
    tokenizer = T5Tokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)
    model = model.to(device)
    max_len = 256

    mod_rows = []
    with output_file.open(mode='w') as csvfile:
        for idx, r in enumerate(statements):
            logging.info(f"Statement {idx}")
            statement = r["claim"]

            text = "paraphrase: " + statement
            encoding = tokenizer.encode_plus(text, padding=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(device), encoding[
                "attention_mask"].to(device)

            beam_outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                do_sample=True,
                max_length=max_len,
                top_k=120,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=10
            )


            final_outputs = []
            filter_bag = {statement.lower()}
            for beam_output in beam_outputs:
                sent = tokenizer.decode(beam_output, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
                if sent.lower() != statement.lower() and sent.lower() not in filter_bag:
                    final_outputs.append(sent)
                    filter_bag.add(sent.lower())
                if len(final_outputs) == 3:
                    break

            logging.info("Oringinal:")
            logging.info(statement)
            logging.info("Paraphrases:")
            for i, final_output in enumerate(final_outputs):
                logging.info(final_output)
                new_r = r
                new_r["claim"] = final_output
                mod_rows.append(new_r)
                csvfile.write(json.dumps(r))
                csvfile.write("\n")

            logging.info("")
            logging.info("")
            # if idx == 2:
            #     break

    # with output_file.open(mode='w') as csvfile:
    #     for r in mod_rows:
    #         csvfile.write(json.dumps(r))
    #         csvfile.write("\n")

    logging.info("DONE")


if __name__ == "__main__":
    main()