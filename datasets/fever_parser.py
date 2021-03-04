import jsonlines
import csv
import os
from tqdm import tqdm

def read_lines(text):
	return [line for line in text.split("\n")]


def read_text(text):
    return [line.split('\t')[1] if len(line.split('\t'))>1 else "" for line in read_lines(text)]

	
def parse_fever_corpus_obj_to_scifact_corpus_obj(fever_obj, index):
	fever_id = fever_obj['id']
	abstract = read_text(fever_obj['lines'])
	res =  {
		"id": index,
		"abstract": abstract,
	}

	return fever_id, res


def main():
	fever_corpus = "fever/wiki-pages/"
	output = 'fever/fever-corpus-as-scifact.jsonl'

	fever_ids = []
	with jsonlines.open(output, 'w') as out:
		for file in tqdm(os.listdir(fever_corpus)):
			with jsonlines.open(fever_corpus + file) as fever:
				for ind, line in enumerate(fever):
					fever_id, scifact_obj = parse_fever_corpus_obj_to_scifact_corpus_obj(line, ind + 1)
					out.write(scifact_obj)
					fever_ids.append(fever_id)

	with open('id_map.csv', 'w') as fever_id_csv:
		writer = csv.writer(fever_id_csv)
		writer.writerow(fever_ids)
	
	print("done")



if __name__ == '__main__':
	main()
		



# {
# 	"id": "1930_Staten_Island_Stapletons_season", 
# 	"text": "The 1930 Staten Island Stapletons season was their second in the league . The team improved on their previous output of 3 -- 4 -- 3 , winning five games . They finished sixth in the league . ", 
	# [
	# 	"0 The 1930 Staten Island Stapletons season was their second in the league . 1930 1930 NFL season Staten Island Stapletons Staten Island Stapletons league National Football League",
	# 	"1 The team improved on their previous output of 3 -- 4 -- 3 , winning five games . previous output 1929 Staten Island Stapletons season
	# 	"2 They finished sixth in the league . league National Football League",
	# ]
# }

# {
# 	"doc_id": 4983, 
# 	"title": "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.", 
# 	"abstract": [
# 		"Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities.", 
# 		"A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in preterm (n = 17) and full-term infants (n = 7).", 
# 		"To assess effects of prematurity on cerebral white matter development, early gestation preterm infants (n = 10) were studied a second time at term.", 
# 		"In the central white matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to 1.2 microm2/ms.", 
# 		"In the posterior limb of the internal capsule, the mean apparent diffusion coefficients at both times were similar (1.2 versus 1.1 microm2/ms).", 
# 		"Relative anisotropy was higher the closer birth was to term with greater absolute values in the internal capsule than in the central white matter.", 
# 		"Preterm infants at term showed higher mean diffusion coefficients in the central white matter (1.4 +/-", 
# 		"0.24 versus 1.15 +/-", 
# 		"0.09 microm2/ms, p = 0.016) and lower relative anisotropy in both areas compared with full-term infants (white matter, 10.9 +/- 0.6 versus 22.9 +/-", 
# 		"3.0%, p = 0.001; internal capsule, 24.0 +/-", 
# 		"4.44 versus 33.1 +/-", 
# 		"0.6% p = 0.006).", 
# 		"Nonmyelinated fibers in the corpus callosum were visible by diffusion tensor MRI as early as 28 wk; full-term and preterm infants at term showed marked differences in white matter fiber organization.", 
# 		"The data indicate that quantitative assessment of water diffusion by diffusion tensor MRI provides insight into microstructural development in cerebral white matter in living infants."
# 	],
# 	"structured": false
# }

# Fever: {
#   "id": 129629, 
#   "verifiable": "VERIFIABLE", 
#   "label": "SUPPORTS", 
#	"claim": "Homeland is an American television spy thriller based on the Israeli television series Prisoners of War.", 
# 	"evidence": [
# 		[
# 			[151831, 166598, "Homeland_-LRB-TV_series-RRB-", 0],
# 			[151831, 166598, "Prisoners_of_War_-LRB-TV_series-RRB-", 0]
# 		]
# 	]
# }

#{
# 	"id": 3, 
# 	"claim": "1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.", 
# 	"evidence": {
# 		"14717500": [
# 			{"sentences": [2, 5], "label": "SUPPORT"}, 
# 			{"sentences": [7], "label": "SUPPORT"}
# 		]
# 	}, 
# 	"cited_doc_ids": [14717500]
# }