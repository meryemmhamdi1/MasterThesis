from googletrans import Translator
import json
from tqdm import tqdm
translator = Translator()
print(translator.translate('bonjour').text)

print("Loading samples from json file")
with open('/Users/MeryemMhamdi/EPFL/Spring2018/Thesis/3AlgorithmsImplementation/demo/static/samples.json') as infile:
    json_samples = json.load(infile)
    
translated = []
for i in tqdm(range(400, len(json_samples['samples']))):
    doc = json_samples['samples'][i]
    translated.append(translator.translate(doc).text)
    
dict_ = {}
dict_['translations'] = json_samples['samples'][0:400] + translated

print("Dumping into json")
with open('/Users/MeryemMhamdi/EPFL/Spring2018/Thesis/3AlgorithmsImplementation/demo/static/translations.json', 'w') as outfile:
    json_samples.dump(dict_,outfile)