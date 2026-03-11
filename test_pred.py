import sys
sys.path.insert(0, 'src')

from pred import load_model, predict_file

model, tokenizer, inv_vocab, context_size = load_model('models/F4_L10_1.pt')
print('Modèle chargé OK')

result = predict_file(model, tokenizer, inv_vocab, context_size,
                      'data/GreenMIR/text_xml_nettoyer/article_11.txt')
print(result)
