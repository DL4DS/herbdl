import open_clip
from PIL import Image
import torch

model, preprocess, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

labels = ["Asteraceae Roldana cordovensis", "Fabaceae Astragalus sophoroides", "Cyperaceae Carex whitneyi"]

image = preprocess(Image.open("/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_images/126/40/12640__012.jpg")).unsqueeze(0)
text = tokenizer(labels)
print(text.shape)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", labels[text_probs.argmax().item()])