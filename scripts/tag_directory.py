import os
from pathlib import Path
from tqdm import tqdm
import paths
import settings
from dataset_tag_editor.tagger import Tagger
from PIL import Image
from dataset_tag_editor.dte_logic import INTERROGATORS

dirpath = Path("/tmp/dataset")

settings.load()
paths.initialize()

threshold = 0.5


EXCLUDE = ["blip2-flan-t5-xxl"] 
INCLUDE = []

def get_models():
    models = []
    for model in INTERROGATORS:
        if len(INCLUDE) > 0:
            if model.name() in INCLUDE:
                models.append(model)
        else:
            if model.name() not in EXCLUDE:
                models.append(model)
    return models

models = get_models()

image_tags = {}
for model in tqdm(models, "models"):
    print(f"* loading {model.name()}")
    try:
        model.start()
        for filename in tqdm([f for f in os.listdir(dirpath) if not f.endswith(".txt")], "images"):
            filepath = dirpath / filename
            image = Image.open(filepath).convert("RGB")
            if filename not in image_tags.keys():
                image_tags[filename] = set()
            
            # generate tags
            if isinstance(model, Tagger):
                predict = model.predict(image, threshold=threshold)
            else:
                predict = model.predict(image)
            
            def clean_tag(tag):
                return str(tag).rstrip("\n").rstrip(".").rstrip(",").rstrip(" ")

            # save tags
            if  isinstance(predict, str):
                image_tags[filename].add(clean_tag(predict))
            elif isinstance(predict, list):
                for i in predict:
                    image_tags[filename].add(clean_tag(i))
            elif isinstance(predict, dict):
                for i in predict.keys():
                    image_tags[filename].add(clean_tag(i))
            else:
                print(f"error type: {predict}")

            
            image.close()
        print(f"+ success processing with model {model.name()}")
    except Exception as e:
        print(f"- error processing with model {model.name()}: {e}")
    finally:
        model.stop()

for filename in os.listdir(dirpath):
    if filename not in image_tags.keys():
        continue
    filepath = dirpath / filename
    filepath = str(filepath).split(".")[0] + ".txt"
    with open(filepath, "w") as f:
        f.writelines([", ".join(image_tags[filename])])
    print(f"writen {filename}")

