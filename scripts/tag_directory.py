import os
from pathlib import Path
from tqdm import tqdm
import paths
import settings
from dataset_tag_editor.tagger import Tagger
from dataset_tag_editor.captioning import Captioning
from PIL import Image
from dataset_tag_editor.dte_logic import INTERROGATORS
from typing import List, Dict

source_dirpath = Path("/srv/data/traning_data/dataset/source/unsplash/data/")
target_dirpath = Path("/tmp/dataset")
threshold = 0.5
keep_existing_tags = True
target_size = 768

EXCLUDE = ["blip2-flan-t5-xxl"] 
INCLUDE = []

settings.load()
paths.initialize()


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

banned_tags = [
    "unused"
]

tag_translations = {
    "rating:safe": "sfw",
}

image_tags = {}


def clean_tag(tag) -> str|None:
    if not tag:
        return None
    return str(tag).rstrip("\n").rstrip(".").rstrip(",").rstrip(" ")

def filter_tags(predict: List[str]|Dict|str) -> List[str]:
    tags = []
    if isinstance(predict, str):
        tags = [predict]
    elif isinstance(predict, dict):
        tags = list(predict.keys())
    elif isinstance(predict, list):
        tags = predict
    else:
        return []

    #return [clean_tag(tag) for tag in tags if clean_tag(tag) != "" and clean_tag(tag) != None]
    for tag in tags:
        tag = clean_tag(tag)
        if tag == "" or tag == None:
            continue
        for banned in banned_tags:
            if banned in tag:
                continue
        if tag in tag_translations.keys():
            tag = tag_translations[tag]
        yield tag

def resize_image(image:Image.Image, size:int) -> Image.Image:
    if image.width > image.height:
        return image.resize((size, int(image.height * (size / image.width))), Image.Resampling.LANCZOS)
    else:
        return image.resize((int(image.width * (size / image.height)), size), Image.Resampling.LANCZOS)

def change_extension(filename:str|Path, extension:str) -> str:
    return os.path.splitext(filename)[0] + extension
    # return ".".join(str(filename).split(".")[:-1]) + extension


def process_image(model:Tagger|Captioning, filename:str, source_dir:str|Path, target_dir:str|Path):
    filepath = Path(source_dir) / filename
    image = Image.open(filepath).convert("RGB")
    
    # generate tags
    if isinstance(model, Tagger):
        predict = model.predict(image, threshold=threshold)
    else:
        predict = model.predict(image)
    

    # save tags
    if filename not in image_tags.keys():
        image_tags[filename] = set()

    if keep_existing_tags:
        existing_tagfile = change_extension(filepath, ".txt")
        if os.path.exists(existing_tagfile):
            with open(existing_tagfile, "r") as f:
                for line in f.readlines():
                    for tag in filter_tags([i.strip() for i in line.split(",")]):
                        image_tags[filename].add(tag)
    
    for i in filter_tags(predict):
        image_tags[filename].add(i)
    target_path = Path(target_dir) / filename

    if not os.path.exists(target_path):
        new_image = resize_image(image, target_size)
        new_image.save(Path(target_dir) / filename)
        new_image.close()
    
    image.close()
        
for model in tqdm(models, "models"):
    print(f"* loading {model.name()}")
    try:
        model.start()
        for filename in tqdm([f for f in os.listdir(source_dirpath) if not f.endswith(".txt")], "process images"):
            try:
                process_image(model, filename, source_dirpath, target_dirpath)
            except Exception as e:
                print(f"! error processing image {filename}: {e}")
        print(f"+ success processing with model {model.name()}")
    except Exception as e:
        print(f"- error processing with model {model.name()}: {e}")
    finally:
        model.stop()

for filename in tqdm(list(image_tags.keys()), "writing tags"):
    if not os.path.exists(target_dirpath):
        os.makedirs(target_dirpath, exist_ok=True)
    filepath = target_dirpath / filename
    filepath = change_extension(filepath, ".txt")
    with open(filepath, "w") as f:
        f.writelines([", ".join(image_tags[filename])])
    print(f"writen {filepath}")

