from io import BytesIO
from lostpaw.config.config import TrainConfig
from lostpaw.data.extract_pets import DetrPetExtractor
from lostpaw.model import PetViTContrastiveModel
from PIL import Image
from torch import Tensor
import yaml
import numpy as np
import torch.nn.functional as F
import torch
import os
import random

from lostpaw.model.trainer import Trainer

with open("lostpaw/configs/container.yaml", "r") as f:
    config = TrainConfig(**yaml.safe_load(f))
    print(config)

trainer = Trainer(config)
model = trainer.vit_model
model.train(False)
extractor = DetrPetExtractor(config.model_path)

def save_image_embedding(image_path, dog_images, dog_id):
    image = Image.open(image_path).convert('RGB')
    extracted_pets = extractor.extract([image], [0], output_size=(384, 384))
    if not extracted_pets:
        print("❌ No dog detected in input image.")
        return False
    feature_tensor = model(extracted_pets[0][0])
    embedding = feature_tensor.view(-1)

    # Ensure both tensors are on the same device (e.g., CPU)
    embedding = embedding.to("cpu")
    dog_images.append( (dog_id, torch.tensor(embedding)) )
    return True


def search_similar_images_test(image_path, dog_id, dog_images, function, top_k=10):
    image = Image.open(image_path).convert('RGB')
    extracted_pets = extractor.extract([image], [0], output_size=(384, 384))
    if not extracted_pets:
        print("❌ No dog detected in input image.")
        return False
    feature_tensor = model(extracted_pets[0][0])
    embedding = feature_tensor.view(-1)
    embedding = embedding.to("cpu")
    ids_with_scores = {}
    for element in dog_images:
        if function == 0:
            # cosine distance
            score = 1-F.cosine_similarity(embedding.unsqueeze(0), element[1].unsqueeze(0))
        elif function == 1:
            # Euclidean distance
            score = torch.norm(embedding - element[1], p=2)
        ids_with_scores[element[0]] = score.item()

    # sort scors in ascending order and check if dogID is present in first topk results
    ids_with_scores = dict(sorted(ids_with_scores.items(), key=lambda item: item[1]))
    dict_keys = list(ids_with_scores.keys())
    for i in range(top_k):
        if dict_keys[i] == dog_id:
            return True
    return False

def main():
    while True:
        print("\n==============================")
        print("🐶 Dog Image Search Terminal App")
        print("==============================")
        print("1. Test cosine distance")
        print("2. Test L2 distance")
        print("3. Exit")
        choice = int(input("Enter your choice (1/2/3): ").strip())
        if choice == 3:
            print("👋 Goodbye!")
            break
        directory_path = input("Enter the path to the directory containing dog images: ").strip()
        numberOfDogs = int(input("How many dogs to test: ").strip())
        top_k = int(input("Top k parameter: ").strip())
        
        if os.path.isdir(directory_path):
            i = 0
            while i < 10:
                i += 1
                dog_images = []
                if choice == 1:
                    filename = f"lostpawTransformer-cosDist-{numberOfDogs}-top{top_k}.csv"
                elif choice == 2:
                    filename = f"lostpawTransformer-L2Dist-{numberOfDogs}-top{top_k}.csv"
                with open(filename, "a") as file:
                    dirsOk = []
                    validDirs = 0
                    success = 0
                    fail = []
                    dogDirs = os.listdir(directory_path)
                    while len(dirsOk) != numberOfDogs:
                        dir = random.choice(dogDirs)
                        if dir in dirsOk:
                            continue
                        dirFullPath = f"{directory_path}/{dir}/"
                        imagePaths = random.sample(os.listdir(dirFullPath), 2)
                        validImages = True
                        for path in imagePaths:
                            image = Image.open(f"{dirFullPath}/{path}").convert('RGB')
                            extracted_pets = extractor.extract([image], [0], output_size=(384, 384))
                            if not extracted_pets:
                                print("❌ No dog detected in input image.")
                                validImages = False
                        if validImages == False:
                            continue
                        firstImage = f"{dirFullPath}{imagePaths[0]}"
                        result = save_image_embedding(firstImage, dog_images, dir)
                        if result == True:
                            dirsOk.append((dir, imagePaths[1]))
                            print(len(dirsOk))
                    for idx, dir in enumerate(dirsOk):
                        print(idx)
                        dirFullPath = f"{directory_path}/{dir[0]}/"
                        secondImage = f"{dirFullPath}{dir[1]}"
                        result = search_similar_images_test(secondImage, dir[0], dog_images, choice-1, top_k=top_k)
                        if result == None:
                            print("\tSecond image doesnt contain a dog!")
                            continue
                        if result == True:
                            success += 1
                        if result == False:
                            fail.append(dir[0])
                        validDirs += 1
                    print(f"Valid dirs: {validDirs}")
                    print(f"Success dirs: {success}")
                    file.write(f"{success},{validDirs},{(success/validDirs) * 100},{' '.join(fail)}\n")
                    print(f"Accuracy: {(success/validDirs) * 100} %")
                    print(f"Failed: {fail}")
        else:
            print("❌ Invalid directory path.")
          

if __name__ == "__main__":
    main()



# extractor = DetrPetExtractor(config.model_path)
# image1 = Image.open("img0.jpg", formats=["JPEG", "PNG"])
# image2 = Image.open("img2.jpg", formats=["JPEG", "PNG"])
# image1.load()
# image2.load()
# image1 = image1.convert("RGB")
# image2 = image2.convert("RGB")
# extracted_pets1 = extractor.extract([image1], [0], output_size=(384, 384))
# extracted_pets2 = extractor.extract([image2], [0], output_size=(384, 384))
# extracted_pets1[0][0].save("cropped.jpg")
# feature_tensor1 = model(extracted_pets1[0][0])
# feature_tensor2 = model(extracted_pets2[0][0])
# embedding1 = feature_tensor1.view(-1)
# embedding2 = feature_tensor2.view(-1)

# # Ensure both tensors are on the same device (e.g., CPU)
# embedding1 = embedding1.to("cuda:0")
# embedding2 = embedding2.to("cuda:0")

# euclidean_distance = torch.norm(embedding1 - embedding2, p=2)
# print(euclidean_distance.item())
# similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
# print(similarity.item())  # similarity is a single-element tensor, so get the scalar value
