from transformers import pipeline
gender_classifier = pipeline(model="/mnt1/daoyuan_mm/pedestrian_gender_recognition")
image_path = "abc.jpg"

results = gender_classifier(image_path)
print(results)
